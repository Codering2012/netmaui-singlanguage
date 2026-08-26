import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
import sys

def align_and_normalize_hand_3d(world_landmarks):
    """
    Translates, rotates, and scales 3D world landmarks to be camera, scale, and orientation invariant.
    
    Args:
        world_landmarks: A numpy array of shape (21, 3) representing the x, y, z world coordinates.
        
    Returns:
        normalized_coords: A numpy array of shape (21, 3) representing the aligned coordinates.
        rotation_matrix: The 3x3 rotation matrix used for alignment.
        scale: The scale factor used for normalization.
    """
    # 1. Translation: Move Wrist (landmark 0) to origin (0, 0, 0)
    wrist = world_landmarks[0]
    translated = world_landmarks - wrist
    
    # 2. Scaling: Calculate standard hand scale (distance from Wrist to Middle Finger MCP joint - landmark 9)
    # Landmark 9 is the middle finger MCP, which is a stable anchor point.
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    
    if scale < 1e-5:
        scale = 1.0
        
    scaled = translated / scale
    
    # 3. Rotation (Alignment):
    # Align the hand so:
    # - The vector from Wrist (0) to Middle Finger MCP (9) points straight UP (+Y axis)
    # - The palm normal vector points forward (+Z axis)
    # - The thumb-side/pinky-side lateral vector points along the +X axis
    
    # Vector representing Hand Direction (Wrist to Middle MCP)
    v_up = scaled[9]  # Since Wrist is at (0,0,0)
    u_up = v_up / np.linalg.norm(v_up)
    
    # Vector from Wrist to Index Finger MCP (landmark 5) to establish the hand plane
    v_index = scaled[5]
    
    # Palm normal is perpendicular to both u_up and v_index (Cross product)
    v_normal = np.cross(u_up, v_index)
    u_normal = v_normal / np.linalg.norm(v_normal)
    
    # Lateral vector is perpendicular to normal and up (completing the orthonormal basis)
    u_side = np.cross(u_normal, u_up)
    
    # Rotation matrix mapping world coords to our new canonical local frame
    # Rotation matrix R is composed of [u_side, u_up, u_normal] as columns
    R = np.vstack([u_side, u_up, u_normal])  # Shape (3, 3)
    
    # Apply rotation to all coordinates: x_local = R * x_world
    aligned = np.dot(scaled, R.T)
    
    return aligned, R, scale

def extract_from_dataset(dataset_dir, output_dir):
    """
    Traverses the dataset directory, extracts normalized 3D landmarks, and saves them.
    """
    print(f"Initializing MediaPipe Hand Landmarker...")
    mp_hands = mp.solutions.hands
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare directories for saving outputs
    coords_dir = output_path / "coords"
    coords_dir.mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    # Initialize MediaPipe Hands in static image mode
    # We set static_image_mode=True since the dataset consists of still images.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        
        # Gather all image files
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(dataset_path.rglob(f"*{ext}")))
            all_images.extend(list(dataset_path.rglob(f"*{ext.upper()}")))
            
        print(f"Found {len(all_images)} images to process.")
        
        results_summary = []
        
        for idx, img_path in enumerate(all_images):
            # Determine class label based on parent directory name
            label = img_path.parent.name
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[{idx+1}/{len(all_images)}] Error loading: {img_path.name}")
                continue
                
            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            if not results.multi_hand_landmarks:
                print(f"[{idx+1}/{len(all_images)}] No hand detected in: {img_path.name}")
                continue
                
            # Note: We use multi_hand_world_landmarks for metric 3D coordinates!
            # world_landmarks coordinates are in meters around the hand's center.
            world_landmarks = results.multi_hand_world_landmarks[0]
            
            # Convert to numpy array (21, 3)
            coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark])
            
            # Apply our coordinate-invariant normalization math
            normalized_coords, R, scale = align_and_normalize_hand_3d(coords_3d)
            
            # Create a unique output name
            rel_path = img_path.relative_to(dataset_path)
            out_filename = rel_path.with_suffix('.json')
            out_file_path = coords_dir / out_filename
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save coordinates and rotation/scale metadata
            data_to_save = {
                "label": label,
                "original_image": str(rel_path),
                "raw_world_landmarks": coords_3d.tolist(),
                "normalized_landmarks": normalized_coords.tolist(),
                "rotation_matrix": R.tolist(),
                "scale_factor": float(scale)
            }
            
            with open(out_file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
                
            results_summary.append({
                "label": label,
                "file_path": str(out_filename),
                "scale": float(scale)
            })
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(all_images):
                print(f"Processed [{idx+1}/{len(all_images)}] images successfully.")
                
        # Write dataset metadata registry
        with open(output_path / "metadata_registry.json", 'w') as f:
            json.dump(results_summary, f, indent=4)
            
        print(f"\nProcessing Complete!")
        print(f"Successfully processed {len(results_summary)} / {len(all_images)} images.")
        print(f"Output saved to: {output_path.absolute()}")

if __name__ == "__main__":
    dataset_dir = r"E:\Projects\ASLtrainingData"
    output_dir = r"E:\Projects\ASLtrainingData_Processed"
    
    # Allow overriding directories from CLI
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory '{dataset_dir}' does not exist on this machine.")
        print("Please run the script passing the path directly: python extract_world_landmarks.py <dataset_path> <output_path>")
    else:
        extract_from_dataset(dataset_dir, output_dir)
