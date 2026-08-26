"""
process_validation.py
=====================
Processes the ASL validation images from the local dataset using MediaPipe's
Task API (HandLandmarker) in IMAGE mode. Extracts 3D world landmarks with
real metric depth (Z axis), applies canonical alignment normalization, and
outputs:

  1. validation_landmarks.csv  — Full spreadsheet of all 21 joint x/y/z per sign
  2. validation_presets.json    — JSON map of sign label -> 21x[x,y,z] arrays
                                  ready to be loaded by the 3D viewer

Both outputs are written to the wwwroot directory so the viewer can load them.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import csv
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def align_and_normalize_hand_3d(world_landmarks):
    """
    Translates, rotates, and scales 3D world landmarks to be camera,
    scale, and orientation invariant.

    Returns:
        aligned:         (21,3) numpy array in canonical space
        rotation_matrix: 3x3 orthonormal basis used
        scale:           wrist-to-MCP9 distance used for normalisation
    """
    # 1. Translation: Wrist (landmark 0) -> origin
    wrist = world_landmarks[0]
    translated = world_landmarks - wrist

    # 2. Scaling: distance from Wrist to Middle Finger MCP (landmark 9)
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    if scale < 1e-5:
        scale = 1.0
    scaled = translated / scale

    # 3. Rotation: build orthonormal basis
    #    Y-up  = wrist -> middle MCP
    #    Z-fwd = palm normal (cross of Y-up and wrist->index MCP)
    #    X-side = completes the frame
    v_up = scaled[9]
    u_up = v_up / np.linalg.norm(v_up)

    v_index = scaled[5]
    v_normal = np.cross(u_up, v_index)
    norm_len = np.linalg.norm(v_normal)
    if norm_len < 1e-5:
        # Fallback: use pinky MCP instead of index if cross product is degenerate
        v_index = scaled[17]
        v_normal = np.cross(u_up, v_index)
        norm_len = np.linalg.norm(v_normal)
    u_normal = v_normal / norm_len

    u_side = np.cross(u_normal, u_up)

    R = np.vstack([u_side, u_up, u_normal])  # (3,3)
    aligned = np.dot(scaled, R.T)

    return aligned, R, scale


def get_detector(task_path):
    """Create a HandLandmarker in IMAGE mode for still-image processing."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.15,   # Low threshold - we want every image
        min_hand_presence_confidence=0.15,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_validation_dataset():
    input_dir = Path(r"E:\Projects\ASLtrainingData\asl_alphabet_test\asl_alphabet_test")
    wwwroot_dir = Path(r"c:\Users\Windows 10 21H1\source\repos\SignLanugaApi\wwwroot")

    # Locate the hand_landmarker.task model file
    task_candidates = [
        Path(r"c:\Users\Windows 10 21H1\source\repos\SignLanugaApi\hand_landmarker.task"),
        Path(__file__).parent.parent / "hand_landmarker.task",
        Path("hand_landmarker.task"),
    ]
    task_path = None
    for c in task_candidates:
        if c.exists():
            task_path = c
            break

    if task_path is None:
        print("ERROR: Cannot find hand_landmarker.task model file.")
        print("Searched:", [str(c) for c in task_candidates])
        return

    csv_output_path = wwwroot_dir / "validation_landmarks.csv"
    json_output_path = wwwroot_dir / "validation_presets.json"

    print(f"Model file   : {task_path}")
    print(f"Input dir    : {input_dir}")
    print(f"CSV output   : {csv_output_path}")
    print(f"JSON output  : {json_output_path}")
    print()

    # Gather image files
    image_files = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )
    if not image_files:
        print(f"ERROR: No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} validation images.\n")

    # Create detector
    detector = get_detector(task_path)

    # CSV header: label, image, x0 y0 z0 ... x20 y20 z20
    csv_headers = ["label", "image_name"]
    for i in range(21):
        csv_headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    csv_rows = []
    json_presets = {}
    detected_count = 0

    for img_path in image_files:
        # Derive label from filename: "A_test.jpg" -> "A"
        label = img_path.stem.split("_")[0].upper()
        if label == "NOTHING":
            label = "Nothing"
        elif label == "SPACE":
            label = "Space"

        print(f"[{img_path.name}] label={label} ... ", end="", flush=True)

        image = cv2.imread(str(img_path))
        if image is None:
            print("SKIP (cannot read)")
            continue

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect in IMAGE mode
        results = detector.detect(mp_image)

        if not results.hand_landmarks:
            print("SKIP (no hand detected)")
            continue

        # Use hand_world_landmarks for real 3D metric coordinates (meters)
        # This gives us actual depth instead of the normalised z from hand_landmarks
        if results.hand_world_landmarks and len(results.hand_world_landmarks) > 0:
            world_lm = results.hand_world_landmarks[0]
            coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in world_lm])
            source = "world"
        else:
            # Fallback to normalised landmarks if world landmarks unavailable
            norm_lm = results.hand_landmarks[0]
            coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in norm_lm])
            source = "norm"

        # Apply canonical alignment
        normalized_coords, R, scale = align_and_normalize_hand_3d(coords_3d)

        # CSV row
        row = [label, img_path.name]
        for coord in normalized_coords:
            row.extend([round(float(coord[0]), 6),
                        round(float(coord[1]), 6),
                        round(float(coord[2]), 6)])
        csv_rows.append(row)

        # JSON preset (rounded for readability)
        json_presets[label] = [
            [round(float(c[0]), 6), round(float(c[1]), 6), round(float(c[2]), 6)]
            for c in normalized_coords
        ]

        detected_count += 1
        print(f"OK ({source}, scale={scale:.4f})")

    detector.close()

    # ---- Write outputs ----
    with open(csv_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)

    with open(json_output_path, "w") as f:
        json.dump(json_presets, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Done! {detected_count}/{len(image_files)} images processed.")
    print(f"CSV  -> {csv_output_path}")
    print(f"JSON -> {json_output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    process_validation_dataset()
