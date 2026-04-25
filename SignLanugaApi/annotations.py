import cv2
import torch
import numpy as np
import onnxruntime as ort # Requires: pip install onnxruntime

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- 1. THE NORMALIZATION MATH ---
# (Kept PyTorch for math operations to minimize changes)
def normalize_hand_tensor(batch_X):
    # Handle single inputs from the webcam safely
    if len(batch_X.shape) == 1:
        batch_X = batch_X.unsqueeze(0)
        
    batch_size = batch_X.shape[0]
    pts = batch_X.view(batch_size, 21, 3)
    
    # Translate to Center
    wrists = pts[:, 0:1, :] 
    pts = pts - wrists
    
    # Scale to standard size
    distances = torch.norm(pts, dim=2) 
    max_dists, _ = torch.max(distances, dim=1, keepdim=True)
    max_dists = torch.clamp(max_dists, min=1e-5) 
    pts = pts / max_dists.unsqueeze(2)
    
    return pts.view(batch_size, 63)

# The ASLNeuralNetwork class has been removed. ONNX handles the architecture natively!

def main():
    # --- 3. LOAD THE TRAINED BRAIN (ONNX) ---
    model_path = 'asl_model.onnx' 
    print("Loading ONNX AI model...")
    try:
        # Initialize the ONNX Inference Session
        session = ort.InferenceSession(model_path)
        
        # Get the expected input name for the ONNX graph
        input_name = session.get_inputs()[0].name
        
        # Define the classes manually since ONNX does not carry Python dictionaries.
        # UPDATE THIS LIST if your model uses a different set of classes.
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
        num_classes = len(classes)
        
        print(f"ONNX Model loaded successfully! Recognizes {num_classes} signs.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 4. INITIALIZE MEDIAPIPE ---
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    print("Webcam activated. Press 'q' to quit.")

    prediction_buffer = []
    buffer_size = 5 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip visually so it acts like a mirror
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_prediction = "Waiting for sign..."
        confidence_str = ""
        hand_type_str = ""

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                mp_handedness = results.multi_handedness[idx].classification[0].label
                hand_type_str = "(Left Hand)" if mp_handedness == "Right" else "(Right Hand)"

                # --- NATIVE EXTRACTION (AI handles the reflection natively!) ---
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])

                input_tensor = torch.tensor(coords, dtype=torch.float32)

                # --- 5. MAKE THE PREDICTION (ONNX) ---
                # Center and scale the webcam data
                norm_tensor = normalize_hand_tensor(input_tensor)
                
                # Convert the normalized PyTorch tensor to a NumPy array for ONNX
                input_data = norm_tensor.numpy()
                
                # Run the ONNX session
                ort_outs = session.run(None, {input_name: input_data})
                
                # Convert output back to PyTorch tensor to reuse your existing confidence math
                output = torch.tensor(ort_outs[0])
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence_val = confidence.item()
                predicted_letter = classes[predicted_idx.item()]

                # --- 6. SMOOTHING AND UI ---
                if confidence_val > 0.80: 
                    prediction_buffer.append(predicted_letter)
                    if len(prediction_buffer) > buffer_size:
                        prediction_buffer.pop(0)
                    
                    if prediction_buffer.count(predicted_letter) > (buffer_size // 2):
                        current_prediction = f"Sign: {predicted_letter} {hand_type_str}"
                        confidence_str = f"Confidence: {confidence_val*100:.1f}%"
                else:
                    prediction_buffer.clear()

        cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, current_prediction, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if confidence_str:
            cv2.putText(frame, confidence_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow('ASL Live Translator (ONNX Build)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()