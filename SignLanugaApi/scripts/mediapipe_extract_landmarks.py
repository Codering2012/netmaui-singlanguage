import json
import sys
import base64
import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "space"
]

def normalize_hand_array(coords):
    pts = np.asarray(coords, dtype=np.float32).reshape(1, 21, 3)
    wrists = pts[:, 0:1, :]
    pts = pts - wrists

    distances = np.linalg.norm(pts, axis=2)
    max_dists = np.max(distances, axis=1, keepdims=True)
    max_dists = np.clip(max_dists, 1e-5, None)
    pts = pts / np.expand_dims(max_dists, axis=2)

    return pts.reshape(1, 63).astype(np.float32)

def softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def resolve_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "asl_model.onnx"),
        os.path.join(script_dir, "asl_model.onnx"),
        os.path.join(script_dir, "..", "asl_model.onnx"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None

def try_load_model_session():
    model_path = resolve_model_path()
    if not model_path:
        return None, None, "asl_model.onnx not found"

    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        return session, input_name, None
    except Exception as exc:
        return None, None, str(exc)

def predict_landmarks(session, input_name, coords):
    norm_input = normalize_hand_array(coords)
    ort_outputs = session.run(None, {input_name: norm_input})
    logits = np.asarray(ort_outputs[0], dtype=np.float32)
    
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    probs = softmax(logits)
    best_idx = int(np.argmax(probs[0]))
    confidence = float(probs[0][best_idx])
    letter = CLASSES[best_idx] if best_idx < len(CLASSES) else CLASSES[-1]

    return {
        "letter": letter,
        "confidence": confidence,
    }

def extract_landmarks(image, hands, session, input_name):
    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return {"success": False, "error": "No hand landmarks detected"}

    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.extend([float(landmark.x), float(landmark.y), float(landmark.z)])

    response = {"success": True, "landmarks": coords}

    if session is not None and input_name is not None:
        try:
            response["prediction"] = predict_landmarks(session, input_name, coords)
        except Exception as exc:
            response["prediction_error"] = str(exc)

    return response

def run_worker() -> int:
    session, input_name, model_error = try_load_model_session()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                image_b64 = payload.get("imageBase64")
                
                if not image_b64:
                    print(json.dumps({"success": False, "error": "imageBase64 is required"}), flush=True)
                    continue

                image_bytes = base64.b64decode(image_b64)
                np_data = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                
                if image is None:
                    print(json.dumps({"success": False, "error": "Could not decode image bytes"}), flush=True)
                    continue

                result = extract_landmarks(image, hands, session, input_name)
                result["model_loaded"] = session is not None
                
                if model_error and session is None:
                    result["model_error"] = model_error
                    
                print(json.dumps(result), flush=True)
            except Exception as exc:
                print(json.dumps({"success": False, "error": str(exc)}), flush=True)

    return 0

def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        return run_worker()

    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Image path argument is required"}))
        return 2

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(json.dumps({"success": False, "error": "Could not read image"}))
        return 3

    session, input_name, model_error = try_load_model_session()

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        result = extract_landmarks(image, hands, session, input_name)

    result["model_loaded"] = session is not None
    if model_error and session is None:
        result["model_error"] = model_error
        
    print(json.dumps(result))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())