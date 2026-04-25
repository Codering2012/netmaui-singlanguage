# pip install requests mediapipe opencv-python

import cv2
import mediapipe as mp
import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API details
BASE_URL = "https://localhost:7084"
EMAIL = "demo@signlanguage.app"
PASSWORD = "DemoPass123!"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Login to get JWT token
def login():
    response = requests.post(f"{BASE_URL}/api/auth/login", json={"email": EMAIL, "password": PASSWORD}, verify=False)
    response.raise_for_status()
    return response.json()["token"]

# Extract landmarks from MediaPipe results
def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z
        })
    return landmarks

# Predict gesture using landmarks
def predict_gesture(token, landmarks):
    payload = {"landmarks": landmarks}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/api/gesture/predict-landmarks", json=payload, headers=headers, verify=False)
    return response.json()

# Main webcam loop
def main():
    token = login()
    print("Logged in, starting webcam...")

    cap = cv2.VideoCapture(0)
    prediction_buffer = []
    buffer_size = 5

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_prediction = "Waiting for sign..."
        confidence_str = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 landmarks
                landmarks = extract_landmarks(hand_landmarks)

                # Send to API for prediction
                try:
                    result = predict_gesture(token, landmarks)
                    if result["status"] == "success":
                        letter = result["data"]["letter"]
                        confidence = result["data"]["confidence"]

                        prediction_buffer.append(letter)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)

                        if prediction_buffer.count(letter) > (buffer_size // 2):
                            current_prediction = f"Sign: {letter}"
                            confidence_str = f"Confidence: {confidence:.1f}%"
                    else:
                        prediction_buffer.clear()
                except Exception as e:
                    print(f"API error: {e}")

        # Display on frame
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, current_prediction, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if confidence_str:
            cv2.putText(frame, confidence_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow('ASL Live Translator (Landmarks API)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()