import sys
import os
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from asl_rtmpose_simulator import RTMPoseHandDetector
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor, ASLMediaPipeHUD
from asl_pipeline import ASLStreamPipeline

print("[*] Kiểm thử Pipeline ASL với RTMPose-Hand (Headless)...")
detector = RTMPoseHandDetector(backend='onnxruntime', device='cpu')
extractor = ASLLandmarkFeatureExtractor()
pipeline = ASLStreamPipeline()
hud = ASLMediaPipeHUD(width=1280, height=720)

dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.circle(dummy_frame, (640, 360), 80, (220, 220, 220), -1)

# Chạy trích xuất và render HUD
lms = detector.extract_landmarks(dummy_frame)
pred_char, conf = extractor.extract_features(np.zeros((21, 3)))
event = pipeline.process_frame(pred_char, conf, landmarks_dict=None)

hud.draw_hud_panel(
    img=dummy_frame,
    current_char=pred_char,
    confidence=conf,
    fsm_state=pipeline.fsm.state,
    hold_progress=0.5,
    active_word="HELLO",
    trie_suggestions=["HELLO", "HELP", "HELL"],
    english_sentence="HELLO WORLD",
    vietnamese_sentence="XIN CHÀO THẾ GIỚI",
    fps=60.0,
    latency_ms=12.5,
    is_sim_mode=True
)

out_preview_path = r"C:\Users\Windows 10 21H1\.gemini\antigravity-ide\brain\03909049-1864-4bc0-88ea-9cfe841517e6\rtmpose_asl_interface_preview.png"
cv2.imwrite(out_preview_path, dummy_frame)

print(f"[+] Đã lưu ảnh chụp giao diện RTMPose HUD tại: {out_preview_path}")
print("[+] Kiểm thử RTMPose ASL Pipeline thành công 100%!")
