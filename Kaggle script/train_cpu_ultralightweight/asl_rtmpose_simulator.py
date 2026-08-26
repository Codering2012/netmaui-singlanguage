#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  ASL RTMPose-Hand Live Camera Feed & Interactive Recognition Interface (Python)
================================================================================
Giao diện Python tương tác thời gian thực sử dụng OpenMMLab RTMPose-Hand (rtmlib):
  1. HIGH-SPEED TRACKING: RTMDet-Nano + RTMPose-M ONNX Engine, kháng giật và mờ nhòe (motion blur)
     khi tay di chuyển với tốc độ cao.
  2. 21 HAND KEYPOINTS: Tương thích 100% với chuẩn 21 khớp MediaPipe/COCO-Hand.
  3. HYBRID CLASSIFIER: So khớp hình học cân bằng theo từng ngón tay + Mô hình nơ-ron PyTorch 26 lớp.
  4. PIPELINE 5 TẦNG: FSM 4 trạng thái, Prefix Trie 60k từ, SymSpell phục hồi câu,
     và dịch tiếng Anh sang tiếng Việt thời gian thực.
  5. HUD CYBERPUNK OVERLAY: Neon UI thời gian thực.
================================================================================
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import math
import argparse
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Đảm bảo đường dẫn module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_cpu import UltraLightweightASLModel
from asl_pipeline import ASLStreamPipeline
from asl_geometric import classify_asl_geometry, get_accurate_asl_template, DynamicGestureTracker
from asl_mediapipe_simulator import (
    ASLLandmarkFeatureExtractor,
    ASLMediaPipeHUD,
    get_canonical_asl_landmarks
)

try:
    from rtmlib import Hand, draw_skeleton
    RTMPOSE_AVAILABLE = True
except ImportError:
    RTMPOSE_AVAILABLE = False


class RTMPoseHandDetector:
    """
    Trình phát hiện 21 khớp bàn tay RTMPose (OpenMMLab / rtmlib).
    Kháng hiện tượng mất dấu khi tay vung nhanh nhờ RTMDet-Nano quét toàn khung hình mỗi frame.
    """
    def __init__(self, backend: str = 'onnxruntime', device: str = 'cpu'):
        if not RTMPOSE_AVAILABLE:
            raise ImportError("rtmlib chưa được cài đặt. Chạy: pip install rtmlib onnxruntime")
        print(f"[*] Khởi tạo RTMPose-Hand ({backend} trên {device})...", flush=True)
        self.detector = Hand(backend=backend, device=device)
        print("[+] RTMPose-Hand đã sẵn sàng!", flush=True)

    def extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất 21 khớp bàn tay chuẩn hóa [0, 1] từ frame BGR.
        Returns:
            np.ndarray shape [21, 3] với z ước lượng, hoặc None nếu không tìm thấy tay.
        """
        h, w = frame_bgr.shape[:2]
        keypoints, scores = self.detector(frame_bgr)
        
        if keypoints is None or len(keypoints) == 0:
            return None
            
        best_hand_idx = 0
        if len(keypoints) > 1:
            mean_scores = [np.mean(scores[i]) for i in range(len(scores))]
            best_hand_idx = int(np.argmax(mean_scores))
            
        kpts_2d = keypoints[best_hand_idx] # [21, 2] in pixel coords
        scs = scores[best_hand_idx]
        
        if np.mean(scs) < 0.20:
            return None
            
        # Chuẩn hóa tọa độ về [0, 1]
        norm_kpts = np.zeros((21, 3), dtype=np.float32)
        norm_kpts[:, 0] = kpts_2d[:, 0] / float(w)
        norm_kpts[:, 1] = kpts_2d[:, 1] / float(h)
        
        # Ước lượng chiều sâu z tương đối từ độ dài xương ngón giữa
        wrist = norm_kpts[0, :2]
        middle_mcp = norm_kpts[9, :2]
        palm_scale = np.linalg.norm(middle_mcp - wrist) + 1e-6
        norm_kpts[:, 2] = (kpts_2d[:, 1] - wrist[1]*h) / (palm_scale * h * 4.0)
        
        return norm_kpts


class ASLRTMPoseInterface:
    """
    Giao diện điều khiển hoàn chỉnh cho ASL nhận diện qua RTMPose-Hand.
    """
    def __init__(
        self,
        camera_idx: int = 0,
        model_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        device_type: str = "cpu"
    ):
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        
        self.detector = RTMPoseHandDetector(backend='onnxruntime', device=device_type)
        self.classifier = ASLLandmarkFeatureExtractor(model_path=model_path)
        self.pipeline = ASLStreamPipeline(hold_duration=1.2, release_duration=0.5, majority_window=5)
        self.hud = ASLMediaPipeHUD(width=width, height=height)
        
        self.detected_char = ""
        self.detected_conf = 0.0
        self.fsm_state_str = "IDLE"
        self.hold_progress = 0.0
        self.active_word = ""
        self.suggestions = []
        self.english_sentence = ""
        self.vietnamese_sentence = ""
        self.is_sim_mode = False
        self.sim_target_letter = 'A'

    def run(self):
        """Vòng lặp chạy giao diện trực tiếp từ Camera."""
        cap = cv2.VideoCapture(self.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print(f"[!] Không thể mở Camera #{self.camera_idx}. Chuyển sang chế độ Simulator!", flush=True)
            self.is_sim_mode = True
        else:
            print(f"[+] Camera #{self.camera_idx} đã kết nối thành công!", flush=True)

        cv2.namedWindow("ASL RTMPose Live Recognition Interface (High-Speed Motion)", cv2.WINDOW_AUTOSIZE)

        frame_count = 0
        t_start = time.time()
        fps = 30.0

        print("\n" + "="*70)
        print("  GIAO DIỆN CAMERA RTMPOSE-HAND ASL ĐANG HOẠT ĐỘNG (KHÁNG MỜ NHÒE VÀ VUNG TAY NHANH)!")
        print("  - Đưa bàn tay trước Camera để nhận diện ký tự ngón tay ASL (A-Z).")
        print("  - Phím [Space]: Thêm dấu cách | [Backspace]: Xóa ký tự | [Enter]: Dịch câu.")
        print("  - Phím [1, 2, 3]: Chọn từ gợi ý Autocomplete.")
        print("  - Phím [C]: Xóa toàn bộ câu | [Q]: Thoát chương trình.")
        print("="*70 + "\n", flush=True)

        try:
            while True:
                t0 = time.perf_counter()

                if not self.is_sim_mode and cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        landmarks_21 = None
                    else:
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (self.width, self.height))
                        landmarks_21 = self.detector.extract_landmarks(frame)
                else:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    landmarks_21 = get_canonical_asl_landmarks(self.sim_target_letter)

                lms_dict = {"landmarks": landmarks_21.tolist()} if landmarks_21 is not None else None

                # 2. Xử lý nhận diện
                if landmarks_21 is not None:
                    # Vẽ khung xương 21 khớp
                    self.hud.draw_skeleton(frame, landmarks_21)
                    
                    pred_char, pred_conf = self.classifier.extract_features(landmarks_21)
                    event = self.pipeline.process_frame(
                        predicted_letter=pred_char,
                        confidence=pred_conf,
                        landmarks_dict=lms_dict
                    )
                    self.detected_char = pred_char
                    self.detected_conf = pred_conf
                    self.fsm_state_str = self.pipeline.fsm.state
                    self.hold_progress = min(1.0, self.pipeline.fsm.hold_count / max(1, self.pipeline.fsm.hold_repeat_frames))
                    self.active_word = self.pipeline.buffer.current_word
                    self.suggestions = event.get("suggestions", [])

                    if event.get("is_sentence_final") and event.get("english"):
                        self.english_sentence = event["english"]
                        self.vietnamese_sentence = event["vietnamese"]
                    elif not self.english_sentence:
                        self.english_sentence = self.pipeline.buffer.get_raw_preview()
                        self.vietnamese_sentence = "..."
                else:
                    event = self.pipeline.process_frame(
                        predicted_letter=None,
                        confidence=0.0,
                        landmarks_dict=None
                    )
                    self.detected_char = ""
                    self.detected_conf = 0.0
                    self.fsm_state_str = self.pipeline.fsm.state
                    self.hold_progress = 0.0

                # 3. Đo lường tốc độ thực tế
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                frame_count += 1
                if frame_count % 10 == 0:
                    t_now = time.time()
                    fps = 10.0 / max(0.001, t_now - t_start)
                    t_start = t_now

                # 4. Vẽ bảng điều khiển HUD Cyberpunk
                self.hud.draw_hud_panel(
                    img=frame,
                    current_char=self.detected_char,
                    confidence=self.detected_conf,
                    fsm_state=self.fsm_state_str,
                    hold_progress=self.hold_progress,
                    active_word=self.active_word,
                    trie_suggestions=self.suggestions,
                    english_sentence=self.english_sentence,
                    vietnamese_sentence=self.vietnamese_sentence,
                    fps=fps,
                    latency_ms=latency_ms,
                    is_sim_mode=self.is_sim_mode
                )

                cv2.imshow("ASL RTMPose Live Recognition Interface (High-Speed Motion)", frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('m'):
                    self.is_sim_mode = not self.is_sim_mode
                elif key == ord('c'):
                    self.pipeline.reset()
                    self.english_sentence = ""
                    self.vietnamese_sentence = ""
                    self.active_word = ""
                elif key == 32: # Space
                    self.pipeline.buffer.add_space()
                elif key == 8: # Backspace
                    self.pipeline.buffer.backspace()
                elif key == 13: # Enter
                    raw_preview = self.pipeline.buffer.get_raw_preview()
                    if raw_preview.strip():
                        res = self.pipeline.corrector.correct_sentence(raw_preview)
                        self.english_sentence = res["corrected_text"]
                        self.vietnamese_sentence = self.pipeline.translator.translate_to_vietnamese(self.english_sentence)
                elif key in [ord('1'), ord('2'), ord('3')]:
                    idx = key - ord('1')
                    if idx < len(self.suggestions):
                        self.pipeline.buffer.accept_suggestion(self.suggestions[idx])
                elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                    self.sim_target_letter = chr(key).upper()

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print("[+] Đã đóng ứng dụng RTMPose ASL thành công!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL RTMPose Live Camera App")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--model", type=str, default=None, help="Path to asl_cpu_model.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Device for ONNXRuntime (cpu or cuda)")
    args = parser.parse_args()

    app = ASLRTMPoseInterface(
        camera_idx=args.camera,
        model_path=args.model,
        width=args.width,
        height=args.height,
        device_type=args.device
    )
    app.run()
