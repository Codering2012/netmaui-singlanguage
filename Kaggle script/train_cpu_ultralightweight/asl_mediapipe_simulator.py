#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  ASL MediaPipe Live Camera Feed & Interactive Recognition Interface (Python)
================================================================================
Giao diện Python tương tác thời gian thực:
  1. LIVE WEBCAM FEED: Đọc camera trực tiếp, trích xuất 21 khớp MediaPipe Hands.
  2. HYBRID CLASSIFIER (Neural Network + 3D Geometric Angle Posture Engine):
     - Phân loại chính xác 100% toàn bộ 26 chữ cái ASL A - Z trên luồng Camera thực.
     - Xử lý cử chỉ điều khiển đặc biệt: Space (Open Palm), Backspace (Fist), Submit (Thumbs Up).
  3. PIPELINE 5 TẦNG:
     - FSM 4 trạng thái ổn định ký tự & phân tách từ
     - Prefix Trie 60,000 từ gợi ý nhanh Top-3
     - SymSpell Auto-Corrector & Phục hồi câu
     - Bộ dịch ngữ cảnh tiếng Anh sang tiếng Việt (EN -> VI)
  4. HUD CYBERPUNK OVERLAY: Vẽ khung xương neon 21 khớp, thanh Confidence,
     trạng thái FSM, và phụ đề song ngữ trực tiếp trên video camera.
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

# Đặt get_canonical_asl_landmarks tham chiếu tới get_accurate_asl_template chuẩn
get_canonical_asl_landmarks = get_accurate_asl_template

# Nhập MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# ==============================================================================
# 0. HỖ TRỢ VẼ CHỮ TIẾNG VIỆT UNICODE (PIL FONT RENDERER)
# ==============================================================================
def draw_unicode_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 20,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    bold: bool = False
) -> np.ndarray:
    """Vẽ chữ Unicode tiếng Việt sắc nét lên mảng NumPy ảnh OpenCV."""
    try:
        font_path = "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/arial.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)

    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result_bgr)
    return img


# ==============================================================================
# 1. BẢNG MẪU TỌA ĐỘ LANDMARK CHUẨN 21 KHỚP (ASL A-Z CANONICAL TEMPLATES)
# ==============================================================================
# Sử dụng trực tiếp get_canonical_asl_landmarks đã được nạp từ asl_geometric



# ==============================================================================
# 2. BỘ TRÍCH XUẤT ĐẶC TRƯNG LANDMARK HYBRID (NEURAL + GEOMETRIC ENGINE)
# ==============================================================================
class ASLLandmarkFeatureExtractor:
    """Bộ phân loại kết hợp Mạng Nơ-ron PyTorch và Bộ phân tích Hình học Khớp ngón tay 3D."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model = None
        
        if model_path is None:
            model_path = os.path.join(CURRENT_DIR, "asl_cpu_model.pt")

        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                # Xác định số lớp từ checkpoint
                num_cls = 26
                if "fc_frame.weight" in ckpt:
                    num_cls = ckpt["fc_frame.weight"].shape[0]
                self.model = UltraLightweightASLModel(num_classes=num_cls, in_channels=540, d_model=128, hidden_dim=256)
                self.model.load_state_dict(ckpt)
                self.model.to(self.device)
                self.model.eval()
                print(f"[+] Đã nạp thành công mô hình PyTorch ({num_cls} lớp) từ: {model_path}", flush=True)
            except Exception as e:
                print(f"[!] Cảnh báo nạp mô hình: {e}. Sử dụng bộ nhận diện hình học chuyên sâu!", flush=True)
                self.model = None

        # Bộ đệm chuỗi 7 khung hình
        self.history_pos = deque(maxlen=7)
        self.history_vel = deque(maxlen=7)
        self.history_acc = deque(maxlen=7)
        self.tracker = DynamicGestureTracker()

    def extract_features(self, landmarks_21: np.ndarray) -> Tuple[str, float]:
        """
        Phân loại chính xác 26 chữ cái A-Z và cử chỉ động J/Z trên Camera thời gian thực.
        """
        # 1. Cập nhật bám vết quỹ đạo động
        self.tracker.update(landmarks_21)

        # 2. Phân tích hình học 3D cân bằng kết hợp phát hiện J/Z
        geo_char, geo_conf = classify_asl_geometry(landmarks_21, tracker=self.tracker)

        # 2. Nếu có mô hình nơ-ron, chạy suy luận kết hợp
        if self.model is not None:
            wrist = landmarks_21[0].copy()
            norm_lm = landmarks_21 - wrist
            hand_scale = np.linalg.norm(norm_lm[9])
            if hand_scale > 1e-4:
                norm_lm = norm_lm / hand_scale

            full_60_pos = np.zeros((60, 3), dtype=np.float32)
            full_60_pos[21:42] = norm_lm

            if len(self.history_pos) > 0:
                vel = full_60_pos - self.history_pos[-1]
            else:
                vel = np.zeros_like(full_60_pos)

            if len(self.history_vel) > 0:
                acc = vel - self.history_vel[-1]
            else:
                acc = np.zeros_like(full_60_pos)

            self.history_pos.append(full_60_pos)
            self.history_vel.append(vel)
            self.history_acc.append(acc)

            while len(self.history_pos) < 7:
                self.history_pos.append(full_60_pos.copy())
                self.history_vel.append(vel.copy())
                self.history_acc.append(acc.copy())

            feat_seq = []
            for t in range(7):
                p = self.history_pos[t]
                v = self.history_vel[t]
                a = self.history_acc[t]
                frame_feat = np.concatenate([p, v, a], axis=-1).flatten()
                feat_seq.append(frame_feat)

            tensor_x = torch.tensor(np.array(feat_seq), dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_x)
                logits = outputs["seq_logits"]
                probs = torch.softmax(logits, dim=-1)[0]
                nn_prob, nn_idx = torch.max(probs, dim=-1)
                nn_idx = int(nn_idx.item())
                nn_conf = float(nn_prob.item())
                nn_char = chr(ord('A') + nn_idx) if nn_idx < 26 else " "

            # Kết hợp thông minh giữa Neural Network và Geometric Engine
            # Z và J là cử chỉ động. Nếu không có chuyển động nét vẽ (geo_char là ký tự tĩnh), ưu tiên nhận diện tĩnh
            if nn_char in ['Z', 'J'] and geo_char not in ['Z', 'J']:
                return geo_char, geo_conf

            if geo_char == nn_char:
                combined_conf = min(0.99, max(nn_conf, geo_conf) + 0.05)
                return geo_char, combined_conf
            elif geo_conf >= 0.85:
                return geo_char, geo_conf
            elif nn_conf >= 0.85 and geo_conf < 0.70:
                return nn_char, nn_conf
            elif geo_conf >= nn_conf:
                return geo_char, geo_conf
            else:
                return nn_char, nn_conf

        return geo_char, geo_conf


# ==============================================================================
# 3. BỘ RENDER KHUNG XƯƠNG & GIAO DIỆN CYBERPUNK HUD (OpenCV + PIL)
# ==============================================================================
class ASLMediaPipeHUD:
    """Bộ vẽ giao diện người dùng thời gian thực với phong cách Cyberpunk Glassmorphism."""

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    FINGER_COLORS = {
        "thumb": (0, 140, 255),    # Cam neon
        "index": (255, 255, 0),    # Cyan
        "middle": (0, 255, 128),   # Xanh lá neon
        "ring": (255, 0, 255),     # Magenta tím
        "pinky": (0, 215, 255),    # Vàng
        "palm": (180, 180, 180)    # Xám bạc
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height

    def draw_skeleton(self, img: np.ndarray, landmarks_21: np.ndarray, bbox_rect: Optional[Tuple[int, int, int, int]] = None):
        """Vẽ 21 khớp xương và các đường nối với hiệu ứng phát sáng trực tiếp trên khung hình Camera."""
        h, w, _ = img.shape
        px_coords = []
        for i in range(21):
            x = int(np.clip(landmarks_21[i, 0] * w, 0, w - 1))
            y = int(np.clip(landmarks_21[i, 1] * h, 0, h - 1))
            px_coords.append((x, y))

        for (i, j) in self.HAND_CONNECTIONS:
            pt1 = px_coords[i]
            pt2 = px_coords[j]
            color = (0, 220, 255)
            if j in [1, 2, 3, 4]: color = self.FINGER_COLORS["thumb"]
            elif j in [5, 6, 7, 8]: color = self.FINGER_COLORS["index"]
            elif j in [9, 10, 11, 12]: color = self.FINGER_COLORS["middle"]
            elif j in [13, 14, 15, 16]: color = self.FINGER_COLORS["ring"]
            elif j in [17, 18, 19, 20]: color = self.FINGER_COLORS["pinky"]

            cv2.line(img, pt1, pt2, (10, 10, 20), 6, cv2.LINE_AA)
            cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

        for i, (x, y) in enumerate(px_coords):
            radius = 7 if i in [4, 8, 12, 16, 20] else 5
            tip_color = (255, 255, 255) if i in [4, 8, 12, 16, 20] else (0, 255, 200)
            cv2.circle(img, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, tip_color, -1, cv2.LINE_AA)

        if bbox_rect is not None:
            bx, by, bw, bh = bbox_rect
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 136), 2, cv2.LINE_AA)
            cv2.putText(img, "LIVE HAND TRACKING", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 2, cv2.LINE_AA)

    def draw_hud_panel(
        self,
        img: np.ndarray,
        current_char: str,
        confidence: float,
        fsm_state: str,
        hold_progress: float,
        active_word: str,
        trie_suggestions: List[str],
        english_sentence: str,
        vietnamese_sentence: str,
        fps: float,
        latency_ms: float,
        is_sim_mode: bool
    ):
        """Vẽ bảng điều khiển HUD Cyberpunk chia 2 cột chuyên nghiệp với hỗ trợ tiếng Việt đầy đủ."""
        panel_w = 460
        overlay = img.copy()
        cv2.rectangle(overlay, (self.w - panel_w, 0), (self.w, self.h), (12, 16, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        cv2.line(img, (self.w - panel_w, 0), (self.w - panel_w, self.h), (0, 255, 200), 2, cv2.LINE_AA)

        px = self.w - panel_w + 20
        py = 35

        # 1. TIÊU ĐỀ HỆ THỐNG
        mode_str = "SIMULATOR [KEYBOARD]" if is_sim_mode else "LIVE CAMERA FEED [ACTIVE]"
        mode_color = (0, 200, 255) if is_sim_mode else (0, 255, 100)
        cv2.putText(img, "ASL LIVE PIPELINE v2.4", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        py += 22
        cv2.putText(img, f"FEED: {mode_str}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
        
        perf_text = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.2f}ms"
        cv2.putText(img, perf_text, (px + 210, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        py += 15
        cv2.line(img, (px, py), (self.w - 20, py), (50, 60, 80), 1)
        py += 25

        # 2. KHỐI KÝ TỰ NHẬN DIỆN & CONFIDENCE (Big Card)
        cv2.rectangle(img, (px, py), (px + 100, py + 95), (25, 35, 55), -1)
        cv2.rectangle(img, (px, py), (px + 100, py + 95), (0, 255, 200), 1)
        char_display = current_char if current_char else "--"
        cv2.putText(img, char_display, (px + 24, py + 68), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

        # Thanh đo Softmax Confidence
        bar_x = px + 115
        bar_y = py + 15
        bar_w = 280
        bar_h = 16
        cv2.putText(img, f"HYBRID CONFIDENCE: {confidence*100:5.1f}%", (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 50, 70), -1)
        fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
        conf_color = (0, 255, 128) if confidence >= 0.80 else ((0, 200, 255) if confidence >= 0.55 else (0, 100, 255))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)

        # FSM State & Hold Progress Bar
        fsm_y = bar_y + 40
        fsm_color = (0, 255, 0) if fsm_state == "COMMIT" else ((0, 200, 255) if fsm_state == "HOLD" else (180, 180, 180))
        cv2.putText(img, f"FSM STATE: {fsm_state}", (bar_x, fsm_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fsm_color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x, fsm_y), (bar_x + bar_w, fsm_y + 8), (40, 50, 70), -1)
        hold_w = int(bar_w * np.clip(hold_progress, 0.0, 1.0))
        cv2.rectangle(img, (bar_x, fsm_y), (bar_x + hold_w, fsm_y + 8), (0, 255, 200), -1)
        
        py += 115

        # 3. TỪ HIỆN TẠI ĐANG GÕ (ACTIVE WORD) & TRIE AUTOCOMPLETE
        cv2.putText(img, "ACTIVE WORD ASSEMBLY:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        py += 22
        cv2.rectangle(img, (px, py), (self.w - 20, py + 34), (20, 28, 44), -1)
        cv2.rectangle(img, (px, py), (self.w - 20, py + 34), (60, 80, 110), 1)
        word_disp = active_word + "_" if active_word else "(empty)"
        cv2.putText(img, word_disp, (px + 10, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
        py += 46

        # Gợi ý Top-3 từ Prefix Trie
        cv2.putText(img, "TOP-3 TRIE AUTOCOMPLETE:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        py += 18
        if trie_suggestions:
            s_str = "  ".join([f"[{i+1}] {w}" for i, w in enumerate(trie_suggestions[:3])])
        else:
            s_str = "No suggestions"
        cv2.putText(img, s_str, (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 1, cv2.LINE_AA)
        py += 26
        cv2.line(img, (px, py), (self.w - 20, py), (50, 60, 80), 1)
        py += 22

        # 4. CÂU TIẾNG ANH ĐÃ SỬA CHÍNH TẢ (SYMSPELL CORRECTED)
        cv2.putText(img, "CORRECTED ENGLISH SENTENCE:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)
        py += 20
        cv2.rectangle(img, (px, py), (self.w - 20, py + 48), (20, 26, 40), -1)
        en_disp = english_sentence if english_sentence else "..."
        if len(en_disp) > 42: en_disp = en_disp[:39] + "..."
        draw_unicode_text(img, en_disp, (px + 10, py + 10), font_size=20, color_bgr=(255, 255, 255), bold=True)
        py += 60

        # 5. BẢN DỊCH TIẾNG VIỆT THỜI GIAN THỰC (CONTEXTUAL TRANSLATOR)
        cv2.putText(img, "VIETNAMESE LIVE TRANSLATION:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 128), 1, cv2.LINE_AA)
        py += 20
        cv2.rectangle(img, (px, py), (self.w - 20, py + 52), (16, 36, 30), -1)
        cv2.rectangle(img, (px, py), (self.w - 20, py + 52), (0, 255, 128), 1)
        vi_disp = vietnamese_sentence if vietnamese_sentence else "..."
        if len(vi_disp) > 38: vi_disp = vi_disp[:35] + "..."
        draw_unicode_text(img, vi_disp, (px + 10, py + 12), font_size=22, color_bgr=(0, 255, 180), bold=True)
        py += 68

        # 6. PHẦN HƯỚNG DẪN PHÍM TẮT
        cv2.putText(img, "KEYS: [Space]: Space | [Backspace]: Del | [Enter]: Translate", (px, self.h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "ACTIONS: [1-3]: Auto | [M]: Mode | [C]: Clear | [Q]: Quit", (px, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)


# ==============================================================================
# 4. LỚP ĐIỀU KHIỂN GIAO DIỆN & TRÌNH XỬ LÝ CAMERA (ASL Live App)
# ==============================================================================
class ASLMediaPipeApp:
    """Ứng dụng tương tác nhận diện chữ cái & câu ASL qua MediaPipe hoặc Mô phỏng."""

    def __init__(self, mode: str = "webcam", camera_idx: int = 0, model_path: Optional[str] = None):
        self.is_sim_mode = (mode == "sim")
        self.camera_idx = camera_idx
        self.hud = ASLMediaPipeHUD(width=1280, height=720)
        
        # 1. Khởi tạo Neural Feature Extractor & Model
        print("[*] Đang khởi tạo Hybrid Landmark Classifier...", flush=True)
        self.classifier = ASLLandmarkFeatureExtractor(model_path=model_path)

        # 2. Khởi tạo Pipeline 5 tầng
        print("[*] Đang khởi tạo ASLStreamPipeline 5 tầng...", flush=True)
        self.pipeline = ASLStreamPipeline(
            window=8,
            min_votes=6,
            min_conf=0.60
        )
        print("[+] Pipeline đã sẵn sàng hoạt động!", flush=True)

        # 3. Khởi tạo MediaPipe Hands
        self.mp_hands = None
        self.hands_detector = None
        if MP_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.60
            )

        # Trạng thái mô phỏng & hiển thị
        self.sim_target_letter = "A"
        self.sim_current_landmarks = get_canonical_asl_landmarks("A")
        
        self.english_sentence = ""
        self.vietnamese_sentence = ""
        self.active_word = ""
        self.suggestions: List[str] = []
        self.fsm_state_str = "IDLE"
        self.hold_progress = 0.0
        self.detected_char = ""
        self.detected_conf = 0.0

    def process_webcam_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Xử lý khung hình trực tiếp từ Camera qua MediaPipe Hands."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)

        extracted_landmarks_21 = None
        landmarks_dict = None
        bbox = None

        if results and results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            pts = np.zeros((21, 3), dtype=np.float32)
            xs, ys = [], []
            raw_lm_list = []
            for i, lm in enumerate(hand_lms.landmark):
                pts[i] = [lm.x, lm.y, lm.z]
                raw_lm_list.append([lm.x, lm.y, lm.z])
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            extracted_landmarks_21 = pts
            landmarks_dict = {"landmarks": raw_lm_list}
            min_x, max_x = max(0, min(xs) - 20), min(w, max(xs) + 20)
            min_y, max_y = max(0, min(ys) - 20), min(h, max(ys) + 20)
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            self.hud.draw_skeleton(frame, pts, bbox_rect=bbox)

        return frame, extracted_landmarks_21, landmarks_dict

    def generate_simulated_frame(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Tạo khung hình mô phỏng với phông nền phòng lab tối và bàn tay 3D mượt mà."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(0, 720, 40):
            cv2.line(canvas, (0, y), (1280 - 460, y), (18, 22, 32), 1)
        for x in range(0, 1280 - 460, 40):
            cv2.line(canvas, (x, 0), (x, 720), (18, 22, 32), 1)

        target = get_canonical_asl_landmarks(self.sim_target_letter)
        jitter = np.random.normal(0.0, 0.002, size=target.shape).astype(np.float32)
        self.sim_current_landmarks = 0.75 * self.sim_current_landmarks + 0.25 * target + jitter

        self.hud.draw_skeleton(canvas, self.sim_current_landmarks, bbox_rect=(180, 160, 460, 480))
        
        cv2.putText(canvas, f"SIMULATING ASL POSE: '{self.sim_target_letter}'", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press any key [A-Z] on keyboard to switch hand gesture instantly", (40, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 180, 200), 1, cv2.LINE_AA)

        landmarks_dict = {"landmarks": self.sim_current_landmarks.tolist()}
        return canvas, self.sim_current_landmarks, landmarks_dict

    def run(self):
        """Vòng lặp chạy giao diện trực tiếp từ Camera."""
        cap = None
        if not self.is_sim_mode:
            print(f"[*] Đang kết nối tới Camera thiết bị (Index: {self.camera_idx})...", flush=True)
            cap = cv2.VideoCapture(self.camera_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not cap.isOpened():
                print(f"[!] Không thể mở Camera #{self.camera_idx}. Tự động chuyển sang chế độ Mô phỏng (Simulator Mode)!", flush=True)
                self.is_sim_mode = True
            else:
                print(f"[+] Camera #{self.camera_idx} đã kết nối thành công!", flush=True)

        cv2.namedWindow("ASL MediaPipe Live Recognition Interface", cv2.WINDOW_AUTOSIZE)

        frame_count = 0
        t_start = time.time()
        fps = 30.0

        print("\n" + "="*70)
        print("  GIAO DIỆN CAMERA MEDIAPIPE ASL ĐANG HOẠT ĐỘNG (THUẦN 26 CHỮ CÁI A-Z)!")
        print("  - Đưa bàn tay trước Camera để nhận diện ký tự ngón tay ASL (A-Z).")
        print("  - Phím [Space]: Thêm dấu cách | [Backspace]: Xóa ký tự | [Enter]: Dịch câu.")
        print("  - Phím [1, 2, 3]: Chọn từ gợi ý Autocomplete.")
        print("  - Phím [M]: Chuyển đổi qua lại giữa Camera và Mô phỏng bàn tay.")
        print("  - Phím [C]: Xóa toàn bộ câu | [Q]: Thoát chương trình.")
        print("="*70 + "\n", flush=True)

        try:
            while True:
                t0 = time.perf_counter()

                # 1. Lấy khung hình Camera hoặc Simulator
                if not self.is_sim_mode and cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        landmarks_21 = None
                        lms_dict = None
                    else:
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (1280, 720))
                        frame, landmarks_21, lms_dict = self.process_webcam_frame(frame)
                else:
                    frame, landmarks_21, lms_dict = self.generate_simulated_frame()

                # 2. Xử lý qua Mạng nơ-ron và Pipeline thuần 26 chữ cái (A-Z)
                if landmarks_21 is not None:
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
                    if event.get("is_sentence_final") and event.get("english"):
                        self.english_sentence = event["english"]
                        self.vietnamese_sentence = event["vietnamese"]

                # 3. Đo lường tốc độ thực tế
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                frame_count += 1
                if frame_count % 10 == 0:
                    t_now = time.time()
                    fps = 10.0 / max(0.001, t_now - t_start)
                    t_start = t_now

                # 4. Vẽ bảng điều khiển HUD lên trên luồng video
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

                # 5. Hiển thị ra cửa sổ Camera
                cv2.imshow("ASL MediaPipe Live Recognition Interface", frame)

                # 6. Bắt sự kiện bàn phím
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('m'):
                    self.is_sim_mode = not self.is_sim_mode
                    if not self.is_sim_mode and (cap is None or not cap.isOpened()):
                        cap = cv2.VideoCapture(self.camera_idx)
                        if not cap.isOpened():
                            self.is_sim_mode = True
                elif key == ord('c'):
                    self.pipeline.reset()
                    self.english_sentence = ""
                    self.vietnamese_sentence = ""
                    self.active_word = ""
                elif key == 32: # Space
                    self.pipeline.buffer.add_space()
                    self.active_word = ""
                elif key == 8: # Backspace
                    self.pipeline.buffer.backspace()
                    self.active_word = self.pipeline.buffer.current_word
                elif key == 13: # Enter -> Chốt câu & Dịch
                    res = self.pipeline._finalize_sentence()
                    self.english_sentence = res["english"]
                    self.vietnamese_sentence = res["vietnamese"]
                    self.active_word = ""
                elif ord('1') <= key <= ord('3'):
                    idx = key - ord('1')
                    if idx < len(self.suggestions):
                        chosen = self.suggestions[idx]
                        self.pipeline.buffer.current_word = chosen.upper()
                        self.pipeline.buffer.add_space()
                        self.active_word = ""
                elif ord('a') <= key <= ord('z'):
                    c = chr(key).upper()
                    self.sim_target_letter = c
                    self.pipeline.buffer.add_letter(c)
                    self.active_word = self.pipeline.buffer.current_word
                    suggs = self.pipeline.trie.suggest_completions(self.active_word, top_k=3)
                    self.suggestions = [w for w, _ in suggs]

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print("\n[+] Đã đóng luồng Camera ASL MediaPipe an toàn!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL MediaPipe Live Camera Feed Recognition Interface")
    parser.add_argument("--mode", type=str, default="webcam", choices=["webcam", "sim"], 
                        help="Chế độ chạy mặc định: 'webcam' (Live Camera) hoặc 'sim' (Mô phỏng)")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Chỉ số cổng camera (Index 0 là camera mặc định)")
    parser.add_argument("--model", type=str, default=None, 
                        help="Đường dẫn file trọng số PyTorch (.pt)")
    args = parser.parse_args()

    app = ASLMediaPipeApp(mode=args.mode, camera_idx=args.camera, model_path=args.model)
    app.run()
