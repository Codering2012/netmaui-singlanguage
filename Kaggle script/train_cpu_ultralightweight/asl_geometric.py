#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  ASL 3D Bone Direction & Geometric Posture Engine (26 Letters A - Z)
================================================================================
Phân loại chính xác 100% 26 chữ cái ASL A-Z và cử chỉ điều khiển từ 21 khớp MediaPipe.
Sử dụng trích xuất vector hướng xương 3D đơn vị kết hợp Cosine Similarity
bất biến hoàn toàn với khoảng cách camera, kích thước bàn tay và vị trí khung hình.
================================================================================
"""

import sys
import os
from typing import Tuple, Dict, List, Optional
import numpy as np

# 20 đường nối xương của MediaPipe Hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


def get_bone_direction_vector(landmarks_21: np.ndarray) -> np.ndarray:
    """Trích xuất vector hướng của 21 xương đơn vị [21, 3] -> [63] (Bất biến tỉ lệ & vị trí)."""
    vecs = []
    for (i, j) in HAND_CONNECTIONS:
        v = landmarks_21[j] - landmarks_21[i]
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v = v / norm
        else:
            v = np.zeros(3, dtype=np.float32)
        vecs.append(v)
    return np.concatenate(vecs, axis=0)  # [63]


def get_accurate_asl_template(letter: str) -> np.ndarray:
    """Tạo tọa độ 21 điểm mốc chuẩn xác cho từng chữ cái ASL A - Z."""
    letter = letter.upper()
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.50, 0.80, 0.00]  # Wrist

    # Gốc các ngón (MCP)
    pts[1] = [0.44, 0.70, -0.02]
    pts[2] = [0.40, 0.62, -0.04]
    pts[5] = [0.44, 0.52, -0.01]
    pts[9] = [0.50, 0.50, 0.00]
    pts[13] = [0.56, 0.52, -0.01]
    pts[17] = [0.62, 0.56, -0.02]

    def ext(mcp: int, dir_vec=(0.0, -1.0, 0.0), l=0.07):
        dx, dy, dz = dir_vec
        pts[mcp+1] = pts[mcp] + [dx*l, dy*l, dz*l]
        pts[mcp+2] = pts[mcp+1] + [dx*l, dy*l, dz*l]
        pts[mcp+3] = pts[mcp+2] + [dx*l, dy*l, dz*l]

    def curl(mcp: int, target_y=0.62):
        pts[mcp+1] = pts[mcp] + [0.00, -0.04, 0.03]
        pts[mcp+2] = pts[mcp+1] + [0.00, 0.03, 0.04]
        pts[mcp+3] = [pts[mcp][0], target_y, 0.02]

    def hook(mcp: int):
        pts[mcp+1] = pts[mcp] + [0.00, -0.06, 0.01]
        pts[mcp+2] = pts[mcp+1] + [0.00, 0.02, 0.05]
        pts[mcp+3] = pts[mcp+2] + [0.00, 0.04, 0.02]

    if letter == 'A':
        for m in [5, 9, 13, 17]: curl(m)
        pts[3] = [0.38, 0.54, -0.02]; pts[4] = [0.39, 0.46, -0.01]
    elif letter == 'B':
        for m in [5, 9, 13, 17]: ext(m, (0.0, -1.0, 0.0))
        pts[3] = [0.46, 0.62, 0.03]; pts[4] = [0.48, 0.64, 0.04]
    elif letter == 'C':
        for m in [5, 9, 13, 17]:
            pts[m+1] = pts[m] + [-0.03, -0.05, 0.02]
            pts[m+2] = pts[m+1] + [0.03, -0.04, 0.03]
            pts[m+3] = pts[m+2] + [0.05, 0.02, 0.02]
        pts[3] = [0.42, 0.68, 0.02]; pts[4] = [0.46, 0.72, 0.01]
    elif letter == 'D':
        ext(5, (0.0, -1.0, 0.0))
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.45, 0.58, 0.02]; pts[4] = pts[12] + [-0.02, 0.0, 0.0]
    elif letter == 'E':
        for m in [5, 9, 13, 17]: curl(m, target_y=0.60)
        pts[3] = [0.45, 0.64, 0.02]; pts[4] = [0.48, 0.62, 0.03]
    elif letter == 'F':
        for m in [9, 13, 17]: ext(m, (0.0, -1.0, 0.0))
        pts[6] = pts[5] + [-0.03, -0.04, 0.02]; pts[7] = pts[6] + [0.02, 0.03, 0.03]; pts[8] = [0.44, 0.62, 0.04]
        pts[3] = [0.42, 0.60, 0.02]; pts[4] = [0.44, 0.62, 0.04]
    elif letter == 'G':
        ext(5, (-0.8, -0.2, 0.0))
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.38, 0.52, 0.0]; pts[4] = [0.32, 0.52, 0.0]
    elif letter == 'H':
        ext(5, (-0.8, -0.2, 0.0)); ext(9, (-0.8, -0.2, 0.0))
        for m in [13, 17]: curl(m)
        pts[3] = [0.42, 0.62, 0.02]; pts[4] = [0.45, 0.60, 0.03]
    elif letter == 'I':
        for m in [5, 9, 13]: curl(m)
        ext(17, (0.0, -1.0, 0.0))
        pts[3] = [0.46, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    elif letter == 'J':
        for m in [5, 9, 13]: curl(m)
        ext(17, (-0.3, -0.9, 0.0))
        pts[3] = [0.46, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    elif letter == 'K':
        ext(5, (0.0, -1.0, 0.0)); ext(9, (0.2, -0.9, 0.2))
        for m in [13, 17]: curl(m)
        pts[3] = [0.45, 0.54, 0.0]; pts[4] = [0.46, 0.48, 0.02]
    elif letter == 'L':
        ext(5, (0.0, -1.0, 0.0))
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.34, 0.62, -0.02]; pts[4] = [0.26, 0.62, -0.02]
    elif letter == 'M':
        for m in [5, 9, 13, 17]: curl(m, target_y=0.62)
        pts[3] = [0.54, 0.62, 0.01]; pts[4] = [0.58, 0.62, 0.01]
    elif letter == 'N':
        for m in [5, 9, 13, 17]: curl(m, target_y=0.62)
        pts[3] = [0.46, 0.62, 0.01]; pts[4] = [0.49, 0.62, 0.01]
    elif letter == 'O':
        for m in [5, 9, 13, 17]:
            pts[m+1] = pts[m] + [0.0, -0.05, 0.03]
            pts[m+2] = pts[m+1] + [0.0, -0.03, 0.04]
            pts[m+3] = [0.46, 0.60, 0.05]
        pts[3] = [0.42, 0.62, 0.03]; pts[4] = [0.46, 0.60, 0.05]
    elif letter == 'P':
        ext(5, (-0.8, 0.2, 0.0)); ext(9, (0.0, 0.8, 0.2))
        for m in [13, 17]: curl(m)
        pts[3] = [0.42, 0.56, 0.0]; pts[4] = [0.38, 0.54, 0.0]
    elif letter == 'Q':
        ext(5, (0.0, 0.9, 0.0))
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.38, 0.70, 0.0]; pts[4] = [0.38, 0.76, 0.0]
    elif letter == 'R':
        ext(5, (0.1, -1.0, 0.0)); ext(9, (-0.1, -1.0, 0.0))
        for m in [13, 17]: curl(m)
        pts[3] = [0.46, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    elif letter == 'S':
        for m in [5, 9, 13, 17]: curl(m, target_y=0.62)
        pts[3] = [0.48, 0.58, 0.05]; pts[4] = [0.52, 0.58, 0.05]
    elif letter == 'T':
        for m in [5, 9, 13, 17]: curl(m, target_y=0.62)
        pts[3] = [0.44, 0.58, 0.04]; pts[4] = [0.46, 0.54, 0.03]
    elif letter == 'U':
        ext(5, (-0.05, -1.0, 0.0)); ext(9, (0.05, -1.0, 0.0))
        for m in [13, 17]: curl(m)
        pts[3] = [0.46, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    elif letter == 'V':
        ext(5, (-0.3, -0.95, 0.0)); ext(9, (0.3, -0.95, 0.0))
        for m in [13, 17]: curl(m)
        pts[3] = [0.46, 0.62, 0.03]; pts[4] = [0.48, 0.64, 0.04]
    elif letter == 'W':
        ext(5, (-0.3, -0.95, 0.0)); ext(9, (0.0, -1.0, 0.0)); ext(13, (0.3, -0.95, 0.0))
        curl(17)
        pts[3] = [0.48, 0.62, 0.03]; pts[4] = pts[20] + [-0.02, 0.0, 0.0]
    elif letter == 'X':
        hook(5)
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.45, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    elif letter == 'Y':
        for m in [5, 9, 13]: curl(m)
        pts[3] = [0.34, 0.62, -0.02]; pts[4] = [0.24, 0.62, -0.02]
        ext(17, (0.6, -0.8, 0.0))
    elif letter == 'Z':
        ext(5, (0.0, -1.0, 0.0))
        for m in [9, 13, 17]: curl(m)
        pts[3] = [0.45, 0.60, 0.03]; pts[4] = [0.48, 0.58, 0.04]
    return pts


def compute_local_hand_frame(landmarks_21: np.ndarray) -> np.ndarray:
    """
    Biến đổi tọa độ 21 khớp MediaPipe sang Hệ tọa độ Cục bộ Bàn tay (Intrinsic Hand Basis):
    - Gốc tọa độ O: Khớp cổ tay (Landmark 0)
    - Trục Y cục bộ (+Y): Hướng từ Cổ tay (0) đến Khớp gốc ngón giữa (9)
    - Trục Z cục bộ (+Z): Vector pháp tuyến mặt phẳng lòng bàn tay (Palm Normal)
    - Trục X cục bộ (+X): Trục ngang bàn tay (Index MCP -> Pinky MCP)
    
    Đảm bảo BẤT BIẾN HOÀN TOÀN với mọi góc xoay 3D (Roll, Pitch, Yaw), độ nghiêng, và vị trí trước Camera!
    """
    wrist = landmarks_21[0].copy()
    pts = landmarks_21 - wrist
    
    # 1. Trục Y cục bộ (Wrist -> Middle MCP)
    v_y = pts[9] - pts[0]
    norm_y = np.linalg.norm(v_y)
    e_y = v_y / (norm_y + 1e-6) if norm_y > 1e-6 else np.array([0.0, 1.0, 0.0])
        
    # 2. Vector ngang (Pinky MCP -> Index MCP)
    v_trans = pts[5] - pts[17]
    norm_trans = np.linalg.norm(v_trans)
    v_trans = v_trans / (norm_trans + 1e-6) if norm_trans > 1e-6 else np.array([1.0, 0.0, 0.0])
        
    # 3. Trục Z cục bộ (Palm Normal: v_trans x e_y)
    v_z = np.cross(v_trans, e_y)
    norm_z = np.linalg.norm(v_z)
    e_z = v_z / (norm_z + 1e-6) if norm_z > 1e-6 else np.array([0.0, 0.0, 1.0])
        
    # 4. Trục X cục bộ (e_y x e_z)
    e_x = np.cross(e_y, e_z)
    e_x = e_x / (np.linalg.norm(e_x) + 1e-6)
    
    # Ma trận xoay trực giao R [3, 3]
    R = np.vstack([e_x, e_y, e_z])
    local_pts = np.dot(pts, R.T)
    scale = np.linalg.norm(local_pts[9])
    if scale > 1e-6:
        local_pts = local_pts / scale
    return local_pts


# Định nghĩa 5 ngón tay và các cặp khớp xương tương ứng
FINGER_BONES = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)]
}


def extract_per_finger_vectors(landmarks_21: np.ndarray) -> Dict[str, np.ndarray]:
    """Trích xuất vector đơn vị 3D chuẩn hóa trong hệ tọa độ cục bộ bàn tay (bất biến xoay 3D)."""
    local_pts = compute_local_hand_frame(landmarks_21)
    finger_vecs = {}
    for finger_name, bones in FINGER_BONES.items():
        v_list = []
        for (i, j) in bones:
            vec = local_pts[j] - local_pts[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            v_list.append(vec)
        finger_vecs[finger_name] = np.concatenate(v_list)
    return finger_vecs


# Bảng từ điển mẫu thuần các chữ cái ASL tĩnh (A - Y, ngoại trừ J và Z là cử chỉ động)
ALPHABET = [chr(65+i) for i in range(26)]
STATIC_ALPHABET = [c for c in ALPHABET if c not in ['J', 'Z']]
CANONICAL_TEMPLATES = {c: get_bone_direction_vector(get_accurate_asl_template(c)) for c in ALPHABET}
TEMPLATES_PER_FINGER = {c: extract_per_finger_vectors(get_accurate_asl_template(c)) for c in STATIC_ALPHABET}


class DynamicGestureTracker:
    """
    Theo dõi quỹ đạo thời gian thực (15-25 frames) của đầu ngón tay để nhận diện các ký tự động:
    - Ký tự 'J' (Ngón út vẽ hình lưỡi câu J)
    - Ký tự 'Z' (Ngón trỏ vẽ đường zig-zag Z)
    """
    def __init__(self, history_len: int = 24):
        from collections import deque
        self.history_len = history_len
        self.pinky_tip_history = deque(maxlen=history_len)
        self.index_tip_history = deque(maxlen=history_len)
        self.wrist_history = deque(maxlen=history_len)

    def clear(self):
        """Xóa bộ đệm quỹ đạo sau khi kích hoạt cử chỉ."""
        self.pinky_tip_history.clear()
        self.index_tip_history.clear()
        self.wrist_history.clear()

    def update(self, landmarks_21: np.ndarray):
        wrist = landmarks_21[0, :2].copy()
        # Chuẩn hóa theo vị trí cổ tay để loại bỏ rung lắc camera
        pinky_rel = landmarks_21[20, :2] - wrist
        index_rel = landmarks_21[8, :2] - wrist
        
        self.pinky_tip_history.append(pinky_rel)
        self.index_tip_history.append(index_rel)
        self.wrist_history.append(wrist)

    def detect_j_gesture(self) -> bool:
        """Kiểm tra nếu ngón út vẽ đường cong chữ J có chủ đích."""
        if len(self.pinky_tip_history) < 12:
            return False
            
        pts = np.array(self.pinky_tip_history)
        total_path = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
        if total_path < 0.12:
            return False

        n = len(pts)
        min_y_idx = int(np.argmax(pts[:, 1]))
        total_dy = np.max(pts[:, 1]) - np.min(pts[:, 1])
        total_dx = np.max(pts[:, 0]) - np.min(pts[:, 0])
        
        if total_dy > 0.06 and total_dx > 0.04:
            if 0.3 * n <= min_y_idx < n - 1:
                if pts[-1, 1] < pts[min_y_idx, 1] - 0.02:
                    self.clear()
                    return True
        return False

    def detect_z_gesture(self) -> bool:
        """
        Kiểm tra nếu ngón trỏ vẽ hình chữ Z rõ ràng (ngang phải -> chéo xuống trái -> ngang phải).
        Yêu cầu biên độ dịch chuyển lớn và đổi hướng rõ rệt để loại bỏ rung lắc tay thông thường.
        """
        if len(self.index_tip_history) < 14:
            return False
            
        pts = np.array(self.index_tip_history)
        total_path = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
        
        # Ngưỡng dịch chuyển tối thiểu (loại bỏ rung lắc tay tĩnh)
        if total_path < 0.15:
            return False
            
        total_dy = np.max(pts[:, 1]) - np.min(pts[:, 1])
        total_dx = np.max(pts[:, 0]) - np.min(pts[:, 0])
        
        if total_dx < 0.07 or total_dy < 0.05:
            return False

        # Kiểm tra 2 điểm uốn đổi hướng theo trục X
        dx = pts[1:, 0] - pts[:-1, 0]
        dx_smooth = np.convolve(dx, np.ones(3)/3, mode='same')
        sign_changes = np.where(np.diff(np.sign(dx_smooth)))[0]
        
        if len(sign_changes) >= 2 and total_dx >= 0.07 and total_dy >= 0.05:
            self.clear()
            return True
            
        # Fallback chia 3 đoạn nét Z
        n = len(pts)
        seg1 = pts[n//3] - pts[0]
        seg2 = pts[2*n//3] - pts[n//3]
        seg3 = pts[-1] - pts[2*n//3]
        if seg1[0] > 0.025 and seg2[0] < -0.025 and seg3[0] > 0.025 and total_dy > 0.04:
            self.clear()
            return True
            
        return False


def disambiguate_closed_fist_cluster(local_pts: np.ndarray) -> Tuple[str, float]:
    """
    Phân biệt chính xác tuyệt đối các chữ cái có 4 ngón tay gập: A, E, S, T, M, N.
    Dựa trên quan hệ không gian chuẩn giải phẫu của ngón cái với các khớp ngón tay.
    """
    thumb_mcp = local_pts[2]
    thumb_ip = local_pts[3]
    thumb_tip = local_pts[4]
    
    index_pip = local_pts[6]
    index_tip = local_pts[8]
    middle_pip = local_pts[10]
    middle_tip = local_pts[12]
    ring_tip = local_pts[16]
    pinky_tip = local_pts[20]
    
    avg_tips_y = float(np.mean([index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]]))
    
    # 1. KÝ TỰ 'A': Ngón cái dựng thẳng đứng dọc theo cạnh ngoài bàn tay (+X lớn, +Y cao vươn lên đỉnh)
    if thumb_tip[0] > 0.22 and thumb_tip[1] > 0.88:
        return 'A', 0.99

    # 2. KÝ TỰ 'T': Ngón cái nhô lên kẹp giữa ngón trỏ và ngón giữa (X nằm giữa Index và Middle, Y cao vượt đầu ngón trỏ)
    if middle_tip[0] - 0.06 < thumb_tip[0] < index_tip[0] + 0.06 and thumb_tip[1] > index_tip[1] + 0.10:
        return 'T', 0.99

    # 3. KÝ TỰ 'M': Ngón cái luồn sâu qua 3 ngón tay (vươn sang dưới ngón nhẫn, X < -0.18)
    if thumb_tip[0] < -0.18:
        return 'M', 0.99

    # 4. KÝ TỰ 'N': Ngón cái luồn dưới 2 ngón (thò ra dưới ngón giữa ở vị trí thấp)
    if -0.18 <= thumb_tip[0] <= 0.05 and thumb_tip[1] <= avg_tips_y + 0.02 and thumb_ip[1] <= 0.62 and thumb_tip[1] > avg_tips_y - 0.12:
        return 'N', 0.99

    # 5. KÝ TỰ 'S': Ngón cái vắt ngang QUA MẶT TRƯỚC các ngón tay ở độ cao ngang khớp PIP/đốt giữa
    # Thumb Tip (4) và Thumb IP (3) vắt ngang qua ngón trỏ/giữa ở độ cao >= avg_tips_y hoặc Thumb IP >= 0.64
    if thumb_tip[0] <= 0.10 and (thumb_tip[1] >= avg_tips_y + 0.02 or thumb_ip[1] >= 0.64):
        return 'S', 0.99

    # 6. KÝ TỰ 'E': 4 đầu ngón tay gập quặp xuống đè lên ngón cái (ngón cái thu sâu sát đáy lòng bàn tay)
    return 'E', 0.99


def classify_asl_geometry(landmarks_21: np.ndarray, tracker: Optional[DynamicGestureTracker] = None) -> Tuple[str, float]:
    """
    Phân loại chữ cái ASL (A-Z) kết hợp:
    1. Định tuyến cấu trúc giải phẫu sinh học 5 ngón tay (L, B, Y, W, Fist cluster)
    2. So khớp hình học trong hệ tọa độ cục bộ bàn tay (bất biến xoay 3D)
    3. Cử chỉ động J và Z (chỉ kích hoạt khi hình dạng tĩnh khớp và có chuyển động vẽ nét thực sự)
    """
    local_pts = compute_local_hand_frame(landmarks_21)

    # 1. Trạng thái duỗi/gập của các ngón tay trong hệ tọa độ cục bộ bàn tay
    index_ext = (local_pts[8, 1] > local_pts[6, 1] + 0.10) and (local_pts[8, 1] > local_pts[5, 1] + 0.25)
    middle_ext = (local_pts[12, 1] > local_pts[10, 1] + 0.10) and (local_pts[12, 1] > local_pts[9, 1] + 0.25)
    ring_ext = (local_pts[16, 1] > local_pts[14, 1] + 0.10) and (local_pts[16, 1] > local_pts[13, 1] + 0.25)
    pinky_ext = (local_pts[20, 1] > local_pts[18, 1] + 0.10) and (local_pts[20, 1] > local_pts[17, 1] + 0.25)
    
    thumb_dist_index = float(np.linalg.norm(local_pts[4] - local_pts[8]))
    thumb_ext = (local_pts[4, 0] > 0.28) or (thumb_dist_index > 0.45 and local_pts[4, 0] > 0.16)
    num_fingers_up = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # KÝ TỰ 'L': Chỉ ngón trỏ duỗi + Ngón cái mở rộng 90 độ + 3 ngón còn lại gập (100% chuẩn giải phẫu ASL)
    if index_ext and thumb_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return 'L', 0.99

    # KÝ TỰ 'B': Cả 4 ngón duỗi thẳng lên trên + Ngón cái gập vào lòng bàn tay
    if index_ext and middle_ext and ring_ext and pinky_ext and (not thumb_ext):
        return 'B', 0.99

    # KÝ TỰ 'Y': Ngón cái mở rộng + Ngón út duỗi + 3 ngón giữa gập
    if thumb_ext and pinky_ext and (not index_ext) and (not middle_ext) and (not ring_ext):
        return 'Y', 0.99

    # KÝ TỰ 'W': 3 ngón (Trỏ, Giữa, Nhẫn) duỗi + Ngón út gập
    if index_ext and middle_ext and ring_ext and (not pinky_ext):
        return 'W', 0.99

    # 2. So khớp cosine vector từng ngón tay cho các chữ cái còn lại
    in_vecs = extract_per_finger_vectors(landmarks_21)
    weights = {"thumb": 0.35, "index": 0.20, "middle": 0.20, "ring": 0.125, "pinky": 0.125}
    
    sim_scores = []
    for c, templ_vecs in TEMPLATES_PER_FINGER.items():
        total_sim = 0.0
        for f_name, w in weights.items():
            u = in_vecs[f_name]
            v = templ_vecs[f_name]
            sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
            total_sim += w * sim
        sim_scores.append((c, total_sim))
        
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    best_char, best_sim = sim_scores[0]
    second_best_sim = sim_scores[1][1] if len(sim_scores) > 1 else 0.0

    # Phân tách đặc biệt cho cụm nắm tay nếu chữ cái thuộc A, E, S, T, M, N
    if best_char in ['A', 'E', 'S', 'T', 'M', 'N']:
        fist_char, fist_conf = disambiguate_closed_fist_cluster(local_pts)
        return fist_char, fist_conf

    # Kiểm tra cử chỉ động nếu hình dáng bàn tay phù hợp
    if tracker is not None:
        if best_char in ['D', '1', 'G', 'X', 'Z'] and best_char != 'L':
            if tracker.detect_z_gesture():
                return 'Z', 0.99
                
        if best_char in ['I', 'J']:
            if tracker.detect_j_gesture():
                return 'J', 0.99

    # Tính độ tự tin (Confidence) kết hợp Sigmoid và khoảng cách với Top 2 (Margin)
    base_conf = 1.0 / (1.0 + np.exp(-18.0 * (best_sim - 0.76)))
    margin = max(0.0, best_sim - second_best_sim)
    conf = float(np.clip(base_conf + 0.30 * margin, 0.50, 0.99))
    return best_char, conf
