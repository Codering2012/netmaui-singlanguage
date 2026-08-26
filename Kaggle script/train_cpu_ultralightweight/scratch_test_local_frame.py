import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    
    # 1. Trục Y cục bộ (Longitudinal Axis: Wrist -> Middle MCP)
    v_y = pts[9] - pts[0]
    norm_y = np.linalg.norm(v_y)
    if norm_y < 1e-6:
        e_y = np.array([0.0, 1.0, 0.0])
    else:
        e_y = v_y / norm_y
        
    # 2. Vector ngang (Transverse: Pinky MCP -> Index MCP)
    v_trans = pts[5] - pts[17]
    norm_trans = np.linalg.norm(v_trans)
    if norm_trans < 1e-6:
        v_trans = np.array([1.0, 0.0, 0.0])
    else:
        v_trans = v_trans / norm_trans
        
    # 3. Trục Z cục bộ (Palm Normal: e_y x v_trans)
    v_z = np.cross(v_trans, e_y)
    norm_z = np.linalg.norm(v_z)
    if norm_z < 1e-6:
        e_z = np.array([0.0, 0.0, 1.0])
    else:
        e_z = v_z / norm_z
        
    # 4. Trục X cục bộ (e_y x e_z để đảm bảo trực giao chuẩn)
    e_x = np.cross(e_y, e_z)
    e_x = e_x / (np.linalg.norm(e_x) + 1e-6)
    
    # Ma trận xoay trực giao R [3, 3]
    R = np.vstack([e_x, e_y, e_z]) # [3, 3]
    
    # Chiếu tất cả 21 điểm vào hệ tọa độ cục bộ
    local_pts = np.dot(pts, R.T)
    
    # Chuẩn hóa kích thước bàn tay theo khoảng cách Cổ tay -> Khớp ngón giữa
    scale = np.linalg.norm(local_pts[9])
    if scale > 1e-6:
        local_pts = local_pts / scale
        
    return local_pts

print("[*] Kiểm thử tính Bất biến xoay 3D (Rotation Invariance) của Hệ tọa độ Cục bộ:")
from asl_geometric import get_accurate_asl_template

# Lấy mẫu chữ L chuẩn
l_template = get_accurate_asl_template('L')
local_l_canonical = compute_local_hand_frame(l_template)

# Tạo mẫu chữ L bị xoay nghiêng 45 độ quanh trục Z
theta = np.pi / 4.0
R_rot = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])
l_rotated_45 = np.dot(l_template, R_rot.T)

local_l_rotated = compute_local_hand_frame(l_rotated_45)

diff = np.linalg.norm(local_l_canonical - local_l_rotated)
print(f"  -> Độ lệch giữa mẫu L thẳng đứng và mẫu L nghiêng 45 độ: {diff:.6f}")
if diff < 1e-4:
    print("  [PASS] Hệ tọa độ cục bộ BẤT BIẾN HOÀN TOÀN 100% với góc nghiêng và hướng bàn tay!")
else:
    print("  [FAIL] Vẫn còn độ lệch.")
