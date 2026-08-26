import numpy as np

def classify_finger_extension_states(local_pts):
    """
    Xác định trạng thái Duỗi (Extended) hay Gập (Curled) của từng ngón tay trong hệ tọa độ cục bộ bàn tay chuẩn hóa.
    local_pts: 21 điểm landmarks trong hệ tọa độ cục bộ bàn tay (e_x: radial, e_y: dọc bàn tay, e_z: pháp tuyến).
    """
    # Chiều dài trục dọc bàn tay: e_y hướng từ Cổ tay (0) lên Khớp gốc ngón giữa (9)
    # Khớp gốc MCP: Index(5), Middle(9), Ring(13), Pinky(17)
    # Khớp đầu ngón Tip: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    
    # 1. Ngón trỏ:
    index_ext = (local_pts[8, 1] > local_pts[6, 1] + 0.15) and (local_pts[8, 1] > local_pts[5, 1] + 0.35)
    
    # 2. Ngón giữa:
    middle_ext = (local_pts[12, 1] > local_pts[10, 1] + 0.15) and (local_pts[12, 1] > local_pts[9, 1] + 0.35)
    
    # 3. Ngón nhẫn:
    ring_ext = (local_pts[16, 1] > local_pts[14, 1] + 0.15) and (local_pts[16, 1] > local_pts[13, 1] + 0.35)
    
    # 4. Ngón út:
    pinky_ext = (local_pts[20, 1] > local_pts[18, 1] + 0.15) and (local_pts[20, 1] > local_pts[17, 1] + 0.35)
    
    # 5. Ngón cái (Thumb): Đo độ mở rộng (Abduction/Extension) theo phương X cục bộ
    # Ngón cái mở rộng sang bên (+X cục bộ > 0.40)
    thumb_ext = (local_pts[4, 0] > 0.40) or (np.linalg.norm(local_pts[4] - local_pts[8]) > 0.55 and local_pts[4, 0] > 0.25)
    
    states = {
        "thumb": bool(thumb_ext),
        "index": bool(index_ext),
        "middle": bool(middle_ext),
        "ring": bool(ring_ext),
        "pinky": bool(pinky_ext)
    }
    return states

# Test on real user L pose
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.58, 0.84, 0.0]
pts[1] = [0.51, 0.82, 0.0]; pts[2] = [0.44, 0.74, 0.0]; pts[3] = [0.39, 0.72, 0.0]; pts[4] = [0.34, 0.72, 0.0]
pts[5] = [0.48, 0.56, 0.0]; pts[6] = [0.47, 0.44, 0.0]; pts[7] = [0.46, 0.36, 0.0]; pts[8] = [0.46, 0.28, 0.0]
pts[9] = [0.52, 0.56, 0.0]; pts[10] = [0.48, 0.60, 0.0]; pts[11] = [0.48, 0.67, 0.0]; pts[12] = [0.50, 0.72, 0.0]
pts[13] = [0.55, 0.58, 0.0]; pts[14] = [0.52, 0.62, 0.0]; pts[15] = [0.52, 0.69, 0.0]; pts[16] = [0.53, 0.72, 0.0]
pts[17] = [0.60, 0.60, 0.0]; pts[18] = [0.59, 0.70, 0.0]; pts[19] = [0.55, 0.65, 0.0]; pts[20] = [0.55, 0.69, 0.0]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame

local_pts = compute_local_hand_frame(pts)
states = classify_finger_extension_states(local_pts)
print("Detected Finger Extension States on User Screenshot L:")
for k, v in states.items():
    print(f"  {k:8s}: {'EXTENDED' if v else 'CURLED'}")

if states["thumb"] and states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
    print("=> 100% UNAMBIGUOUS LETTER 'L'!")
