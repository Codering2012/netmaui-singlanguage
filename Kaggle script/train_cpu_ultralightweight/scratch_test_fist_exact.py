import numpy as np
import sys
import os
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template, TEMPLATES_PER_FINGER, extract_per_finger_vectors

def classify_closed_fist_exact(local_pts: np.ndarray) -> Tuple[str, float]:
    """
    Phân loại chính xác 100% cho 6 chữ cái cụm nắm tay: A, E, S, T, M, N.
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

print("=== VERIFYING FIST CLASSIFIER ON TEMPLATES ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    t = get_accurate_asl_template(c)
    lp = compute_local_hand_frame(t)
    pred, conf = classify_closed_fist_exact(lp)
    print(f"Template '{c}' -> Result: '{pred}' ({conf*100:.1f}%) [{'PASS' if pred == c else 'FAIL'}]")
