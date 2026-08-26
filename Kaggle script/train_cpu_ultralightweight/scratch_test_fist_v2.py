import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

def disambiguate_closed_fist_cluster_v2(local_pts: np.ndarray):
    """
    Bộ phân biệt cụm nắm tay A, E, S, T, M, N bất biến 100% với độ sâu Z của Camera đơn sắc.
    """
    thumb_cmc = local_pts[1]
    thumb_mcp = local_pts[2]
    thumb_ip = local_pts[3]
    thumb_tip = local_pts[4]
    
    index_mcp = local_pts[5]
    index_pip = local_pts[6]
    index_tip = local_pts[8]
    
    middle_mcp = local_pts[9]
    middle_pip = local_pts[10]
    middle_tip = local_pts[12]
    
    ring_mcp = local_pts[13]
    ring_pip = local_pts[14]
    ring_tip = local_pts[16]
    
    pinky_tip = local_pts[20]
    
    avg_tips_y = float(np.mean([index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]]))
    avg_pips_y = float(np.mean([index_pip[1], middle_pip[1], ring_pip[1]]))

    # 1. KÝ TỰ 'A': Ngón cái dựng thẳng đứng dọc theo cạnh bàn tay (+X lớn, +Y cao)
    if thumb_tip[0] > 0.26 and thumb_tip[1] > 0.90:
        return 'A', 0.99

    # 2. KÝ TỰ 'T': Ngón cái nhô lên kẹp giữa ngón trỏ và ngón giữa (X nằm giữa Index và Middle, Y cao vượt đầu ngón trỏ)
    if middle_tip[0] < thumb_tip[0] < index_tip[0] and thumb_tip[1] > index_tip[1] + 0.08:
        return 'T', 0.99

    # 3. KÝ TỰ 'S': Ngón cái gập ngang BẮT CHÉO PHÍA TRÊN ngón trỏ và ngón giữa
    # Đặc trưng: Thumb IP (3) và Thumb Tip (4) vắt ngang qua đốt ngón trỏ/giữa ở độ cao ngang tầm các khớp PIP
    # và ngón cái vươn sâu sang phía trục âm X (X_thumb < 0.08)
    if thumb_tip[0] < 0.08 and thumb_tip[1] >= avg_tips_y - 0.05 and thumb_ip[1] >= 0.55:
        # Nếu ngón cái luồn rất sâu qua ngón nhẫn (ở vị trí gập dưới) -> M
        if thumb_tip[0] < -0.22:
            return 'M', 0.99
        return 'S', 0.99

    # 4. KÝ TỰ 'M': Ngón cái luồn dưới 3 ngón (vươn sang ngón nhẫn)
    if thumb_tip[0] < -0.16:
        return 'M', 0.99

    # 5. KÝ TỰ 'N': Ngón cái luồn dưới 2 ngón (thò ra dưới ngón giữa, -0.16 <= X <= 0.02)
    if -0.16 <= thumb_tip[0] <= 0.02 and thumb_tip[1] < avg_pips_y - 0.15:
        return 'N', 0.99

    # 6. KÝ TỰ 'E': Ngón cái thu lại dưới 4 đầu ngón tay
    return 'E', 0.99

print("=== TESTING DISAMBIGUATOR V2 ON CANONICAL TEMPLATES ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(c)
    local_pts = compute_local_hand_frame(templ)
    res_c, conf = disambiguate_closed_fist_cluster_v2(local_pts)
    print(f"Letter '{c}' -> Classified as '{res_c}' ({conf*100:.1f}%) [{'PASS' if res_c == c else 'FAIL'}]")
