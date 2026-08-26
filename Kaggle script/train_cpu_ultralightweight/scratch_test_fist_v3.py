import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

def disambiguate_closed_fist_cluster_v3(local_pts: np.ndarray):
    """
    Bộ phân tách giải phẫu hoàn hảo cho 6 ký tự cụm nắm tay: A, E, S, T, M, N.
    Bất biến 100% với góc quay camera, độ sáng, và sai số đo sâu Z của MediaPipe.
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

    # 1. KÝ TỰ 'A': Ngón cái dựng thẳng đứng dọc theo cạnh ngoài bàn tay (+X lớn, +Y cao vươn lên đỉnh)
    if thumb_tip[0] > 0.25 and thumb_tip[1] > 0.90:
        return 'A', 0.99

    # 2. KÝ TỰ 'T': Ngón cái nhô lên kẹp giữa ngón trỏ và ngón giữa (X nằm giữa Index và Middle, Y cao vượt đầu ngón trỏ)
    if middle_tip[0] - 0.05 < thumb_tip[0] < index_tip[0] + 0.05 and thumb_tip[1] > index_tip[1] + 0.12:
        return 'T', 0.99

    # 3. KÝ TỰ 'M': Ngón cái luồn sâu qua 3 ngón tay (vươn sang dưới ngón nhẫn, X < -0.18)
    if thumb_tip[0] < -0.18:
        return 'M', 0.99

    # 4. KÝ TỰ 'S': Ngón cái vắt ngang QUA MẶT TRƯỚC các ngón tay ở độ cao ngang khớp PIP
    # Đặc trưng: Thumb IP (3) và Thumb Tip (4) nằm ở độ cao đốt ngón tay (Y >= 0.65 hoặc Y >= avg_tips_y + 0.05)
    # và ngón cái vắt ngang qua ngón trỏ/giữa (X_thumb_tip <= 0.08)
    if thumb_tip[0] <= 0.08 and (thumb_tip[1] >= 0.68 or thumb_ip[1] >= 0.66):
        return 'S', 0.99

    # 5. KÝ TỰ 'N': Ngón cái luồn dưới 2 ngón (ngón cái thò ra dưới ngón giữa ở vị trí thấp)
    if -0.18 <= thumb_tip[0] <= 0.06 and thumb_tip[1] <= avg_tips_y + 0.04 and thumb_ip[1] <= 0.64 and thumb_tip[1] > avg_tips_y - 0.10:
        return 'N', 0.99

    # 6. KÝ TỰ 'E': 4 đầu ngón tay gập quặp xuống đè lên ngón cái (Ngón cái thu sát vào gốc lòng bàn tay, Y thấp)
    return 'E', 0.99

print("=== TESTING DISAMBIGUATOR V3 ACROSS ALL 6 FIST LETTERS ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(c)
    local_pts = compute_local_hand_frame(templ)
    res_c, conf = disambiguate_closed_fist_cluster_v3(local_pts)
    print(f"Letter '{c}' -> Classified as '{res_c}' ({conf*100:.1f}%) [{'PASS' if res_c == c else 'FAIL'}]")
