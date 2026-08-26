import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

def disambiguate_closed_fist_cluster(local_pts):
    thumb_tip = local_pts[4]
    index_tip = local_pts[8]
    middle_tip = local_pts[12]
    ring_tip = local_pts[16]
    pinky_tip = local_pts[20]
    
    avg_tips_y = float(np.mean([index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]]))
    avg_tips_z = float(np.mean([index_tip[2], middle_tip[2], ring_tip[2], pinky_tip[2]]))

    # 1. A: Ngón cái dựng thẳng dọc theo cạnh bàn tay (+X lớn, +Y cao)
    if thumb_tip[0] > 0.26 and thumb_tip[1] > 0.90:
        return 'A', 0.99

    # 2. T: Ngón cái nhô lên kẹp giữa ngón trỏ và ngón giữa
    if middle_tip[0] < thumb_tip[0] < index_tip[0] and thumb_tip[1] > avg_tips_y + 0.10:
        return 'T', 0.99

    # 3. S: Ngón cái đè lên trên mặt trước các ngón tay (Z cao hơn hẳn mặt trước ngón tay)
    if thumb_tip[2] > avg_tips_z + 0.06 and thumb_tip[1] >= avg_tips_y - 0.05:
        return 'S', 0.99

    # 4. M: Ngón cái luồn sâu sang phía ngón nhẫn (X < -0.15)
    if thumb_tip[0] < -0.15:
        return 'M', 0.99
        
    # 5. N: Ngón cái luồn dưới ngón trỏ và giữa (-0.15 <= X <= 0.04)
    if -0.15 <= thumb_tip[0] <= 0.04 and thumb_tip[1] >= avg_tips_y - 0.08:
        return 'N', 0.99

    # 6. E: Ngón cái thu lại dưới 4 đầu ngón tay
    return 'E', 0.99

print("=== TESTING DISAMBIGUATOR ON ALL 6 FIST LETTERS ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(c)
    local_pts = compute_local_hand_frame(templ)
    res_c, conf = disambiguate_closed_fist_cluster(local_pts)
    print(f"Template '{c}' -> Classified as: '{res_c}' ({conf*100:.1f}%) [{'PASS' if res_c == c else 'FAIL'}]")
