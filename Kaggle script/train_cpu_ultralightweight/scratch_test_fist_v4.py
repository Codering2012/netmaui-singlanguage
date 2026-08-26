import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

def disambiguate_closed_fist_cluster_v4(local_pts: np.ndarray):
    thumb_tip = local_pts[4]
    thumb_ip = local_pts[3]
    thumb_mcp = local_pts[2]
    
    index_pip = local_pts[6]
    index_tip = local_pts[8]
    middle_tip = local_pts[12]
    ring_tip = local_pts[16]
    pinky_tip = local_pts[20]
    
    avg_tips_y = float(np.mean([index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]]))
    
    # 1. KÝ TỰ 'A': Ngón cái dựng thẳng đứng dọc cạnh ngoài bàn tay (+X lớn, +Y cao vươn lên đỉnh)
    if thumb_tip[0] > 0.22 and thumb_tip[1] > 0.86:
        return 'A', 0.99

    # 2. KÝ TỰ 'T': Ngón cái nhô lên kẹp giữa ngón trỏ và ngón giữa (X nằm giữa Index và Middle, Y cao vượt đầu ngón trỏ)
    if middle_tip[0] - 0.05 < thumb_tip[0] < index_tip[0] + 0.05 and thumb_tip[1] > index_tip[1] + 0.10:
        return 'T', 0.99

    # 3. KÝ TỰ 'M': Ngón cái luồn sâu qua 3 ngón tay (vươn sang dưới ngón nhẫn, X < -0.18)
    if thumb_tip[0] < -0.18:
        return 'M', 0.99

    # 4. KÝ TỰ 'N': Ngón cái luồn dưới 2 ngón (thò ra dưới ngón giữa ở vị trí thấp)
    if -0.18 <= thumb_tip[0] <= 0.06 and thumb_tip[1] <= avg_tips_y + 0.02 and thumb_ip[1] <= 0.63 and thumb_tip[1] > avg_tips_y - 0.12:
        return 'N', 0.99

    # 5. KÝ TỰ 'S' vs 'E':
    # - 'S': Ngón cái vắt ngang qua MẶT NGOÀI các ngón tay (Thumb Tip cao hơn hoặc ngang đầu ngón trỏ/giữa, Y_thumb > avg_tips_y)
    # - 'E': Ngón cái thu lại DƯỚI 4 đầu ngón tay (Thumb Tip thấp hơn đầu các ngón tay, Y_thumb <= avg_tips_y)
    if thumb_tip[1] > avg_tips_y + 0.02 or thumb_ip[1] > index_pip[1] - 0.15:
        # Nếu ngón cái vắt ngang qua ngón trỏ/giữa
        if thumb_tip[0] <= 0.15:
            return 'S', 0.99

    return 'E', 0.99

print("=== STRESS TESTING V4 ON E -> A -> S WITH 3D ROTATION & NOISE ===")
np.random.seed(42)
for target in ['E', 'A', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(target)
    pass_cnt = 0
    total = 200
    for _ in range(total):
        theta_z = (np.random.rand() - 0.5) * np.pi * 0.5
        cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        theta_x = (np.random.rand() - 0.5) * np.pi * 0.3
        cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        R = np.dot(R_z, R_x)
        noise = np.random.normal(0, 0.008, size=templ.shape).astype(np.float32)
        noisy_pose = np.dot(templ + noise, R.T)
        
        local_pts = compute_local_hand_frame(noisy_pose)
        pred, conf = disambiguate_closed_fist_cluster_v4(local_pts)
        if pred == target:
            pass_cnt += 1
            
    print(f"Target '{target}': {pass_cnt}/{total} PASS ({pass_cnt/total*100:.1f}%) across random 3D rotations & noise!")
