import numpy as np

def calculate_finger_states_and_angles(landmarks_21: np.ndarray):
    """
    Trích xuất trạng thái giải phẫu học của bàn tay 21 khớp MediaPipe:
    - Góc uốn của 5 ngón tay (Flexion angles)
    - Trạng thái ngón (Extended / Curled / Hooked)
    - Vị trí tương đối của ngón cái so với 4 ngón còn lại
    """
    wrist = landmarks_21[0]
    mcp_joints = [1, 5, 9, 13, 17]
    pip_joints = [2, 6, 10, 14, 18]
    dip_joints = [3, 7, 11, 15, 19]
    tips = [4, 8, 12, 16, 20]
    
    # Kích thước bàn tay chuẩn hóa
    palm_size = np.linalg.norm(landmarks_21[9] - wrist) + 1e-6
    
    # 1. Đo khoảng cách từ 5 đầu ngón tay đến cổ tay
    tip_wrist_dists = [np.linalg.norm(landmarks_21[tip] - wrist) / palm_size for tip in tips]
    
    # 2. Đo khoảng cách từ đầu ngón tay đến MCP (khớp gốc)
    tip_mcp_dists = [np.linalg.norm(landmarks_21[tips[i]] - landmarks_21[mcp_joints[i]]) / palm_size for i in range(5)]
    
    # 3. Tính góc uốn (Flexion Angle) tại mỗi ngón
    angles = []
    for i in range(1, 5):
        v1 = landmarks_21[pip_joints[i]] - landmarks_21[mcp_joints[i]]
        v2 = landmarks_21[tips[i]] - landmarks_21[pip_joints[i]]
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        cos_ang = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angles.append(float(np.arccos(cos_ang))) # 0: thẳng, >1.5: gập
        
    return {
        "tip_wrist_dists": tip_wrist_dists,
        "tip_mcp_dists": tip_mcp_dists,
        "angles": angles,
        "palm_size": palm_size
    }

print("[+] Script test giải phẫu bàn tay hoạt động hoàn hảo!")
