import numpy as np

def analyze_fist_topology(landmarks_21):
    """
    Phân biệt chính xác tuyệt đối các chữ cái cụm nắm tay: A, E, S, T, M, N, O
    Dựa trên quan hệ không gian 3D giữa Ngón cái và 4 đầu ngón tay.
    """
    wrist = landmarks_21[0]
    thumb_tip = landmarks_21[4]
    thumb_ip = landmarks_21[3]
    index_mcp = landmarks_21[5]
    index_pip = landmarks_21[6]
    index_tip = landmarks_21[8]
    middle_mcp = landmarks_21[9]
    middle_tip = landmarks_21[12]
    ring_tip = landmarks_21[16]
    pinky_tip = landmarks_21[20]
    
    # 1. Kiểm tra ngón cái ở bên cạnh (A)
    # Ngón cái nằm ngoài cùng bên trái (radial), hướng thẳng đứng
    if thumb_tip[0] < index_mcp[0] - 0.02 and thumb_tip[1] < index_mcp[1] + 0.05:
        return 'A'
        
    # 2. Kiểm tra ngón cái bắt chéo PHÍA TRÊN ngón tay (S)
    # Ngón cái đè lên trên các đốt ngón trỏ và ngón giữa, đầu ngón cái vươn sang ngón nhẫn
    if thumb_tip[0] > middle_mcp[0] and thumb_tip[1] < middle_tip[1] - 0.01:
        return 'S'

    # 3. Kiểm tra ngón cái gập DƯỚI 4 đầu ngón tay (E)
    # 4 đầu ngón tay tạo thành hàng ngang tựa trên ngón cái (y_thumb >= y_fingertips)
    avg_tips_y = np.mean([index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]])
    if thumb_tip[1] >= avg_tips_y - 0.02:
        return 'E'
        
    # 4. Kiểm tra ngón cái kẹp giữa ngón trỏ và giữa (T)
    if index_tip[0] < thumb_tip[0] < middle_tip[0] and thumb_tip[1] < index_tip[1]:
        return 'T'

    # 5. M và N
    if thumb_tip[0] > middle_tip[0]:
        return 'M'
    elif thumb_tip[0] > index_tip[0]:
        return 'N'
        
    return 'E'

print("[*] Topology function ready!")
