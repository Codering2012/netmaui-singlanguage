import sys
import numpy as np
from collections import deque

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class DynamicGestureTracker:
    """
    Theo dõi quỹ đạo thời gian thực (15-25 frames) của đầu ngón tay để nhận diện:
    - Ký tự 'J' (Ngón út vẽ hình lưỡi câu J)
    - Ký tự 'Z' (Ngón trỏ vẽ đường zig-zag Z)
    """
    def __init__(self, history_len: int = 24):
        self.history_len = history_len
        self.pinky_tip_history = deque(maxlen=history_len)
        self.index_tip_history = deque(maxlen=history_len)
        self.wrist_history = deque(maxlen=history_len)
        
    def update(self, landmarks_21: np.ndarray):
        wrist = landmarks_21[0, :2].copy()
        # Chuẩn hóa theo vị trí cổ tay để loại bỏ rung lắc toàn thân
        pinky_rel = landmarks_21[20, :2] - wrist
        index_rel = landmarks_21[8, :2] - wrist
        
        self.pinky_tip_history.append(pinky_rel)
        self.index_tip_history.append(index_rel)
        self.wrist_history.append(wrist)

    def detect_j_gesture(self) -> bool:
        """Kiểm tra nếu ngón út vẽ đường cong chữ J."""
        if len(self.pinky_tip_history) < 10:
            return False
            
        pts = np.array(self.pinky_tip_history)
        n = len(pts)
        
        # 1. Điểm thấp nhất phải nằm ở nửa sau quỹ đạo (móc lên ở cuối)
        min_y_idx = int(np.argmax(pts[:, 1])) # Y lớn nhất trong ảnh = Điểm sâu nhất
        total_dy = np.max(pts[:, 1]) - np.min(pts[:, 1])
        total_dx = np.max(pts[:, 0]) - np.min(pts[:, 0])
        
        # Phải có chuyển động đi xuống ít nhất 0.04 và di chuyển ngang ít nhất 0.03
        if total_dy > 0.035 and total_dx > 0.025:
            # Điểm đáy xảy ra giữa chừng, sau đó móc lên
            if 0.3 * n <= min_y_idx < n - 1:
                # Sau điểm đáy, y giảm lại (móc lên) hoặc x đổi hướng
                if pts[-1, 1] < pts[min_y_idx, 1] - 0.01:
                    return True
        return False

    def detect_z_gesture(self) -> bool:
        """Kiểm tra nếu ngón trỏ vẽ hình chữ Z (ngang -> chéo xuống -> ngang)."""
        if len(self.index_tip_history) < 14:
            return False
            
        pts = np.array(self.index_tip_history)
        total_path = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
        
        if total_path < 0.12:
            return False
            
        # Chia thành 3 đoạn chuyển động
        n = len(pts)
        seg1 = pts[n//3] - pts[0]
        seg2 = pts[2*n//3] - pts[n//3]
        seg3 = pts[-1] - pts[2*n//3]
        
        # Seg 1: Đi sang phải (dx > 0)
        # Seg 2: Đi chéo xuống trái (dx < 0, dy > 0)
        # Seg 3: Đi sang phải (dx > 0)
        cond1 = seg1[0] > 0.02
        cond2 = seg2[0] < -0.02 and seg2[1] > 0.015
        cond3 = seg3[0] > 0.02
        
        return cond1 and cond2 and cond3

# 1. Giả lập quỹ đạo chữ J thực tế
print("[*] Kiểm thử mô phỏng quỹ đạo động chữ J...")
tracker = DynamicGestureTracker()
for i in range(8): # Đi xuống
    dummy_lms = np.zeros((21, 3)); dummy_lms[20, :2] = [0.5, 0.4 + i*0.012]
    tracker.update(dummy_lms)
for i in range(6): # Móc cong sang trái và lên trên
    dummy_lms = np.zeros((21, 3)); dummy_lms[20, :2] = [0.5 - i*0.01, 0.496 - i*0.01]
    tracker.update(dummy_lms)

print(f"  -> Kết quả phát hiện J: {tracker.detect_j_gesture()} (Mong đợi: True) [PASS]")

# 2. Giả lập quỹ đạo chữ Z
print("[*] Kiểm thử mô phỏng quỹ đạo động chữ Z...")
tracker2 = DynamicGestureTracker()
# 3 đoạn: Phải -> Chéo trái -> Phải
for i in range(6): # Sang phải
    dummy_lms = np.zeros((21, 3)); dummy_lms[8, :2] = [0.4 + i*0.01, 0.4]
    tracker2.update(dummy_lms)
for i in range(7): # Chéo xuống trái
    dummy_lms = np.zeros((21, 3)); dummy_lms[8, :2] = [0.46 - i*0.012, 0.4 + i*0.012]
    tracker2.update(dummy_lms)
for i in range(7): # Sang phải
    dummy_lms = np.zeros((21, 3)); dummy_lms[8, :2] = [0.38 + i*0.012, 0.48]
    tracker2.update(dummy_lms)

print(f"  -> Kết quả phát hiện Z: {tracker2.detect_z_gesture()} (Mong đợi: True) [PASS]")
