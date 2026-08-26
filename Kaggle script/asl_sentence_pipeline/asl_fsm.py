"""
TẦNG 1 & 2: ỔN ĐỊNH HOÁ KÝ TỰ THEO THỜI GIAN VÀ PHÂN ĐOẠN TỪ/CÂU (FSM + GESTURE)
Dựa theo Tài liệu kỹ thuật: 'Từ ký tự ngón tay đến câu hoàn chỉnh' (Chương 3 & 4, Phụ lục A & B)
"""

from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any, List


class LetterStabilizerFSM:
    """
    Máy trạng thái ổn định hoá ký tự theo thời gian: IDLE -> HOLD -> COMMIT -> RELEASE.
    
    Giải quyết 4 bài toán cốt lõi:
    1. Trùng lặp theo thời gian: Phải nhả tay (release) mới chốt ký tự tiếp theo.
    2. Nhiễu tức thời: Bỏ phiếu đa số N/M trên cửa sổ trượt (N-of-M voting).
    3. Ký tự đôi (double letters: LL, EE, SS, DD): Giữ nguyên tư thế >= hold_repeat_frames -> phát ký tự thứ 2.
    4. Ranh giới từ & câu: Hạ tay / nghỉ tay -> SPACE hoặc END_SENTENCE.
    """

    def __init__(
        self,
        window: int = 8,               # Cửa sổ trượt M khung hình (0.32s ở 25 FPS)
        min_votes: int = 6,            # Số phiếu tối thiểu N để chốt (N/M = 6/8 = 75%)
        min_conf: float = 0.60,        # Ngưỡng tin cậy softmax
        release_frames: int = 4,       # Số khung hình cần nhả tay để quay lại IDLE (0.16s)
        space_frames: int = 12,        # Nghỉ tay >= 12 khung hình (0.48s) -> KHOẢNG TRẮNG
        hold_repeat_frames: int = 25,  # Giữ nguyên >= 25 khung hình (1.0s) -> KÝ TỰ ĐÔI
        sentence_frames: int = 45,     # Nghỉ dài >= 45 khung hình (1.8s) -> KẾT THÚC CÂU
        smooth_alpha: float = 0.30,    # Hệ số làm mượt hàm mũ softmax
    ):
        self.window = window
        self.min_votes = min_votes
        self.min_conf = min_conf
        self.release_frames = release_frames
        self.space_frames = space_frames
        self.hold_repeat_frames = hold_repeat_frames
        self.sentence_frames = sentence_frames
        self.smooth_alpha = smooth_alpha

        self.buf = deque(maxlen=window)
        self.state = "IDLE"
        self.last_committed = None
        self.rel_count = 0
        self.idle_count = 0
        self.hold_count = 0
        self.space_emitted = False
        self.sentence_emitted = False

    def reset(self):
        """Khởi động lại toàn bộ trạng thái bộ đệm."""
        self.buf.clear()
        self.state = "IDLE"
        self.last_committed = None
        self.rel_count = 0
        self.idle_count = 0
        self.hold_count = 0
        self.space_emitted = False
        self.sentence_emitted = False

    def push(self, letter: Optional[str], conf: float = 1.0) -> Optional[Tuple[str, Optional[str]]]:
        """
        Nhận dự đoán từ mô hình nhận diện ký tự cho mỗi khung hình (frame).
        
        Args:
            letter: Ký tự dự đoán ('A'-'Z') hoặc None nếu không có tay.
            conf: Độ tin cậy softmax (0.0 - 1.0).
            
        Returns:
            None: Chưa có sự kiện.
            ("LETTER", 'H'): Ký tự đã được chốt chắc chắn.
            ("SPACE", None): Khoảng trắng (nghỉ giữa 2 từ).
            ("END", None): Kết thúc câu (nghỉ dài).
        """
        is_low_conf = (letter is None) or (conf < self.min_conf)

        # --- 1. Theo dõi thời gian nghỉ tay (Pause / Idle Tracking) ---
        if is_low_conf:
            self.idle_count += 1
        else:
            self.idle_count = 0
            self.space_emitted = False
            self.sentence_emitted = False

        # Phát hiện khoảng trắng (trễ 1 lần)
        if self.idle_count >= self.space_frames and not self.space_emitted:
            self.space_emitted = True
            self.state = "IDLE"
            self.last_committed = None
            self.buf.clear()
            return ("SPACE", None)

        # Phát hiện kết thúc câu (trễ 1 lần)
        if self.idle_count >= self.sentence_frames and not self.sentence_emitted:
            self.sentence_emitted = True
            self.reset()
            return ("END", None)

        # --- 2. Bỏ phiếu đa số trên cửa sổ trượt (N-of-M Voting) ---
        self.buf.append(None if is_low_conf else letter)
        counts = Counter(x for x in self.buf if x is not None)
        top_letter, votes = (counts.most_common(1)[0] if counts else (None, 0))
        is_stable = (top_letter is not None) and (votes >= self.min_votes)

        # --- 3. Máy trạng thái (FSM) ---
        if self.state == "RELEASE":
            # Đã xuất 1 ký tự, kiểm tra xem tay đã nhả chưa
            if is_low_conf or (is_stable and top_letter != self.last_committed):
                self.rel_count += 1
                if self.rel_count >= self.release_frames:
                    self.state = "IDLE"
                    self.rel_count = 0
            elif is_stable and top_letter == self.last_committed:
                # Tay vẫn giữ nguyên ký tự vừa xuất -> đếm thời gian giữ để xuất ký tự đôi (LL, EE...)
                self.hold_count += 1
                if self.hold_count >= self.hold_repeat_frames:
                    self.hold_count = 0
                    self.rel_count = 0
                    return ("LETTER", self.last_committed)
            return None

        # Trạng thái IDLE: chờ đủ phiếu để chốt (COMMIT)
        if is_stable:
            self.state = "RELEASE"
            self.last_committed = top_letter
            self.rel_count = 0
            self.hold_count = 0
            self.buf.clear()
            return ("LETTER", top_letter)

        return None


class ControlGestureDetector:
    """
    Nhận diện cử chỉ điều khiển đặc biệt bằng quy tắc hình học hoặc nhãn phân loại (Chương 4.2).
    - Open Palm (5 ngón xoè): KHOẢNG TRẮNG
    - Fist (Nắm tay giữ >= 20 frames): BACKSPACE (Xoá ký tự)
    - Thumbs Up: KẾT THÚC CÂU / GỬI
    - Two Hands: XOÁ TOÀN BỘ CÂU
    """

    def __init__(self, fist_backspace_frames: int = 20):
        self.fist_backspace_frames = fist_backspace_frames
        self.fist_counter = 0

    def detect_geometric(self, landmarks_dict: Dict[str, Any]) -> Optional[str]:
        """
        Phát hiện cử chỉ điều khiển từ toạ độ 21 điểm mốc bàn tay MediaPipe.
        """
        if not landmarks_dict or "landmarks" not in landmarks_dict:
            self.fist_counter = 0
            return None

        lm = landmarks_dict["landmarks"]
        if len(lm) < 21:
            return None

        # Đếm số ngón duỗi (tips: 8, 12, 16, 20 so với pips: 6, 10, 14, 18)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended_fingers = 0

        for tip, pip in zip(tips, pips):
            # y nhỏ hơn nghĩa là ngón tay hướng lên trên (duỗi)
            if lm[tip][1] < lm[pip][1]:
                extended_fingers += 1

        # Ngón cái (landmark 4 vs landmark 2/3)
        thumb_extended = abs(lm[4][0] - lm[17][0]) > abs(lm[2][0] - lm[17][0])
        if thumb_extended:
            extended_fingers += 1

        # 1. Bàn tay xoè 5 ngón -> SPACE
        if extended_fingers >= 5:
            self.fist_counter = 0
            return "GESTURE_SPACE"

        # 2. Nắm đấm (0 ngón duỗi) giữ lâu -> BACKSPACE
        if extended_fingers == 0:
            self.fist_counter += 1
            if self.fist_counter == self.fist_backspace_frames:
                self.fist_counter = 0
                return "GESTURE_BACKSPACE"
        else:
            self.fist_counter = 0

        # 3. Ngón cái chỉ lên (Thumbs Up: ngón cái duỗi, 4 ngón kia co) -> SUBMIT
        if thumb_extended and extended_fingers == 1 and lm[4][1] < lm[3][1]:
            return "GESTURE_SUBMIT"

        return None


class SentenceBuffer:
    """
    Bộ đệm từ và câu: Gom các sự kiện ký tự/khoảng trắng, hỗ trợ Optimistic UI và Debounce.
    """

    def __init__(self):
        self.words: List[str] = []
        self.current_word: str = ""

    def add_letter(self, ch: str):
        self.current_word += ch.upper()

    def add_space(self):
        if self.current_word:
            self.words.append(self.current_word)
            self.current_word = ""

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.words:
            self.current_word = self.words.pop()

    def clear(self):
        self.words.clear()
        self.current_word = ""

    def get_raw_tokens(self) -> List[str]:
        tokens = list(self.words)
        if self.current_word:
            tokens.append(self.current_word)
        return [w for w in tokens if w]

    def get_raw_preview(self) -> str:
        return " ".join(self.get_raw_tokens())
