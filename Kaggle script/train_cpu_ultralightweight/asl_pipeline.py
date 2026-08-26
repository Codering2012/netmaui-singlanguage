"""
PIPELINE TOÀN DIỆN: TỪ DÒNG KHUNG HÌNH CAMERA ĐẾN CÂU HOÀN CHỈNH (EN & VI)
Hợp nhất đầy đủ 5 Tầng xử lý theo đúng Tài liệu kỹ thuật: 'Từ ký tự ngón tay đến câu hoàn chỉnh'
"""

from typing import Optional, Dict, Any, List, Set, Tuple
from asl_fsm import LetterStabilizerFSM, ControlGestureDetector, SentenceBuffer
from asl_trie import PrefixTrie
from asl_corrector import ASLSpellCorrector
from asl_translator import ASLTranslator


class ASLStreamPipeline:
    """
    Hệ thống hợp nhất toàn diện 5 Tầng xử lý chữ cái ngón tay ASL thành câu hoàn chỉnh:
    
    Tầng 1: Ổn định hoá ký tự (FSM: IDLE -> HOLD -> COMMIT -> RELEASE).
    Tầng 2: Phân đoạn từ & câu (Nghỉ tay + Cử chỉ điều khiển + Prefix Trie Suggestions).
    Tầng 3: Sửa lỗi chính tả & Mô hình ngôn ngữ (SymSpell + 3 Lớp bảo vệ + ASL Confusion Matrix).
    Tầng 4: Dựng câu & Dấu câu thông minh (Viết hoa, dấu ?, dấu .).
    Tầng 5: Dịch sang tiếng Việt (Bảo tồn tên riêng, tuỳ biến xưng hô).
    """

    def __init__(
        self,
        window: int = 8,
        min_votes: int = 6,
        min_conf: float = 0.60,
        release_frames: int = 4,
        space_frames: int = 12,
        hold_repeat_frames: int = 25,
        sentence_frames: int = 45,
        speaker_pronoun: str = "tôi",
        listener_pronoun: str = "bạn",
    ):
        # Tầng 1 & 2
        self.fsm = LetterStabilizerFSM(
            window=window,
            min_votes=min_votes,
            min_conf=min_conf,
            release_frames=release_frames,
            space_frames=space_frames,
            hold_repeat_frames=hold_repeat_frames,
            sentence_frames=sentence_frames,
        )
        self.gesture_detector = ControlGestureDetector()
        self.buffer = SentenceBuffer()

        # Tầng 2 & 3 Trie Suggestions
        self.trie = PrefixTrie()
        self.trie.load_from_symspell(max_words=60000)

        # Tầng 3 & 4 Corrector & Punctuator
        self.corrector = ASLSpellCorrector()

        # Tầng 5 Translator
        self.translator = ASLTranslator(
            speaker_pronoun=speaker_pronoun,
            listener_pronoun=listener_pronoun
        )

        # Danh sách tên riêng theo phiên (Session lexicon)
        self.session_lexicon: Set[str] = set()

    def register_proper_nouns(self, nouns: List[str]):
        """Đăng ký danh sách tên riêng / thuật ngữ cho phiên làm việc."""
        for n in nouns:
            clean_n = str(n).strip().lower()
            if clean_n:
                self.session_lexicon.add(clean_n)
                self.corrector.add_to_session_lexicon(clean_n)

    def reset(self):
        """Khởi động lại toàn bộ chu trình."""
        self.fsm.reset()
        self.buffer.clear()

    def process_frame(
        self,
        predicted_letter: Optional[str],
        confidence: float = 1.0,
        landmarks_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý từng khung hình trực tiếp từ Camera (20-30 FPS).
        
        Returns:
            Dict chứa sự kiện, bản xem trước (Optimistic UI), gợi ý Top-3 từ, và kết quả câu hoàn chỉnh nếu kết thúc.
        """
        event_info = {
            "event_type": "NONE",
            "letter": None,
            "raw_preview": self.buffer.get_raw_preview(),
            "suggestions": [],
            "is_sentence_final": False,
            "english": None,
            "vietnamese": None,
            "confidence": 0.0
        }

        # 1. Kiểm tra cử chỉ điều khiển đặc biệt (Open Palm / Fist / Thumbs Up)
        if landmarks_dict:
            gesture = self.gesture_detector.detect_geometric(landmarks_dict)
            if gesture == "GESTURE_SPACE":
                self.buffer.add_space()
                event_info["event_type"] = "SPACE"
                event_info["raw_preview"] = self.buffer.get_raw_preview()
                return event_info
            elif gesture == "GESTURE_BACKSPACE":
                self.buffer.backspace()
                event_info["event_type"] = "BACKSPACE"
                event_info["raw_preview"] = self.buffer.get_raw_preview()
                return event_info
            elif gesture == "GESTURE_SUBMIT":
                return self._finalize_sentence()

        # 2. Đưa qua Máy trạng thái FSM (Tầng 1)
        fsm_out = self.fsm.push(predicted_letter, confidence)
        if not fsm_out:
            return event_info

        ev_type, ev_val = fsm_out

        if ev_type == "LETTER" and ev_val:
            self.buffer.add_letter(ev_val)
            event_info["event_type"] = "LETTER"
            event_info["letter"] = ev_val
            event_info["raw_preview"] = self.buffer.get_raw_preview()
            # Gợi ý Top-3 từ theo tiền tố
            suggs = self.trie.suggest_completions(self.buffer.current_word, top_k=3)
            event_info["suggestions"] = [w for w, _ in suggs]

        elif ev_type == "SPACE":
            self.buffer.add_space()
            event_info["event_type"] = "SPACE"
            event_info["raw_preview"] = self.buffer.get_raw_preview()

        elif ev_type == "END":
            return self._finalize_sentence()

        return event_info

    def _finalize_sentence(self) -> Dict[str, Any]:
        """Hoàn tất câu hiện tại qua Tầng 3, 4, 5."""
        raw_tokens = self.buffer.get_raw_tokens()
        if not raw_tokens:
            self.reset()
            return {
                "event_type": "END_SENTENCE",
                "raw_preview": "",
                "suggestions": [],
                "is_sentence_final": True,
                "english": "",
                "vietnamese": "",
                "confidence": 1.0
            }

        # Tầng 3: Sửa lỗi chính tả
        corrected_tokens = self.corrector.correct_phrase(raw_tokens, self.session_lexicon)

        # Tầng 4: Dựng câu & Dấu câu
        english_sentence = self.corrector.punctuate_and_capitalize(corrected_tokens, self.session_lexicon)

        # Tầng 5: Dịch Anh -> Việt
        vietnamese_sentence = self.translator.translate_to_vietnamese(english_sentence, self.session_lexicon)

        raw_str = " ".join(raw_tokens)
        self.reset()

        return {
            "event_type": "END_SENTENCE",
            "raw_preview": raw_str,
            "suggestions": [],
            "is_sentence_final": True,
            "english": english_sentence,
            "vietnamese": vietnamese_sentence,
            "confidence": 0.98
        }

    def process_raw_word_list(self, words: List[str], custom_lexicon: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Xử lý danh sách từ thô trực tiếp (API Contract - Chương 8.2).
        Input: ["HELO", "MY", "NAM", "IS", "ADLEY"]
        """
        active_lex = set(self.session_lexicon)
        if custom_lexicon:
            active_lex.update({str(w).lower() for w in custom_lexicon})

        # Sửa lỗi
        corrected = self.corrector.correct_phrase(words, active_lex)

        # Dựng câu
        en = self.corrector.punctuate_and_capitalize(corrected, active_lex)

        # Dịch
        vi = self.translator.translate_to_vietnamese(en, active_lex)

        raw_str = " ".join(words).lower()

        return {
            "raw": raw_str,
            "english": en,
            "translated": vi,
            "confidence": 0.98,
            "engine": "asl_fsm_symspell_v1"
        }

    def process_unspaced_blob(self, blob: str) -> Dict[str, Any]:
        """
        Xử lý chuỗi dính liền không dấu cách (Chương 4.3).
        Input: "whereisthehospital" -> "Where is the hospital?" -> "Bệnh viện ở đâu?"
        """
        segmented = self.corrector.segment_unspaced_stream(blob)
        words = segmented.split()
        return self.process_raw_word_list(words)
