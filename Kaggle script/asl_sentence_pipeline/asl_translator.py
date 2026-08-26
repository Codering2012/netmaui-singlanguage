"""
TẦNG 5: DỊCH MÁY ANH -> VIỆT (ENGLISH TO VIETNAMESE TRANSLATION)
Dựa theo Tài liệu kỹ thuật: 'Từ ký tự ngón tay đến câu hoàn chỉnh' (Chương 6.2 & 6.3, Phụ lục C)
"""

import re
from typing import Optional, Set, Dict, List


class ASLTranslator:
    """
    Bộ dịch Anh -> Việt tích hợp:
    1. Tự động bảo tồn danh từ riêng / Whitelist tên người.
    2. Tuỳ biến cặp đại từ xưng hô tiếng Việt (tôi/mình/em/anh/chị).
    3. Hỗ trợ động cơ dịch siêu nhẹ tức thời (< 1ms trên CPU) và giao diện kết nối LLM / Transformer.
    """

    def __init__(self, speaker_pronoun: str = "tôi", listener_pronoun: str = "bạn"):
        self.speaker_pronoun = speaker_pronoun
        self.listener_pronoun = listener_pronoun

        # Kho mẫu dịch câu hội thoại ASL ngón tay thường gặp
        self.phrase_dict: Dict[str, str] = {
            "hello": "xin chào",
            "hi": "chào bạn",
            "thank you very much": "cảm ơn bạn rất nhiều",
            "thank you": "cảm ơn bạn",
            "thanks": "cảm ơn",
            "where is the hospital": "bệnh viện ở đâu",
            "where is the bathroom": "nhà vệ sinh ở đâu",
            "where is the toilet": "nhà vệ sinh ở đâu",
            "i need help now": "{sp} cần giúp đỡ ngay bây giờ",
            "i need help": "{sp} cần giúp đỡ",
            "can you help me": "{lp} có thể giúp {sp} được không",
            "can you call a doctor": "{lp} có thể gọi bác sĩ giúp {sp} được không",
            "call an ambulance": "hãy gọi xe cấp cứu",
            "nice to meet you": "rất vui được gặp {lp}",
            "good morning": "chào buổi sáng",
            "good afternoon": "chào buổi chiều",
            "good evening": "chào buổi tối",
            "good night": "chúc ngủ ngon",
            "goodbye": "tạm biệt",
            "see you later": "hẹn gặp lại {lp}",
            "how are you": "{lp} có khỏe không",
            "i am fine": "{sp} khỏe",
            "i love you": "{sp} yêu {lp}",
            "what is your name": "tên của {lp} là gì",
            "what is this": "đây là cái gì",
            "who are you": "{lp} là ai",
            "where are you": "{lp} đang ở đâu",
            "why are you here": "tại sao {lp} lại ở đây",
            "please wait": "xin vui lòng đợi một chút",
            "i don't understand": "{sp} không hiểu",
            "i do not understand": "{sp} không hiểu",
            "yes": "vâng",
            "no": "không",
            "sorry": "xin lỗi",
            "excuse me": "xin thứ lỗi",
        }

        # Từ điển từ vựng độc lập
        self.word_dict: Dict[str, str] = {
            "hello": "xin chào",
            "hi": "chào",
            "name": "tên",
            "is": "là",
            "am": "là",
            "are": "là",
            "hospital": "bệnh viện",
            "doctor": "bác sĩ",
            "nurse": "y tá",
            "police": "cảnh sát",
            "help": "giúp đỡ",
            "need": "cần",
            "now": "bây giờ",
            "today": "hôm nay",
            "tomorrow": "ngày mai",
            "yesterday": "hôm qua",
            "water": "nước",
            "food": "thức ăn",
            "good": "tốt",
            "bad": "xấu",
            "happy": "vui vẻ",
            "sad": "buồn",
            "call": "gọi",
            "can": "có thể",
            "you": "{lp}",
            "i": "{sp}",
            "my": "của {sp}",
            "your": "của {lp}",
            "where": "ở đâu",
            "what": "cái gì",
            "when": "khi nào",
            "why": "tại sao",
            "how": "như thế nào",
            "who": "ai",
            "thank": "cảm ơn",
            "very": "rất",
            "much": "nhiều",
            "please": "làm ơn",
            "school": "trường học",
            "home": "nhà",
            "family": "gia đình",
            "father": "bố",
            "mother": "mẹ",
            "friend": "bạn bè",
        }

    def set_pronouns(self, speaker: str, listener: str):
        """Thay đổi cặp xưng hô tiếng Việt (Chương 6.3)."""
        self.speaker_pronoun = speaker
        self.listener_pronoun = listener

    def translate_to_vietnamese(
        self,
        english_sentence: str,
        custom_lexicon: Optional[Set[str]] = None
    ) -> str:
        """
        Dịch câu tiếng Anh sang tiếng Việt với sự bảo toàn danh từ riêng.
        """
        if not english_sentence:
            return ""

        clean_en = english_sentence.strip()
        has_question = clean_en.endswith("?")
        raw_text = clean_en.rstrip(".?!").strip()
        raw_lower = raw_text.lower()

        # 1. Kiểm tra mẫu câu đặc biệt (Pattern Matching)
        # Khớp: "Hello my name is [NAME]"
        match_name = re.match(r"^(?:hello|hi)?\s*,?\s*my name is\s+([A-Za-z]+)$", raw_lower)
        if match_name:
            name = match_name.group(1).capitalize()
            sp = self.speaker_pronoun
            vi_text = f"Xin chào, tên {sp} là {name}."
            return vi_text

        # Khớp: "My name is [NAME]"
        match_name_short = re.match(r"^my name is\s+([A-Za-z]+)$", raw_lower)
        if match_name_short:
            name = match_name_short.group(1).capitalize()
            sp = self.speaker_pronoun
            vi_text = f"Tên của {sp} là {name}."
            return vi_text

        # 2. Khớp trong kho mẫu câu chính xác
        if raw_lower in self.phrase_dict:
            res = self.phrase_dict[raw_lower]
            res = res.replace("{sp}", self.speaker_pronoun).replace("{lp}", self.listener_pronoun)
            res = res[0].upper() + res[1:]
            res += "?" if has_question else "."
            return res

        # 3. Dịch kết hợp từ và cụm từ ngữ pháp
        words = raw_text.split()
        translated_tokens: List[str] = []
        skip_indices: Set[int] = set()

        # Bảo tồn tên riêng
        active_lexicon = {w.lower() for w in custom_lexicon} if custom_lexicon else set()

        for idx, w in enumerate(words):
            if idx in skip_indices:
                continue

            w_clean = re.sub(r"[^\w]", "", w)
            w_lower = w_clean.lower()

            if w_lower in active_lexicon:
                translated_tokens.append(w_clean.capitalize())
            elif w_lower in self.word_dict:
                tr = self.word_dict[w_lower]
                tr = tr.replace("{sp}", self.speaker_pronoun).replace("{lp}", self.listener_pronoun)
                translated_tokens.append(tr)
            else:
                # Nếu từ chưa biết, giữ nguyên
                translated_tokens.append(w_clean)

        if not translated_tokens:
            return ""

        res_vi = " ".join(translated_tokens)
        res_vi = res_vi[0].upper() + res_vi[1:]
        res_vi += "?" if has_question else "."

        return res_vi
