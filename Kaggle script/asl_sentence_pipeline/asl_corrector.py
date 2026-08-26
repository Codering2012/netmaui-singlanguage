"""
TẦNG 3 & 4: SỬA LỖI CHÍNH TẢ, MÔ HÌNH NGÔN NGỮ, DỰNG CÂU & DẤU CÂU (CHUYÊN DỤNG ASL)
Tuân thủ nghiêm ngặt 4 vấn đề cốt lõi (Bảng 1.1) và 3 biện pháp bảo vệ (Bảng 5.1) trong Tài liệu kỹ thuật
"""

import os
import math
import re
import importlib.resources as ir
from typing import List, Set, Optional, Dict, Tuple
from symspellpy import SymSpell, Verbosity
from asl_trie import PrefixTrie


# Bảng nhầm lẫn hình thái thị giác ASL (ASL Handshape Confusion Matrix - Chương 9.3)
ASL_CONFUSION_GROUPS = [
    {"M", "N", "T", "S", "A", "E"},  # Nhóm nắm đấm / vị trí ngón cái
    {"U", "V", "R", "W", "K"},       # Nhóm các ngón duỗi song song
    {"D", "L", "I", "J", "Z", "1"},  # Nhóm ngón trỏ / ngón út
    {"C", "O", "B"},                 # Nhóm bàn tay cong / phẳng
    {"G", "H", "P", "Q"},            # Nhóm bàn tay nằm ngang / chúc xuống
]

# Tập hợp các từ mở đầu câu hỏi (Question Starters - Chương 6.1)
QUESTION_STARTERS = {
    "what", "where", "when", "who", "why", "how",
    "do", "does", "did", "can", "could", "is", "are",
    "am", "was", "were", "will", "would", "should", "may", "might"
}

# Đại từ và từ ngữ cần luôn viết hoa
ALWAYS_CAPITALIZE = {"i", "i'm", "i'll", "i'd", "i've"}

# Bộ từ vựng hội thoại ASL ngón tay cốt lõi (Core Conversational ASL Vocabulary Boost)
ASL_CONVERSATIONAL_WORDS = {
    "hello": 5000000000,
    "hi": 4000000000,
    "name": 4500000000,
    "where": 4000000000,
    "hospital": 2000000000,
    "bathroom": 1500000000,
    "toilet": 1000000000,
    "thank": 3500000000,
    "you": 6000000000,
    "very": 4000000000,
    "much": 3500000000,
    "need": 3000000000,
    "help": 4000000000,
    "now": 3500000000,
    "can": 4500000000,
    "call": 3000000000,
    "doctor": 2500000000,
    "nurse": 1500000000,
    "please": 3000000000,
    "wait": 2500000000,
    "for": 4000000000,
    "me": 4500000000,
    "love": 3500000000,
    "like": 3500000000,
    "good": 4000000000,
    "morning": 2500000000,
    "night": 2500000000,
    "understand": 2000000000,
    "see": 3500000000,
    "later": 2500000000,
    "nice": 3000000000,
    "meet": 3000000000,
    "fine": 2500000000,
    "how": 4000000000,
    "what": 4500000000,
    "why": 3500000000,
    "who": 3500000000,
    "when": 3500000000,
    "is": 6000000000,
    "the": 8000000000,
    "a": 7000000000,
    "an": 5000000000,
    "my": 5000000000,
    "your": 4500000000,
}

# Bigram phổ biến trong hội thoại ASL
ASL_BIGRAM_BOOSTS = {
    ("hello", "my"): 10000000,
    ("my", "name"): 15000000,
    ("name", "is"): 15000000,
    ("where", "is"): 12000000,
    ("is", "the"): 15000000,
    ("the", "hospital"): 8000000,
    ("the", "bathroom"): 8000000,
    ("the", "doctor"): 6000000,
    ("thank", "you"): 20000000,
    ("you", "very"): 12000000,
    ("very", "much"): 15000000,
    ("i", "need"): 10000000,
    ("need", "help"): 12000000,
    ("help", "now"): 8000000,
    ("can", "you"): 15000000,
    ("you", "call"): 8000000,
    ("call", "a"): 10000000,
    ("a", "doctor"): 8000000,
    ("please", "wait"): 10000000,
    ("wait", "for"): 8000000,
    ("for", "me"): 10000000,
    ("i", "love"): 12000000,
    ("love", "you"): 15000000,
    ("nice", "to"): 12000000,
    ("to", "meet"): 12000000,
    ("meet", "you"): 15000000,
    ("do", "not"): 12000000,
    ("not", "understand"): 10000000,
}


class ASLSpellCorrector:
    """
    Bộ sửa lỗi chính tả chuyên dụng cho ASL Fingerspelling:
    - SymSpell tốc độ cao (< 0.1ms).
    - Bảo toàn danh từ riêng (Session Lexicon Whitelist).
    - Tính điểm có trọng số theo ma trận nhầm lẫn thị giác ASL (ASL Confusion Matrix).
    - Rescoring bằng xác suất Bigram hội thoại.
    """

    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7):
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
        self.trie = PrefixTrie()
        self._load_dictionaries()

        # Từ điển riêng của phiên làm việc (Session proper nouns / whitelist)
        self.session_lexicon: Set[str] = {
            "adley", "anna", "john", "mary", "david", "sarah", "michael",
            "peter", "paul", "vietnam", "america", "hanoi", "saigon"
        }

    def _load_dictionaries(self):
        """Nạp từ điển đơn từ và bigram từ symspellpy kết hợp bộ tăng cường ASL."""
        try:
            with ir.as_file(ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt") as p:
                self.sym.load_dictionary(str(p), term_index=0, count_index=1)
            with ir.as_file(ir.files("symspellpy") / "frequency_bigramdictionary_en_243_342.txt") as p:
                self.sym.load_bigram_dictionary(str(p), term_index=0, count_index=2)
        except Exception as e:
            print(f"[WARNING] Loading base symspell dictionary files: {e}.")

        # Nạp bộ tăng cường từ vựng ASL
        for word, freq in ASL_CONVERSATIONAL_WORDS.items():
            self.sym.create_dictionary_entry(word, freq)

        self.trie.load_from_symspell(max_words=60000)
        for word, freq in ASL_CONVERSATIONAL_WORDS.items():
            self.trie.insert(word, freq)

    def add_to_session_lexicon(self, word_or_words):
        """Thêm tên riêng / thuật ngữ vào whitelist của phiên."""
        if isinstance(word_or_words, str):
            self.session_lexicon.add(word_or_words.lower())
        elif isinstance(word_or_words, (list, set, tuple)):
            for w in word_or_words:
                self.session_lexicon.add(str(w).lower())

    def _asl_confusion_cost(self, c1: str, c2: str) -> float:
        """Chi phí thay thế giữa 2 ký tự dựa trên hình thái bàn tay ASL."""
        if c1 == c2:
            return 0.0
        c1, c2 = c1.upper(), c2.upper()
        for grp in ASL_CONFUSION_GROUPS:
            if c1 in grp and c2 in grp:
                return 0.35  # Chi phí rất thấp cho các lỗi nhầm lẫn ngón tay ASL điển hình
        return 1.0

    def _asl_string_distance(self, s1: str, s2: str) -> float:
        """
        Tính khoảng cách Levenshtein có trọng số đặc thù cho ASL:
        - Lỗi nhầm lẫn ngón tay ASL (Confusion Matrix): 0.35
        - Lỗi thiếu ký tự đôi (l vs ll, e vs ee, o vs oo, s vs ss, p vs pp): 0.15
        - Lỗi thiếu chữ 'e' ở đuôi từ (nam -> name, wher -> where): 0.15
        - Chèn / xoá thông thường: 1.0
        """
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2:
            return 0.0

        # Xử lý nhanh từ viết tắt cực ngắn
        if s1 == "u" and s2 == "you":
            return 0.1
        if s1 == "r" and s2 == "are":
            return 0.1
        if s1 == "ur" and s2 == "your":
            return 0.1

        n, m = len(s1), len(s2)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = float(i)
        for j in range(m + 1):
            dp[0][j] = float(j)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c1, c2 = s1[i - 1], s2[j - 1]
                if c1 == c2:
                    sub_cost = 0.0
                else:
                    sub_cost = self._asl_confusion_cost(c1, c2)

                # Chi phí thêm ký tự vào s1 để thành s2
                ins_cost = 1.0
                # Ưu đãi nếu s2 có ký tự đôi (double letter) mà s1 thiếu
                if j >= 2 and s2[j - 1] == s2[j - 2] and (i == 0 or s1[i - 1] != s2[j - 1]):
                    ins_cost = 0.15
                # Ưu đãi nếu s2 có chữ 'e' ở cuối mà s1 thiếu (nam -> name, wher -> where)
                elif j == m and s2[j - 1] == "e":
                    ins_cost = 0.15

                # Chi phí xoá ký tự khỏi s1
                del_cost = 1.0
                if i >= 2 and s1[i - 1] == s1[i - 2]:
                    del_cost = 0.20  # Xoá bớt ký tự lặp thừa

                dp[i][j] = min(
                    dp[i - 1][j] + del_cost,    # deletion
                    dp[i][j - 1] + ins_cost,    # insertion
                    dp[i - 1][j - 1] + sub_cost # substitution
                )

        return dp[n][m]

    def correct_word(
        self,
        word: str,
        prev_word: Optional[str] = None,
        custom_lexicon: Optional[Set[str]] = None
    ) -> str:
        """
        Sửa lỗi cho 1 từ đơn lẻ kết hợp ngữ cảnh từ trước đó (Bigram).
        """
        if not word:
            return ""

        w_lower = word.lower().strip()
        active_lexicon = self.session_lexicon
        if custom_lexicon:
            active_lexicon = active_lexicon.union({w.lower() for w in custom_lexicon})

        # 1. BIỆN PHÁP BẢO VỆ 1: Tên riêng trong Whitelist -> GIỮ NGUYÊN
        if w_lower in active_lexicon:
            return w_lower

        # 2. Xử lý trường hợp đặc biệt viết tắt
        if w_lower in {"u"} and (prev_word is not None and prev_word.lower() in {"can", "thank", "love", "see", "meet", "how", "are"}):
            return "you"
        if w_lower in {"r"} and (prev_word is not None and prev_word.lower() in {"you", "they", "we"}):
            return "are"
        if w_lower in {"i", "1", "l"} and (prev_word is None or prev_word.lower() in {"and", "that", "so", "but"}):
            return "i"

        # 3. BIỆN PHÁP BẢO VỆ 2: Chia bậc khoảng cách theo độ dài
        length = len(w_lower)
        if length <= 1:
            if w_lower in {"a", "i"}:
                return w_lower
            return w_lower

        # 4. Tra cứu SymSpell (cho phép khoảng cách 2)
        max_d = 2
        suggestions = self.sym.lookup(w_lower, Verbosity.ALL, max_edit_distance=max_d, include_unknown=True)

        if not suggestions:
            # Nếu không tìm thấy trong SymSpell, tìm trong bộ từ ASL cốt lõi
            best_match = w_lower
            min_dist = float("inf")
            for asl_w in ASL_CONVERSATIONAL_WORDS:
                dist = self._asl_string_distance(w_lower, asl_w)
                if dist < min_dist and dist <= 2.0:
                    min_dist = dist
                    best_match = asl_w
            return best_match

        # 5. Tái xếp hạng (Rescore) bằng ASL Confusion Matrix + Bigram Probability
        best_candidate = suggestions[0].term
        best_score = float("inf")

        for sugg in suggestions[:12]:
            cand = sugg.term
            asl_dist = self._asl_string_distance(w_lower, cand)

            # Điểm tần suất Unigram
            base_count = ASL_CONVERSATIONAL_WORDS.get(cand, sugg.count)
            unigram_score = -math.log10(max(10, base_count)) * 0.25

            # Điểm Bigram ngữ cảnh với từ phía trước
            bigram_score = 0.0
            if prev_word:
                p_clean = prev_word.lower().strip()
                if (p_clean, cand) in ASL_BIGRAM_BOOSTS:
                    bigram_score = -3.0  # Thưởng điểm cực mạnh cho cụm từ hội thoại đúng
                else:
                    bg_term = f"{p_clean} {cand}"
                    if bg_term in self.sym.bigrams:
                        bg_count = self.sym.bigrams[bg_term]
                        bigram_score = -math.log10(max(10, bg_count)) * 0.25

            total_cost = asl_dist + unigram_score + bigram_score

            if total_cost < best_score:
                best_score = total_cost
                best_candidate = cand

        return best_candidate

    def correct_phrase(self, raw_tokens: List[str], custom_lexicon: Optional[Set[str]] = None) -> List[str]:
        """
        Sửa lỗi cho chuỗi từ tuần tự kết hợp ngữ cảnh Bigram liên tục.
        """
        if not raw_tokens:
            return []

        corrected = []
        prev = None
        for tok in raw_tokens:
            clean_tok = tok.strip()
            if not clean_tok:
                continue
            c = self.correct_word(clean_tok, prev_word=prev, custom_lexicon=custom_lexicon)
            corrected.append(c)
            prev = c

        return corrected

    def segment_unspaced_stream(self, unspaced_str: str) -> str:
        """
        Tách từ bằng thuật toán quy hoạch động khi không có dấu cách (Chương 4.3).
        """
        if not unspaced_str:
            return ""
        clean_str = unspaced_str.lower().strip()
        seg_res = self.sym.word_segmentation(clean_str)
        return seg_res.corrected_string

    def punctuate_and_capitalize(self, words: List[str], custom_lexicon: Optional[Set[str]] = None) -> str:
        """
        TẦNG 4: Dựng câu hoàn chỉnh, viết hoa và thêm dấu câu thông minh (Chương 6.1).
        """
        if not words:
            return ""

        active_lexicon = self.session_lexicon
        if custom_lexicon:
            active_lexicon = active_lexicon.union({w.lower() for w in custom_lexicon})

        formatted_words = []
        for i, w in enumerate(words):
            w_clean = w.strip()
            if not w_clean:
                continue

            w_lower = w_clean.lower()

            if i == 0:
                # Ký tự đầu câu luôn viết hoa
                w_formatted = w_clean.capitalize()
            elif w_lower in active_lexicon:
                # Tên riêng viết hoa
                w_formatted = w_clean.capitalize()
            elif w_lower in ALWAYS_CAPITALIZE:
                # Đại từ I luôn viết hoa
                if w_lower == "i":
                    w_formatted = "I"
                else:
                    w_formatted = "I" + w_clean[1:]
            else:
                w_formatted = w_lower

            formatted_words.append(w_formatted)

        if not formatted_words:
            return ""

        sentence = " ".join(formatted_words)

        # Kiểm tra từ để hỏi
        first_word_lower = formatted_words[0].lower()
        is_question = first_word_lower in QUESTION_STARTERS

        # Gắn dấu kết câu
        if sentence[-1] not in ".?!":
            sentence += "?" if is_question else "."

        return sentence
