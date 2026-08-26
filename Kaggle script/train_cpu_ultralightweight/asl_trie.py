"""
TẦNG 2 & 3: CÂY TIỀN TỐ (PREFIX TRIE) & GỢI Ý TỪ THÔNG MINH KHI ĐANG ĐÁNH VẦN
Dựa theo Tài liệu kỹ thuật: 'Từ ký tự ngón tay đến câu hoàn chỉnh' (Chương 5.2 & 5.3)
"""

from typing import List, Tuple, Optional, Dict
import os
import importlib.resources as ir


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.frequency: int = 0
        self.word: Optional[str] = None


class PrefixTrie:
    """
    Cây tiền tố (Trie) nạp từ điển tần suất:
    1. Gợi ý từ thông minh khi đang đánh vần (Top-3 suggestions).
    2. Cắt tỉa không gian tìm kiếm trong Beam Search (Chương 5.3).
    3. Phát hiện lỗi sớm: Nếu tiền tố không có trong Trie, ký tự vừa nhận có xác suất sai cao.
    """

    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word: str, frequency: int = 1):
        if not word:
            return
        node = self.root
        w_lower = word.lower()
        for ch in w_lower:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True
        node.frequency = max(node.frequency, frequency)
        node.word = w_lower
        self.word_count += 1

    def has_prefix(self, prefix: str) -> bool:
        """Kiểm tra xem prefix có thể mở rộng thành một từ hợp lệ nào không."""
        if not prefix:
            return True
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def is_valid_word(self, word: str) -> bool:
        """Kiểm tra xem word có phải là một từ hoàn chỉnh hợp lệ không."""
        if not word:
            return False
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_word

    def suggest_completions(self, prefix: str, top_k: int = 3) -> List[Tuple[str, int]]:
        """
        Trả về top_k từ phổ biến nhất bắt đầu bằng prefix (ví dụ 'hel' -> [('hello', 10000), ('help', 8000), ('held', 2000)]).
        """
        if not prefix:
            return []

        node = self.root
        p_lower = prefix.lower()
        for ch in p_lower:
            if ch not in node.children:
                return []
            node = node.children[ch]

        results: List[Tuple[str, int]] = []

        def _dfs(curr: TrieNode):
            if curr.is_word and curr.word:
                results.append((curr.word, curr.frequency))
            for child in curr.children.values():
                _dfs(child)

        _dfs(node)
        # Sắp xếp theo tần suất giảm dần
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def load_from_symspell(self, max_words: int = 50000):
        """Nạp từ điển tần suất từ symspellpy frequency dictionary."""
        try:
            with ir.as_file(ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt") as p:
                with open(p, "r", encoding="utf-8") as f:
                    count = 0
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            term = parts[0]
                            try:
                                freq = int(parts[1])
                            except ValueError:
                                freq = 1
                            self.insert(term, freq)
                            count += 1
                            if count >= max_words:
                                break
        except Exception as e:
            # Fallback nếu không có file
            default_words = [
                ("hello", 100000), ("help", 80000), ("held", 20000), ("name", 90000),
                ("is", 500000), ("my", 400000), ("where", 200000), ("hospital", 30000),
                ("thank", 150000), ("you", 600000), ("very", 250000), ("much", 200000),
                ("can", 300000), ("doctor", 40000), ("need", 120000), ("now", 220000)
            ]
            for w, f in default_words:
                self.insert(w, f)
