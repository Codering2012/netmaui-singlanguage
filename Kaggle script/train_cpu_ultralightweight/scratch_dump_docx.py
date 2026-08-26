import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = Path(r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\Ghep_ky_tu_thanh_cau_ASL.docx")

with zipfile.ZipFile(p, 'r') as docx:
    xml_content = docx.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = []
    for p_node in tree.findall('.//w:p', ns):
        texts = [t.text for t in p_node.findall('.//w:t', ns) if t.text]
        if texts:
            paragraphs.append(''.join(texts))

with open(r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train_cpu_ultralightweight\docx_content_dump.txt", "w", encoding="utf-8") as f:
    for i, para in enumerate(paragraphs):
        f.write(f"[{i+1}] {para}\n")

print(f"Dumped {len(paragraphs)} paragraphs to docx_content_dump.txt")

# Print major headings and outline
for i, para in enumerate(paragraphs):
    if any(h in para.lower() for h in ["chương", "mục", "phần", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "bước", "kiến trúc", "tính năng", "khiếm khuyết", "hạn chế", "kết luận", "ghép", "fsm", "trie"]):
        if len(para) < 120:
            print(f"Line {i+1}: {para}")
