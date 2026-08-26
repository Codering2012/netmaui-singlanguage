import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

docx_path = r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\Ghep_ky_tu_thanh_cau_ASL.docx"
out_txt_path = r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train_tpu\scratch\docx_content.txt"

with zipfile.ZipFile(docx_path) as z:
    xml_content = z.read("word/document.xml")

tree = ET.fromstring(xml_content)
paragraphs = []
for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
    texts = [node.text for node in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if node.text]
    if texts:
        paragraphs.append("".join(texts))

full_text = "\n\n".join(paragraphs)

with open(out_txt_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted {len(paragraphs)} paragraphs to {out_txt_path}")
print(f"Total characters: {len(full_text)}")
