import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

docx_paths = [
    Path(r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\Ghep_ky_tu_thanh_cau_ASL.docx"),
    Path(r"C:\Users\Windows 10 21H1\Downloads\Ghep_ky_tu_thanh_cau_ASL.docx")
]

for p in docx_paths:
    if p.exists():
        print(f"=== READING: {p} ===")
        with zipfile.ZipFile(p, 'r') as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            # Namespace for Word processing ML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p_node in tree.findall('.//w:p', ns):
                texts = [t.text for t in p_node.findall('.//w:t', ns) if t.text]
                if texts:
                    paragraphs.append(''.join(texts))
            
            print(f"Total paragraphs: {len(paragraphs)}")
            for i, para in enumerate(paragraphs):
                print(f"[{i+1}] {para}")
        break
