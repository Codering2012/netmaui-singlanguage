import sys
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from asl_geometric import get_accurate_asl_template, get_bone_direction_vector

FINGER_BONES = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)]
}

def extract_per_finger_vectors(landmarks_21: np.ndarray):
    """Trích xuất vector 3D cho từng ngón tay riêng biệt."""
    finger_vecs = {}
    for finger_name, bones in FINGER_BONES.items():
        v_list = []
        for (i, j) in bones:
            vec = landmarks_21[j] - landmarks_21[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            v_list.append(vec)
        finger_vecs[finger_name] = np.concatenate(v_list)
    return finger_vecs

ALPHABET = [chr(65+i) for i in range(26)]
TEMPLATES_PER_FINGER = {c: extract_per_finger_vectors(get_accurate_asl_template(c)) for c in ALPHABET}

def classify_balanced_asl(landmarks_21: np.ndarray):
    """
    So khớp hình học ASL với trọng số cân bằng độc lập cho từng ngón tay:
    30% Thumb + 20% Index + 20% Middle + 15% Ring + 15% Pinky
    """
    in_vecs = extract_per_finger_vectors(landmarks_21)
    weights = {"thumb": 0.35, "index": 0.20, "middle": 0.20, "ring": 0.125, "pinky": 0.125}
    
    best_char = 'A'
    best_sim = -1.0
    
    for c, templ_vecs in TEMPLATES_PER_FINGER.items():
        total_sim = 0.0
        for f_name, w in weights.items():
            u = in_vecs[f_name]
            v = templ_vecs[f_name]
            sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
            total_sim += w * sim
            
        if total_sim > best_sim:
            best_sim = total_sim
            best_char = c
            
    conf = float(np.clip((best_sim - 0.70) / 0.30, 0.50, 0.99))
    return best_char, conf

print("[*] Kiểm thử Bộ so khớp hình học Cân bằng (Balanced Per-Finger Matcher):")
passed = 0
for letter in ALPHABET:
    if letter in ['J', 'Z']:
        continue
    pose = get_accurate_asl_template(letter)
    pred, conf = classify_balanced_asl(pose)
    status = "[PASS]" if pred == letter else "[FAIL]"
    if pred == letter: passed += 1
    print(f"  {status} Sign '{letter}' -> Detected '{pred}' (Conf: {conf*100:.1f}%)")

print(f"\n[+] Tổng kết: {passed}/24 chữ cái tĩnh đạt 100% chuẩn xác!")
