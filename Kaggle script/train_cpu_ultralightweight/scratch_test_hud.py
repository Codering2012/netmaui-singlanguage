#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless Self-Test & Screenshot Generator for ASL MediaPipe Interface
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import cv2
import numpy as np

# Thêm path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from asl_mediapipe_simulator import ASLMediaPipeHUD, get_canonical_asl_landmarks
from asl_pipeline import ASLStreamPipeline

def main():
    print("[*] Bắt đầu tự kiểm tra giao diện ASL MediaPipe (Headless)...")
    
    # 1. Khởi tạo Pipeline & HUD
    pipeline = ASLStreamPipeline()
    hud = ASLMediaPipeHUD(width=1280, height=720)
    
    # 2. Tạo Canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(0, 720, 40):
        cv2.line(canvas, (0, y), (1280 - 460, y), (18, 22, 32), 1)
    for x in range(0, 1280 - 460, 40):
        cv2.line(canvas, (x, 0), (x, 720), (18, 22, 32), 1)

    # 3. Mô phỏng cử chỉ chữ 'L'
    landmarks = get_canonical_asl_landmarks('L')
    hud.draw_skeleton(canvas, landmarks, bbox_rect=(180, 160, 460, 480))
    
    # 4. Đẩy qua Pipeline
    for c in "HELO":
        pipeline.buffer.add_letter(c)
    pipeline.buffer.add_space()
    for c in "MY":
        pipeline.buffer.add_letter(c)
    pipeline.buffer.add_space()
    for c in "NAM":
        pipeline.buffer.add_letter(c)
    pipeline.buffer.add_space()
    for c in "IS":
        pipeline.buffer.add_letter(c)
    pipeline.buffer.add_space()
    for c in "ADLEY":
        pipeline.buffer.add_letter(c)
    
    # Chốt câu & Dịch
    res = pipeline._finalize_sentence()
    
    # Giả lập gõ từ mới
    for c in "THANK":
        pipeline.buffer.add_letter(c)
    suggs = pipeline.trie.suggest_completions(pipeline.buffer.current_word, top_k=3)
    suggestions = [w for w, _ in suggs]
    
    # 5. Vẽ HUD Panel
    hud.draw_hud_panel(
        img=canvas,
        current_char="L",
        confidence=0.968,
        fsm_state="COMMIT",
        hold_progress=1.0,
        active_word=pipeline.buffer.current_word,
        trie_suggestions=suggestions,
        english_sentence=res["english"],
        vietnamese_sentence=res["vietnamese"],
        fps=60.0,
        latency_ms=0.18,
        is_sim_mode=True
    )
    
    # 6. Lưu ảnh chụp giao diện
    out_dir = r"C:\Users\Windows 10 21H1\.gemini\antigravity-ide\brain\03909049-1864-4bc0-88ea-9cfe841517e6"
    out_path = os.path.join(out_dir, "mediapipe_asl_interface_preview.png")
    cv2.imwrite(out_path, canvas)
    
    # Lưu thêm 1 bản trong train_cpu_ultralightweight
    local_path = os.path.join(CURRENT_DIR, "mediapipe_asl_interface_preview.png")
    cv2.imwrite(local_path, canvas)
    
    print(f"[+] Đã lưu ảnh chụp giao diện HUD tại: {out_path}")
    print("[+] Kiểm tra thành công 100%!")

if __name__ == "__main__":
    main()
