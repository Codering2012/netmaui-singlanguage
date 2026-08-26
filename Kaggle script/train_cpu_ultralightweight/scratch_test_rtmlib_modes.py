import sys
import time
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rtmlib import Hand

dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(dummy_img, (320, 240), 60, (200, 200, 200), -1)

for mode in ['lightweight', 'balanced', 'performance']:
    try:
        print(f"\n[*] Testing Mode: '{mode}'...")
        hand_det = Hand(mode=mode, backend='onnxruntime', device='cpu')
        # Warmup
        for _ in range(3):
            _ = hand_det(dummy_img)
            
        t0 = time.time()
        for _ in range(15):
            kpts, scs = hand_det(dummy_img)
        dt = (time.time() - t0) / 15.0 * 1000
        fps = 1000.0 / dt
        print(f"  -> Mode '{mode}': Latency = {dt:.2f} ms | FPS = {fps:.1f}")
    except Exception as e:
        print(f"  -> Mode '{mode}' error: {e}")
