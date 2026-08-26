import sys
import os
import time
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("[*] Khởi tạo RTMPose-Hand bằng rtmlib...")
try:
    from rtmlib import Hand, draw_skeleton
    print("[+] Import rtmlib thành công!")
    
    # Khởi tạo detector Hand (hỗ trợ CPU/CUDA ONNXRuntime)
    hand_detector = Hand(backend='onnxruntime', device='cpu')
    print("[+] Khởi tạo RTMPose Hand detector thành công!")
    
    # Tạo ảnh dummy 640x480 để đo latency
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Vẽ hình bàn tay giả lập
    cv2.circle(dummy_img, (320, 240), 60, (200, 200, 200), -1)
    
    t0 = time.time()
    keypoints, scores = hand_detector(dummy_img)
    dt = (time.time() - t0) * 1000
    
    print(f"[+] Output Keypoints Shape: {keypoints.shape}, Scores Shape: {scores.shape}")
    print(f"[+] Initial Warmup Inference Latency: {dt:.2f} ms")
    
    # Đo FPS trên 20 iterations
    latencies = []
    for _ in range(20):
        t1 = time.time()
        kpts, scs = hand_detector(dummy_img)
        latencies.append((time.time() - t1) * 1000)
        
    avg_lat = np.mean(latencies)
    fps = 1000.0 / avg_lat
    print(f"[+] Average Latency: {avg_lat:.2f} ms | Throughput: {fps:.1f} FPS (CPU)")

except Exception as e:
    print(f"[!] Lỗi khi chạy RTMPose: {e}")
    import traceback
    traceback.print_exc()
