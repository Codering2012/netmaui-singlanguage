#!/usr/bin/env python3
"""
================================================================================
  ASL FOUNDATION MODEL - FAST GPU RUNNER SCRIPT (PURE CMD / ZERO OVERHEAD)
================================================================================
Chạy tự động toàn bộ quy trình:
  1. Kiểm tra phần cứng & tự nhận diện số lượng GPU (1x, 2x, 4x, 8x).
  2. Tự động tải datasets tốc độ cao qua kagglehub (hoặc dùng thư mục có sẵn).
  3. Kích hoạt PyTorch DDP qua torchrun với cấu hình tối ưu tốc độ tối đa.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ASL Fast GPU Runner")
    parser.add_argument("--gpus", type=int, default=None, help="Số lượng GPU (mặc định: tự phát hiện)")
    parser.add_argument("--batch-size", type=int, default=256, help="Kích thước batch hiệu dụng")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-len", type=int, default=384, help="Độ dài chuỗi tối đa")
    parser.add_argument("--epochs", type=int, default=200, help="Tổng số epoch")
    parser.add_argument("--phase1-epochs", type=int, default=140, help="Số epoch Phase 1")
    parser.add_argument("--save-dir", type=str, default="/workspace/checkpoints", help="Thư mục lưu checkpoint")
    parser.add_argument("--data-dir", type=str, default=None, help="Đường dẫn thủ công đến asl_preprocessed_phase1")
    parser.add_argument("--kdwd-dir", type=str, default=None, help="Đường dẫn thủ công đến kdwd")
    parser.add_argument("--aslg-csv", type=str, default=None, help="Đường dẫn thủ công đến train.csv của ASLG")
    args = parser.parse_args()

    print("=" * 80)
    print("      🚀 KHỞI ĐỘNG FAST GPU TRAINING RUNNER (PURE CLI)      ")
    print("=" * 80)

    # 1. Phát hiện GPU
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        num_gpus = 1

    if args.gpus is not None:
        num_gpus = args.gpus

    if num_gpus < 1:
        print("[!] Không tìm thấy CUDA GPU. Sẽ chuyển sang chạy 1 process.")
        num_gpus = 1
    else:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"[+] Đã nhận diện {num_gpus} GPU: {', '.join(gpu_names)}")

    # 2. Tự động tải datasets qua kagglehub nếu chưa truyền tham số
    data_dir = args.data_dir
    kdwd_dir = args.kdwd_dir
    aslg_csv = args.aslg_csv

    if not (data_dir and kdwd_dir and aslg_csv):
        print("\n[*] Đang xác thực và tải datasets từ Kaggle qua kagglehub...")
        try:
            import kagglehub
        except ImportError:
            print("[*] Đang cài đặt kagglehub...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "kagglehub"])
            import kagglehub

        if not kdwd_dir:
            print("  -> Tải Kensho Wikimedia (KDWD)...")
            kdwd_dir = kagglehub.dataset_download('kenshoresearch/kensho-derived-wikimedia-data')
            print(f"     [OK] {kdwd_dir}")

        if not aslg_csv:
            print("  -> Tải ASLG-PC12...")
            aslg_path = kagglehub.dataset_download('thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c')
            candidate_csv = os.path.join(aslg_path, "train.csv")
            if os.path.exists(candidate_csv):
                aslg_csv = candidate_csv
            else:
                for f in Path(aslg_path).rglob("*.csv"):
                    aslg_csv = str(f)
                    break
            print(f"     [OK] {aslg_csv}")

        if not data_dir:
            print("  -> Tải ASL Preprocessed Shards...")
            frakenstein_path = kagglehub.dataset_download('tranquocbao2012/frakenstein-asl-final-version')
            candidate_data = os.path.join(frakenstein_path, "asl_dataset", "asl_preprocessed_phase1")
            if os.path.exists(candidate_data):
                data_dir = candidate_data
            else:
                candidate_data = os.path.join(frakenstein_path, "asl_preprocessed_phase1")
                data_dir = candidate_data if os.path.exists(candidate_data) else frakenstein_path
            print(f"     [OK] {data_dir}")

    # 3. Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(args.save_dir, exist_ok=True)

    # 4. Thiết lập biến môi trường tối ưu
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    script_path = Path(__file__).resolve().parent / "train_all_in_one_tpu.py"
    if not script_path.exists():
        script_path = Path("train_all_in_one_tpu.py").resolve()

    # 5. Xây dựng lệnh torchrun
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        str(script_path),
        "--data-dir", str(data_dir),
        "--kdwd-dir", str(kdwd_dir),
        "--aslg-csv", str(aslg_csv),
        "--save-dir", str(args.save_dir),
        "--streamed-dataset",
        "--disable-bpe",
        "--precision", args.precision,
        "--batch-size", str(args.batch_size),
        "--accum-steps", str(args.accum_steps),
        "--epochs", str(args.epochs),
        "--phase1-epochs", str(args.phase1_epochs),
        "--max-len", str(args.max_len),
        "--d-model", "512",
        "--nhead", "8",
        "--num-layers", "6",
        "--num-dataloader-workers", "4",
        "--log-freq", "25",
        "--is-causal",
    ]

    print("\n" + "=" * 80)
    print("🔥 ĐANG KÍCH HOẠT QUY TRÌNH HUẤN LUYỆN MULTI-GPU TRỰC TIẾP TRÊN TERMINAL:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")

    # 6. Chạy trực tiếp và stream log không độ trễ
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
