with open("train_tpu/train_all_in_one_tpu.py", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if "xm.save" in line or "torch.save" in line:
            print(f"Line {i:04d}: {line.strip()}")
