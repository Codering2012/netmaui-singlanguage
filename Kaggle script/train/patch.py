import os

train_all_path = r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train\train_all_in_one_tpu.py"
dataset_path = r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train\dataset.py"
train_tpu_path = r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train\train_tpu.py"

# 1. Read the original files
with open(train_all_path, 'r', encoding='utf-8') as f:
    all_content = f.read()

with open(dataset_path, 'r', encoding='utf-8') as f:
    ds_content = f.read()

with open(train_tpu_path, 'r', encoding='utf-8') as f:
    tp_content = f.read()

# 2. Extract Dataset code
ds_start = ds_content.find("_GLOBAL_SHARD_DATA_CACHE = {}")
if ds_start == -1:
    ds_start = ds_content.find("class ASLShardedDataset")
ds_code = ds_content[ds_start:]

# 3. Extract train code
worker_start = tp_content.find("def _tpu_worker_fn")
tp_code = tp_content[worker_start:]

# 4. Modify Model Instantiation
old_model_init = """    if getattr(args, "arch", "ultralight") == "ultralight":
        log_msg(f"[*] Instantiating UltraLightSignModel (~0.8M parameters, optimized for real-time 30+ FPS CPU execution)...", is_master=is_master)
        model = UltraLightSignModel(
            num_classes=num_classes,
            num_keypoints=60,
            channels_per_kp=9,
            d_model=getattr(args, "d_model", 128),
            nhead=getattr(args, "nhead", 4),
            num_layers=getattr(args, "num_layers", 3),
            dim_feedforward=getattr(args, "dim_feedforward", 256),
            dropout=args.dropout,
            max_len=args.max_len
        ).to(device)
    else:
        log_msg(f"[*] Instantiating LandmarkTransformer (~5.5M parameters)...", is_master=is_master)
        model = LandmarkTransformer(
            num_classes=num_classes,
            num_keypoints=60,
            channels_per_kp=9,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_len
        ).to(device)"""

new_model_init = """    log_msg(f"[*] Instantiating ASLFoundationModel (~31M parameters)...", is_master=is_master)
    model = ASLFoundationModel(
        vocab_size=num_classes,
        num_keypoints=60,
        channels_per_kp=9,
        d_enc=320,
        nhead_enc=8,
        num_enc_layers=8,
        ffn_enc=1280,
        d_dec=320,
        nhead_dec=8,
        num_dec_layers=8,
        ffn_dec=1280,
        dropout=args.dropout
    ).to(device)"""

if old_model_init in tp_code:
    tp_code = tp_code.replace(old_model_init, new_model_init)
else:
    # Use fallback string replacement if exact match fails
    tp_code = tp_code.replace("UltraLightSignModel", "ASLFoundationModel")
    tp_code = tp_code.replace("num_classes=num_classes,", "vocab_size=num_classes,")

# 5. Remove the "print('[+] ... module compiled successfully.')" at the end of all_content
lines = all_content.splitlines()
if len(lines) > 0 and 'module compiled successfully' in lines[-1]:
    lines = lines[:-1]
all_content = "\\n".join(lines) + "\\n\\n"

# 6. Append and save
final_content = all_content + "\\n# ==============================================================================\\n# 7. DATASET & DATALOADER\\n# ==============================================================================\\n\\n" + ds_code + "\\n\\n# ==============================================================================\\n# 8. TRAINING LOOP & MAIN\\n# ==============================================================================\\n\\n" + tp_code

with open(train_all_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print(f"Successfully appended {len(ds_code)} chars of dataset and {len(tp_code)} chars of main loop to {train_all_path}")
