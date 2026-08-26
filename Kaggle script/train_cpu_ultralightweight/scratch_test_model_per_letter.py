import os
import sys
import torch
import numpy as np

sys.path.insert(0, 'train_cpu_ultralightweight')
from train_cpu import UltraLightweightASLModel

def main():
    model = UltraLightweightASLModel(num_classes=26, in_channels=540, d_model=128, hidden_dim=256)
    ckpt = torch.load('train_cpu_ultralightweight/asl_cpu_model.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    print("[*] Kiểm tra mô hình trên các mẫu thực tế từ shard_0000.pt...")
    d = torch.load('E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_0000.pt', map_location='cpu')
    
    char_correct = {chr(65+i): [0, 0] for i in range(26)}
    
    for item in d:
        lbl = item.get('label', '')
        if not isinstance(lbl, str) or len(lbl) != 1 or not lbl.isalpha():
            continue
        lbl = lbl.upper()
        
        feats = item['features'] # [7, 60, 9]
        feats = feats.reshape(1, 7, -1).float() # [1, 7, 540]
        
        with torch.no_grad():
            out = model(feats)
            logits = out['seq_logits']
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_char = chr(65 + pred_idx)
            
        char_correct[lbl][1] += 1
        if pred_char == lbl:
            char_correct[lbl][0] += 1

    print("\nKết quả kiểm tra theo từng chữ cái:")
    for c, (corr, tot) in char_correct.items():
        if tot > 0:
            acc = corr / tot * 100.0
            print(f"Letter '{c}': {corr}/{tot} ({acc:.1f}%)")
        else:
            print(f"Letter '{c}': 0 mẫu trong shard_0000")

if __name__ == "__main__":
    main()
