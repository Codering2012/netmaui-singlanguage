import sys
import os
import glob
from collections import Counter, defaultdict
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    train_shards = sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_*.pt'))
    val_shards = sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/val/shard_*.pt'))
    all_shards = train_shards + val_shards
    
    print(f"[*] Scanning all {len(all_shards)} shards ({len(train_shards)} train + {len(val_shards)} val)...", flush=True)
    
    letter_counts = Counter()
    source_counts = Counter()
    letter_sources = defaultdict(Counter)
    task_counts = Counter()
    total_samples = 0
    total_alphabet_samples = 0
    lengths_per_letter = defaultdict(list)
    
    valid_letters = set([chr(65+i) for i in range(26)] + [chr(97+i) for i in range(26)])
    
    for s_idx, s_path in enumerate(all_shards):
        try:
            data = torch.load(s_path, map_location='cpu', weights_only=False)
            s_name = os.path.basename(s_path)
            shard_alph_cnt = 0
            
            for item in data:
                total_samples += 1
                if not isinstance(item, dict):
                    continue
                
                src = item.get('source', 'unknown')
                source_counts[src] += 1
                
                task = item.get('task', 'unknown')
                task_counts[task] += 1
                
                raw_lbl = item.get('label')
                lbl_str = str(raw_lbl).strip() if raw_lbl is not None else ""
                
                # Check if it is a single alphabet letter
                if len(lbl_str) == 1 and lbl_str in valid_letters:
                    char_upper = lbl_str.upper()
                    letter_counts[char_upper] += 1
                    letter_sources[char_upper][src] += 1
                    total_alphabet_samples += 1
                    shard_alph_cnt += 1
                    
                    f = item.get('features')
                    if f is not None and hasattr(f, 'shape'):
                        lengths_per_letter[char_upper].append(f.shape[0])
                        
            print(f"Shard {s_name} [{s_idx+1}/{len(all_shards)}]: total={len(data)}, alphabet_samples={shard_alph_cnt}", flush=True)
        except Exception as e:
            print(f"[!] Error on {s_path}: {e}", flush=True)
            
    print("\n" + "="*80)
    print(f"TOTAL SAMPLES ACROSS ALL SHARDS: {total_samples:,}")
    print(f"TOTAL SINGLE-LETTER ALPHABET SAMPLES: {total_alphabet_samples:,}")
    print("="*80)
    
    print("\n[+] Per-Letter Sample Distribution (A - Z):")
    for i in range(26):
        c = chr(65 + i)
        cnt = letter_counts[c]
        lens = lengths_per_letter[c]
        avg_len = sum(lens) / max(1, len(lens)) if lens else 0
        sources_str = ", ".join([f"{src}: {sc}" for src, sc in letter_sources[c].most_common(3)])
        print(f"  Letter '{c}': {cnt:6d} samples | Avg Frame Length: {avg_len:4.1f} frames | Sources: [{sources_str}]")
        
    print("\n[+] Total Counts By Source:")
    for src, cnt in source_counts.most_common():
        print(f"  Source '{src}': {cnt:7d} samples")

if __name__ == "__main__":
    main()
