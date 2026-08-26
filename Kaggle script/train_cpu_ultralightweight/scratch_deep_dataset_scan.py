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
    shards = sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_*.pt')) + \
             sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/val/shard_*.pt'))
    
    print(f"[*] Deep scanning all {len(shards)} shards for single letters in WLASL, ASL_Citizen, etc...", flush=True)
    
    letter_matches = defaultdict(lambda: defaultdict(int))
    all_sources = Counter()
    potential_letter_labels = defaultdict(Counter)
    
    letters_set = set([chr(65+i) for i in range(26)] + [chr(97+i) for i in range(26)])
    
    for s_idx, s in enumerate(shards):
        try:
            data = torch.load(s, map_location='cpu', weights_only=False)
            for item in data:
                if not isinstance(item, dict):
                    continue
                src = str(item.get('source', 'unknown'))
                all_sources[src] += 1
                raw_lbl = item.get('label')
                lbl_str = str(raw_lbl).strip() if raw_lbl is not None else ""
                
                # Check direct single letter
                if len(lbl_str) == 1 and lbl_str in letters_set:
                    letter_matches[lbl_str.upper()][src] += 1
                elif len(lbl_str) <= 3 or "letter" in lbl_str.lower() or "alphabet" in lbl_str.lower():
                    potential_letter_labels[src][lbl_str] += 1
                    
        except Exception as e:
            print(f"Error on {s}: {e}", flush=True)
            
    print("\n[+] Total Counts per source across all shards:")
    for src, count in all_sources.most_common():
        print(f"  {src:25s}: {count:7d} samples")
        
    print("\n[+] Alphabet letter counts breakdown by source (A - Z):")
    for i in range(26):
        c = chr(65 + i)
        src_dict = letter_matches[c]
        total_c = sum(src_dict.values())
        detail = ", ".join([f"{src}: {cnt}" for src, cnt in src_dict.items()])
        print(f"  Letter '{c}': Total = {total_c:5d} | [{detail}]")
        
    print("\n[+] Short or special label patterns in WLASL / ASL_Citizen / ChicagoFSWild:")
    for src, l_dict in potential_letter_labels.items():
        if src in ['WLASL', 'ASL_Citizen', 'ChicagoFSWild']:
            common = l_dict.most_common(15)
            print(f"  Source '{src}': {common}")

if __name__ == "__main__":
    main()
