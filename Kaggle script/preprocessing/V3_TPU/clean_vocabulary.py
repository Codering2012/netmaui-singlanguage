import os
import glob
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def clean_vocabulary_mappings(input_root: str):
    input_path = Path(input_root)
    print(f"[*] Scanning preprocessed shards under: {input_path.resolve()}")

    shard_files = sorted(input_path.rglob("shard_*.pt")) + sorted(input_path.rglob("temp_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(input_path.rglob("*.pt"))

    if not shard_files:
        print(f"[!] Error: No shard files found under {input_path}")
        return

    print(f"[*] Discovered {len(shard_files)} shards. Extracting task labels...")

    isolated_labels = set()
    fingerspelling_words = set()
    sentence_transcripts = set()
    all_raw_labels = set()

    for sp in shard_files:
        try:
            records = torch.load(sp, map_location="cpu", weights_only=False)
            for rec in records:
                task = rec.get("task", "unknown")
                lbl = rec.get("label", "").strip().lower()
                if not lbl:
                    continue
                all_raw_labels.add(lbl)

                if task in ("isolated_gloss", "static_alphabet", "isolated_number"):
                    isolated_labels.add(lbl)
                elif task == "fingerspelling_sequence":
                    fingerspelling_words.add(lbl)
                elif task == "sentence_level":
                    sentence_transcripts.add(lbl)
                else:
                    # Heuristic fallback: if label is 1 or 2 words without names, check if isolated
                    words = lbl.split()
                    if len(words) <= 2 and not any(char.isdigit() for char in lbl):
                        isolated_labels.add(lbl)
        except Exception as e:
            print(f"[!] Error loading {sp.name}: {e}")

    # Build Clean Isolated Classification Vocabulary
    sorted_isolated = sorted(list(isolated_labels))
    isolated_map = {lbl: idx for idx, lbl in enumerate(sorted_isolated)}

    # Build Character Token Vocabulary for CTC / Fingerspelling
    char_tokens = ["<pad>", "<none>", "<delete>", " "] + [chr(i) for i in range(ord('a'), ord('z')+1)] + [str(i) for i in range(10)]
    char_map = {tok: idx for idx, tok in enumerate(char_tokens)}

    print("\n" + "="*80)
    print("                    VOCABULARY CLEANING SUMMARY")
    print("="*80)
    print(f"[*] Raw Unfiltered Unique Labels   : {len(all_raw_labels)} (contains How2Sign sentences)")
    print(f"[+] Clean Isolated Class Vocabulary: {len(sorted_isolated)} canonical glosses/letters")
    print(f"[+] Character Token Vocabulary     : {len(char_map)} tokens (for CTC / Fingerspelling)")
    print(f"[-] Filtered Continuous Sentences  : {len(sentence_transcripts)} How2Sign sentences separated")
    print("="*80)

    # Save cleaned vocabulary mapping files
    isolated_out = input_path / "vocabulary_mapping_isolated.json"
    clean_train_out = input_path / "vocabulary_mapping_train.json"
    char_out = input_path / "vocabulary_mapping_character.json"

    with open(isolated_out, "w", encoding="utf-8") as f:
        json.dump({"task": "isolated_classification", "total_classes": len(sorted_isolated), "label_to_idx": isolated_map}, f, indent=2)

    with open(clean_train_out, "w", encoding="utf-8") as f:
        json.dump({"task": "cleaned_primary_classification", "total_classes": len(sorted_isolated), "label_to_idx": isolated_map}, f, indent=2)

    with open(char_out, "w", encoding="utf-8") as f:
        json.dump({"task": "character_ctc", "total_tokens": len(char_map), "token_to_idx": char_map}, f, indent=2)

    print(f"\n[+] Saved cleaned classification map -> {clean_train_out}")
    print(f"[+] Saved isolated gloss map          -> {isolated_out}")
    print(f"[+] Saved character token map         -> {char_out}\n")

if __name__ == "__main__":
    import sys
    target = r"E:\datasets\results\asl_preprocessed_phase1"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    clean_vocabulary_mappings(target)
