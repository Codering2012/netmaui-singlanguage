import torch
from pathlib import Path
from collections import Counter
import json

DATASET_DIR = Path(r"E:\datasets\asl_dataset\asl_preprocessed_phase1")
OUTPUT_PATH = DATASET_DIR / "english_vocab.json"

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

def main():
    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} does not exist.")
        return

    counter = Counter()
    total_rows = 0
    valid_rows = 0
    empty_rows = 0
    
    print("Processing shards directly...")
    train_dir = DATASET_DIR / "train"
    shard_files = list(train_dir.glob("*.pt"))
    print(f"Found {len(shard_files)} shards in {train_dir}")
    
    for shard_path in shard_files:
        try:
            shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
            items = shard_data.items() if isinstance(shard_data, dict) else enumerate(shard_data)
            
            for key_or_idx, rec in items:
                if not isinstance(rec, dict):
                    continue
                
                source_str = str(rec.get("source", "")).strip()
                task_str = str(rec.get("task", "")).strip()
                raw_label_str = str(rec.get("label", "")).strip()
                
                # Check if it's How2Sign / sentence level
                if task_str == "sentence_level" or source_str.startswith("How2Sign") or raw_label_str == "how2sign_sequence":
                    total_rows += 1
                    
                    sent_text = ""
                    if raw_label_str and raw_label_str != "how2sign_sequence":
                        sent_text = raw_label_str
                        
                    if not sent_text:
                        empty_rows += 1
                        continue
                        
                    valid_rows += 1
                    tokens = sent_text.lower().split()
                    counter.update(tokens)
                    
        except Exception as e:
            print(f"Error reading {shard_path}: {e}")

    # Deterministic ordering:
    # frequency descending, then lexical ascending.
    sorted_tokens = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], kv[0])
    )

    token_to_id = dict(SPECIAL_TOKENS)

    for token, _count in sorted_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    id_to_token = {
        str(idx): token
        for token, idx in token_to_id.items()
    }

    vocab = {
        "version": 1,
        "tokenizer": "lower().split()",
        "source": "E:/datasets/asl_dataset/asl_preprocessed_phase1/train (Extracted from shards)",
        "special_tokens": SPECIAL_TOKENS,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "stats": {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "empty_rows": empty_rows,
            "unique_words": len(counter),
            "vocab_size": len(token_to_id),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("HOW2SIGN ENGLISH VOCABULARY (FROM SHARDS)")
    print("=" * 70)
    print(f"Directory:       {train_dir}")
    print(f"Shards parsed:   {len(shard_files)}")
    print(f"Rows (H2S):      {total_rows:,}")
    print(f"Valid sentences: {valid_rows:,}")
    print(f"Empty sentences: {empty_rows:,}")
    print(f"Unique words:    {len(counter):,}")
    print(f"Vocabulary size: {len(token_to_id):,}")
    print(f"Saved:           {OUTPUT_PATH}")
    print()
    print("First 30 tokens:")
    for token, idx in list(token_to_id.items())[:30]:
        print(f"{idx:5d}  {token}")

if __name__ == "__main__":
    main()
