import sys
import os
import json
import random
import time
import torch
import numpy as np
from pathlib import Path

sys.path.append("/home/binhhanh409/train")
from train_all_in_one_tpu import ASLFoundationModel, GlossVocabulary

def evaluate_real_accuracy(epoch=10, num_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 70)
    print(f"      STARTING REAL ACCURACY EVALUATION FOR EPOCH {epoch}")
    print("=" * 70)
    print(f"[*] Evaluation Device: {device}")

    data_dir = Path("/home/binhhanh409/results/asl_preprocessed_phase1")
    mapping_path = data_dir / "output_mapping.json"
    if not mapping_path.exists():
        mapping_path = data_dir / "vocabulary_mapping_train.json"

    with open(mapping_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        if isinstance(raw_data, dict) and "label_to_idx" in raw_data:
            label_to_idx = raw_data["label_to_idx"]
        else:
            label_to_idx = raw_data

    clean_label_to_idx = {}
    idx_to_label = {}
    for idx, (k, v) in enumerate(label_to_idx.items()):
        if isinstance(v, (int, float)):
            clean_label_to_idx[str(k)] = int(v)
            idx_to_label[int(v)] = str(k)
        elif isinstance(v, str) and v.isdigit():
            clean_label_to_idx[str(k)] = int(v)
            idx_to_label[int(v)] = str(k)
        elif isinstance(v, str) and not k.startswith("_") and k not in ("name", "num_classes", "task", "description"):
            clean_label_to_idx[str(k)] = idx
            idx_to_label[idx] = str(k)

    vocab = GlossVocabulary(label_to_idx=label_to_idx)
    vocab_size = len(vocab)
    print(f"[*] Vocabulary loaded: {vocab_size} tokens")

    ckpt_path = Path(f"/home/binhhanh409/checkpoints/asl_model_epoch_{epoch}.pt")
    if not ckpt_path.exists():
        print(f"[!] Checkpoint '{ckpt_path.name}' does not exist yet.")
        return

    # Allow file to finish writing
    time.sleep(2)
    print(f"[+] Checkpoint '{ckpt_path.name}' found! Loading weights...")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    ckpt_vocab_size = state_dict["decoder.token_emb.weight"].shape[0]
    print(f"[*] Instantiating ASLFoundationModel with checkpoint vocab_size: {ckpt_vocab_size}")

    model = ASLFoundationModel(vocab_size=ckpt_vocab_size, d_enc=512, d_dec=512).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Find test / val shard files
    test_dir = data_dir / "test"
    if not test_dir.exists() or not list(test_dir.glob("*.pt")):
        test_dir = data_dir / "val"
    if not test_dir.exists() or not list(test_dir.glob("*.pt")):
        test_dir = data_dir / "train"

    shard_files = sorted(list(test_dir.glob("*.pt")))
    print(f"[*] Found {len(shard_files)} dataset shards in '{test_dir.name}'")

    all_records = []
    for sf in shard_files:
        try:
            recs = torch.load(sf, map_location="cpu", weights_only=False)
            for r in recs:
                if isinstance(r, dict) and "features" in r:
                    label = str(r.get("label", "")).strip().lower()
                    if label and len(label.split()) == 1:
                        all_records.append(r)
        except Exception:
            pass

    print(f"[*] Total isolated sign test records collected: {len(all_records)}")
    if not all_records:
        print("[!] Error: No isolated test records found.")
        return

    random.seed(42)
    selected_samples = random.sample(all_records, min(num_samples, len(all_records)))
    print(f"[*] Evaluating real autoregressive accuracy on {len(selected_samples)} test samples...")

    correct_count = 0
    batch_accuracies = []
    wrong_examples = []

    batch_size = 32
    for b_idx in range(0, len(selected_samples), batch_size):
        batch = selected_samples[b_idx : b_idx + batch_size]
        b_correct = 0

        for sample in batch:
            feat = sample["features"]
            if isinstance(feat, np.ndarray):
                feat_t = torch.from_numpy(feat).float()
            else:
                feat_t = feat.float()

            if feat_t.ndim == 2:
                feat_t = feat_t.unsqueeze(0).to(device)
            elif feat_t.ndim == 3:
                feat_t = feat_t.unsqueeze(0).to(device)
            else:
                feat_t = feat_t.to(device)

            gt_label = str(sample.get("label", "")).strip().lower()

            with torch.no_grad():
                # Real Autoregressive Evaluation: Input <BOS> token
                gloss_seq = torch.tensor([[GlossVocabulary.BOS_ID]], dtype=torch.long, device=device)
                
                # Forward pass to get predicted token logits
                out = model(feat_t, gloss_seq=gloss_seq, return_aux=True)
                dec_logits = out["dec_logits"][:, -1, :].float() # (1, V)
                
                # Mask out special tokens (PAD, BOS, EOS, UNK)
                dec_logits[:, :GlossVocabulary.OFFSET] = -1e9
                
                # Apply temperature-scaled softmax decoding (T=0.7)
                temp = 0.7
                probs = torch.softmax(dec_logits / temp, dim=-1)
                top5_ids = torch.topk(probs, k=5, dim=-1).indices[0].tolist()
                pred_token_id = top5_ids[0]

                # Convert predicted token ID to clean label
                pred_raw_idx = pred_token_id - GlossVocabulary.OFFSET
                pred_label = idx_to_label.get(pred_raw_idx, f"<token_{pred_token_id}>").strip().lower()

                if pred_label == gt_label:
                    correct_count += 1
                    b_correct += 1
                else:
                    if len(wrong_examples) < 10:
                        wrong_examples.append({
                            "gt": gt_label,
                            "pred": pred_label,
                            "pred_token_id": pred_token_id
                        })

        batch_acc = (b_correct / len(batch)) * 100.0
        batch_accuracies.append(batch_acc)

    overall_accuracy = (correct_count / len(selected_samples)) * 100.0
    median_accuracy = float(np.median(batch_accuracies))

    print("\n" + "=" * 70)
    print(f"      REAL AUTOREGRESSIVE EVALUATION RESULTS (EPOCH {epoch})")
    print("=" * 70)
    print(f"[*] Total Test Samples Evaluated: {len(selected_samples)}")
    print(f"[+] Total Correct Predictions : {correct_count} / {len(selected_samples)}")
    print(f"[+] REAL MODEL ACCURACY       : {overall_accuracy:.2f}%")
    print(f"[+] MEDIAN BATCH ACCURACY     : {median_accuracy:.2f}%")
    print("=" * 70)

    print("\n--- FIRST 10 WRONG ANSWERS ---")
    for idx, ex in enumerate(wrong_examples, start=1):
        print(f"  {idx:02d}. Ground Truth: '{ex['gt']:<15}' --> Predicted: '{ex['pred']:<15}' (Token ID: {ex['pred_token_id']})")
    print("=" * 70 + "\n")

    # Append output to ~/eval_results.log for persistent logging
    try:
        with open("/home/binhhanh409/eval_results.log", "a", encoding="utf-8") as f_res:
            f_res.write(f"\n=== EPOCH {epoch} EVALUATION ===\n")
            f_res.write(f"Real Model Accuracy: {overall_accuracy:.2f}%\n")
            f_res.write(f"Median Batch Accuracy: {median_accuracy:.2f}%\n")
            f_res.write("First 10 Wrong Answers:\n")
            for idx, ex in enumerate(wrong_examples, start=1):
                f_res.write(f"  {idx:02d}. GT: '{ex['gt']}' --> Pred: '{ex['pred']}'\n")
            f_res.write("-" * 50 + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real Autoregressive Test Set Evaluator")
    parser.add_argument("--epoch", type=int, default=10, help="Epoch checkpoint to evaluate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of randomized test samples")
    parser.add_argument("--monitor", action="store_true", help="Monitor background loop for every 10th epoch (10, 20, 30...)")
    args = parser.parse_args()

    if args.monitor:
        print("[*] Continuous Evaluation Monitor Active for every 10th Epoch (10, 20, 30, ...)...")
        evaluated = set()
        while True:
            for ep in range(10, 130, 10):
                ckpt_p = Path(f"/home/binhhanh409/checkpoints/asl_model_epoch_{ep}.pt")
                if ep not in evaluated and ckpt_p.exists():
                    time.sleep(2)
                    evaluate_real_accuracy(epoch=ep, num_samples=args.samples)
                    evaluated.add(ep)
            time.sleep(10)
    else:
        evaluate_real_accuracy(epoch=args.epoch, num_samples=args.samples)
