import torch
import sys
import json
import os
import time

sys.path.append('/home/binhhanh409/train')
from train_all_in_one_tpu import ASLFoundationModel, GlossVocabulary

def eval_epoch(epoch=20, num_samples=2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    vocab_path = '/home/binhhanh409/results/asl_preprocessed_phase1/vocabulary_mapping_global.json'
    with open(vocab_path, 'r') as f:
        vocab_mapping = json.load(f)
    
    gloss_vocab = GlossVocabulary(vocab_mapping)
    model = ASLFoundationModel(vocab_size=len(gloss_vocab)).to(device)
    
    ckpt_path = f'/home/binhhanh409/checkpoints/asl_model_epoch_{epoch}.pt'
    if not os.path.exists(ckpt_path):
        print(f"Waiting for {ckpt_path} to be available...")
        while not os.path.exists(ckpt_path):
            time.sleep(10)
    
    print(f"Loading {ckpt_path}...")
    checkpoint = None
    while checkpoint is None:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Failed to load checkpoint (maybe still writing?): {e}. Retrying in 10s...")
            time.sleep(10)
    
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    
    inv_vocab = {v: k for k, v in vocab_mapping.items()}
    
    # Load validation shards
    val_dir = '/home/binhhanh409/results/asl_preprocessed_phase1/val/'
    shard_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.pt')])
    
    correct_exact = 0
    total_samples = 0
    total_wer = 0.0
    
    print(f'Evaluating up to {num_samples} samples...')
    import numpy as np
    
    for shard in shard_files:
        if total_samples >= num_samples:
            break
            
        data = torch.load(os.path.join(val_dir, shard), weights_only=False)
        for i in range(len(data)):
            if total_samples >= num_samples:
                break
                
            sample = data[i]
            features_raw = sample['features']
            if isinstance(features_raw, np.ndarray):
                features = torch.from_numpy(features_raw).float()
            else:
                features = features_raw.float()
                
            if features.dim() == 3:
                frames = features.unsqueeze(0).to(device)
            else:
                frames = features.unsqueeze(0).unsqueeze(0).to(device)
                
            T = frames.size(1)
            # Calculate real valid frame mask (non-zero frames)
            has_valid = (frames.abs().sum(dim=(-1, -2, -3)) > 1e-5) # (1, T)
            if not has_valid.any():
                has_valid[:, :10] = True # fallback if all zero
            
            seq = sample.get('label_idx', None)
            if isinstance(seq, int):
                seq = [seq]
            elif isinstance(seq, torch.Tensor) and seq.dim() == 0:
                seq = [seq.item()]
                
            target_words = []
            if seq is not None:
                for idx in seq:
                    val = idx.item() if isinstance(idx, torch.Tensor) else idx
                    target_words.append(inv_vocab.get(val, f'<{val}>'))
            target_sentence = ' '.join(target_words)
            
            with torch.no_grad():
                B = frames.size(0)
                # Test with full target sequence
                target_gloss_full = torch.tensor([[GlossVocabulary.BOS_ID, seq[0] + GlossVocabulary.OFFSET, GlossVocabulary.EOS_ID]], dtype=torch.long, device=device)
                outputs_full = model(frames, None, target_gloss_full, None, return_aux=True)
                pred_full = outputs_full['dec_logits'].argmax(dim=-1)
                print(f"  Pred (with full teacher forcing): {pred_full.tolist()} (Expected: [{seq[0] + GlossVocabulary.OFFSET}, {GlossVocabulary.EOS_ID}])")
                
                # Test with just BOS
                dummy_gloss = torch.full((B, 2), GlossVocabulary.BOS_ID, dtype=torch.long, device=device)
                outputs = model(frames, None, dummy_gloss, None, return_aux=True)
                dec_logits = outputs['dec_logits'] # (B, 1, V)
                predictions = dec_logits[:, 0, :].argmax(dim=-1) # (B,)
                
                pred_words = []
                last_idx = -1
                for p in predictions:
                    idx = p.item()
                    if idx not in [GlossVocabulary.PAD_ID, GlossVocabulary.BOS_ID, GlossVocabulary.EOS_ID]:
                        pred_words.append(inv_vocab.get(idx - GlossVocabulary.OFFSET, f'<{idx}>'))
                
                pred_sentence = ' '.join(pred_words)
            
            if total_samples < 5:
                print(f"Sample {total_samples}:")
                print(f"  Target: {target_sentence}")
                print(f"  Pred  : {pred_sentence} (Raw idxs: {predictions.tolist()})")
            
            if pred_sentence == target_sentence:
                correct_exact += 1
                
            # Basic Word Error Rate (WER) using Levenshtein distance
            def levenshtein(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                return previous_row[-1]
                
            p_list = pred_sentence.split()
            t_list = target_sentence.split()
            if len(t_list) > 0:
                wer = levenshtein(p_list, t_list) / len(t_list)
            else:
                wer = 1.0 if len(p_list) > 0 else 0.0
                
            total_wer += wer
            total_samples += 1
            
            if total_samples % 100 == 0:
                print(f"Evaluated {total_samples}/{num_samples} samples... Exact: {correct_exact/total_samples:.2%} | Avg WER: {total_wer/total_samples:.2f}", flush=True)

    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Total Samples  : {total_samples}")
    print(f"Exact Match Acc: {correct_exact/total_samples:.2%}")
    print(f"Average WER    : {total_wer/total_samples:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--samples', type=int, default=2000)
    args = parser.parse_args()
    eval_epoch(epoch=args.epoch, num_samples=args.samples)
