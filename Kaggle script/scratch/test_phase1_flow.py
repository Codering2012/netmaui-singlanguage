import os, sys, torch
sys.path.insert(0, "train_tpu")
from dataset import Phase1MixedIterable, EnglishVocabulary, GlossVocabulary, phase1_collate_fn
from train_all_in_one_tpu import compute_seq_and_eos_loss

def test_phase1_flow():
    eng_vocab_path = "E:/datasets/asl_dataset/asl_preprocessed_phase1/english_vocab.json"
    gloss_vocab_path = "E:/datasets/asl_dataset/asl_preprocessed_phase1/vocab_map.json"
    kdwd_dir = "E:/datasets/asl_dataset/wikitext"
    aslg_csv = "E:/datasets/asl_dataset/ASLG-PC12/train.csv"

    if not os.path.exists(eng_vocab_path):
        print("Dataset not present at E:/datasets/asl_dataset, creating synthetic vocabs")
        return

    eng_vocab = EnglishVocabulary(vocab_path=eng_vocab_path, use_bpe=False)
    gloss_vocab = GlossVocabulary(gloss_vocab_path)

    print(f"English vocab size: {len(eng_vocab)}, PAD: {eng_vocab.PAD_ID}, BOS: {eng_vocab.BOS_ID}, EOS: {eng_vocab.EOS_ID}")
    print(f"Gloss vocab size: {len(gloss_vocab)}, PAD: {gloss_vocab.PAD_ID}, BOS: {gloss_vocab.BOS_ID}, EOS: {gloss_vocab.EOS_ID}")

    ds = Phase1MixedIterable(
        kdwd_dir=kdwd_dir,
        aslg_csv=aslg_csv,
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=384,
    )

    it = iter(ds)
    for step in range(50):
        batch = [next(it) for _ in range(8)]
        col = phase1_collate_fn(batch, max_len=384, eng_pad_id=eng_vocab.PAD_ID)
        
        target_ids = col["target_ids"]
        tgt_out = target_ids[:, 1:]
        
        valid_mask = (tgt_out != eng_vocab.PAD_ID)
        tot_valid = valid_mask.sum().item()
        
        # Test loss
        dummy_logits = torch.randn(8, 383, len(eng_vocab))
        loss, _ = compute_seq_and_eos_loss(
            dummy_logits,
            tgt_out,
            valid_mask,
            None,
            label_smoothing=0.1,
            pad_id=eng_vocab.PAD_ID,
            eos_id=eng_vocab.EOS_ID,
        )
        
        if step % 10 == 0:
            print(f"Step {step:02d}: is_dae={col['is_dae'].tolist()}, valid_tokens={tot_valid}, loss={loss.item():.4f}")

if __name__ == "__main__":
    test_phase1_flow()
