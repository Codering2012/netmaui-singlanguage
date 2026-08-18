import torch
import sys
import tempfile
from pathlib import Path

# Fix sys.path for Kaggle repo
sys.path.insert(0, 'c:\\Users\\Windows 10 21H1\\source\\repos\\Kaggle script\\train')

try:
    from dataset1 import ASLShardedDataset, LandmarkAugmenter
    from train_all_in_one_tpu1 import ASLFoundationModel, HomoscedasticLossWrapper
    print("[+] All modules imported successfully.")
except Exception as e:
    print(f"[-] Import failed: {e}")
    sys.exit(1)

def run_test():
    try:
        # Initialize the model to check if architecture compiles properly
        print("[+] Instantiating ASLFoundationModel...")
        model = ASLFoundationModel(
            d_enc=128,
            nhead_enc=4,
            num_enc_layers=3,
            d_dec=128,
            nhead_dec=4,
            kv_heads_dec=2,
            num_dec_layers=3,
            vocab_size=6152,
            max_enc_len=256,
            dropout=0.1,
            use_mamba=False # disable mamba to simplify test dependencies if it's external
        )
        print("[+] ASLFoundationModel instantiated successfully.")

        print("[+] Instantiating HomoscedasticLossWrapper...")
        loss_wrapper = HomoscedasticLossWrapper()
        print("[+] HomoscedasticLossWrapper instantiated successfully.")

        # Test forward pass with dummy data
        print("[+] Testing forward pass...")
        B, T, V, C = 2, 256, 60, 9 # num_keypoints=60, channels_per_kp=9
        dummy_x = torch.randn(B, T, V, C)
        dummy_mask = torch.ones(B, T, dtype=torch.bool)
        
        # MLM Mask setup
        mlm_mask = (torch.rand(B, T) < 0.15) & dummy_mask
        
        # Forward pass
        out = model(dummy_x, dummy_mask, mlm_mask=mlm_mask, return_aux=True)
        print("[+] Forward pass successful!")
        print(f"    dec_logits shape: {out['dec_logits'].shape if out['dec_logits'] is not None else 'None'}")
        print(f"    mlm_logits shape: {out['mlm_logits'].shape if out['mlm_logits'] is not None else 'None'}")
        print(f"    ctc_log_probs shape: {out['ctc_log_probs'].shape if out['ctc_log_probs'] is not None else 'None'}")

        print("\nAll basic unit tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
