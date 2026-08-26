import os, sys, torch, torch.nn as nn
sys.path.insert(0, "train_tpu")
from dataset import EnglishVocabulary, GlossVocabulary
from train_all_in_one_tpu import ASLFoundationModel

class Phase1TextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder_token_emb = model.decoder.token_emb
        self.english_decoder = model.english_decoder

    def forward(self, safe_input_ids_gloss, safe_input_ids_eng, aslg_mask_b, tgt_in, mask):
        embedded_gloss = self.decoder_token_emb(safe_input_ids_gloss)
        embedded_eng = self.english_decoder.token_emb(safe_input_ids_eng)
        memory = torch.where(aslg_mask_b, embedded_gloss, embedded_eng)
        out = self.english_decoder(tgt_in, memory=memory, memory_key_padding_mask=mask)
        return out[0] if isinstance(out, tuple) else out

def test_wrapper():
    eng_vocab = EnglishVocabulary()
    gloss_vocab = GlossVocabulary("E:/datasets/asl_dataset/asl_preprocessed_phase1/vocab_map.json")
    model = ASLFoundationModel(
        channels_per_kp=9,
        num_enc_layers=0,
        d_enc=512,
        vocab_size=23473,
        d_dec=512,
        nhead_enc=8,
        nhead_dec=8,
        kv_heads_dec=2,
        num_dec_layers=10,
        max_enc_len=384,
        max_dec_len=384,
        english_vocab_size=23473,
        enable_aux_decoders=True,
    )
    
    net = Phase1TextWrapper(model)
    bsz, slen = 4, 384
    g_ids = torch.randint(0, 100, (bsz, slen))
    e_ids = torch.randint(0, 100, (bsz, slen))
    aslg_mask = torch.tensor([True, False, True, False]).unsqueeze(-1).unsqueeze(-1)
    tgt_in = torch.randint(0, 100, (bsz, slen))
    mask = (g_ids == 0)
    
    out = net(g_ids, e_ids, aslg_mask, tgt_in, mask)
    print("Wrapper forward output shape:", out.shape)
    loss = out.sum()
    loss.backward()
    print("Backward successful! Grad on decoder_token_emb:", net.decoder_token_emb.weight.grad is not None)
    print("Grad on english_decoder lm_head:", net.english_decoder.lm_head.weight.grad is not None)

if __name__ == "__main__":
    test_wrapper()
