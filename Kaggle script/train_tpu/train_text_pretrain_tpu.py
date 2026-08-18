import os
import sys
import argparse
import random
import time
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# PyTorch XLA setup
os.environ["PJRT_DEVICE"] = "TPU"
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.distributed.parallel_loader as pl

from train_all_in_one_tpu import (
    ASLTransformerDecoder,
    GlossVocabulary,
    EnglishVocabulary,
    ModelEMA,
    Poly1CrossEntropyLoss,
    compute_seq_loss,
    setup_env,
)

class ASLGDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, max_len: int = 128, use_bpe: bool = True, bpe_model: str = "Qwen/Qwen2.5-0.5B"):
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['gloss', 'text'])
        
        self.eng_vocab = EnglishVocabulary(use_bpe=use_bpe, model_name=bpe_model)
        self.gloss_vocab = GlossVocabulary()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gloss_str = str(row['gloss'])
        text_str = str(row['text'])

        gloss_ids = self.gloss_vocab.encode(gloss_str, is_chicago=False)
        text_ids = self.eng_vocab.encode(text_str)

        gloss_ids = [self.gloss_vocab.BOS_ID] + gloss_ids + [self.gloss_vocab.EOS_ID]
        text_ids = [self.eng_vocab.BOS_ID] + text_ids + [self.eng_vocab.EOS_ID]

        gloss_ids = gloss_ids[:self.max_len]
        text_ids = text_ids[:self.max_len]

        return {
            "gloss_ids": torch.tensor(gloss_ids, dtype=torch.long),
            "text_ids": torch.tensor(text_ids, dtype=torch.long)
        }

def collate_fn(batch):
    max_gloss = max(len(x['gloss_ids']) for x in batch)
    max_text = max(len(x['text_ids']) for x in batch)

    bsz = len(batch)
    gloss_padded = torch.full((bsz, max_gloss), 0, dtype=torch.long) # 0 is PAD
    text_padded = torch.full((bsz, max_text), 151643, dtype=torch.long) # Qwen PAD
    
    for i, x in enumerate(batch):
        gloss_padded[i, :len(x['gloss_ids'])] = x['gloss_ids']
        text_padded[i, :len(x['text_ids'])] = x['text_ids']

    return {
        "gloss_seq": gloss_padded,
        "english_seq": text_padded
    }

class TextPretrainModel(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 384,
        gloss_vocab_size: int = 4000,
        eng_vocab_size: int = 151936,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.gloss_emb = nn.Embedding(gloss_vocab_size, d_model, padding_idx=0)
        
        # 1-layer text projection encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.gloss_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Decoder
        self.decoder = ASLTransformerDecoder(
            vocab_size=eng_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len,
            pad_id=151643, # Qwen PAD
            mtp_depth=0 # No MTP for english decoder currently
        )

    def forward(self, gloss_seq: torch.Tensor, english_seq: torch.Tensor):
        gloss_mask = (gloss_seq == 0) # True where PAD
        
        memory = self.gloss_emb(gloss_seq)
        memory = self.gloss_encoder(memory, src_key_padding_mask=gloss_mask)

        tgt_in = english_seq[:, :-1]
        tgt_out = english_seq[:, 1:]

        tgt_key_padding_mask = (tgt_in == 151643)

        out = self.decoder(
            tgt_in, memory, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=gloss_mask
        )
        
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        return logits, tgt_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to ASLG-PC12 train.csv")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    device = xm.xla_device()
    setup_env()
    is_master = xm.is_master_ordinal()

    if is_master:
        os.makedirs(args.save_dir, exist_ok=True)
        print("Initializing dataset...")

    dataset = ASLGDataset(args.data_dir, max_len=args.max_len)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True, drop_last=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, collate_fn=collate_fn, drop_last=True
    )

    model = TextPretrainModel(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, max_len=args.max_len,
        gloss_vocab_size=len(dataset.gloss_vocab.token_to_id),
        eng_vocab_size=dataset.eng_vocab.tokenizer.vocab_size if getattr(dataset.eng_vocab, "tokenizer", None) else 151936
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if is_master:
        print(f"Starting Text Pre-training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        para_loader = pl.ParallelLoader(dataloader, [device])
        for step, batch in enumerate(para_loader.per_device_loader(device)):
            gloss_seq = batch["gloss_seq"].to(device)
            english_seq = batch["english_seq"].to(device)

            logits, tgt_out = model(gloss_seq, english_seq)
            
            valid_mask = (tgt_out != 151643) & (tgt_out != 151643) # Qwen padding
            loss = compute_seq_loss(logits, tgt_out, valid_mask, label_smoothing=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            if step % 50 == 0:
                reduced_loss = xm.all_reduce(xm.REDUCE_SUM, loss.detach()) / xm.xrt_world_size()
                if is_master:
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {step} | Loss: {reduced_loss.item():.4f}")

        xm.master_print(f"Epoch {epoch+1} finished.")
        xm.save(model.decoder.state_dict(), os.path.join(args.save_dir, "english_decoder_pretrained.pt"))

if __name__ == "__main__":
    import torch.distributed as dist
    dist.init_process_group("xla", init_method="xla://")
    main()
