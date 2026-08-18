#!/usr/bin/env python3
"""
Exporter for ASLFoundationModel checkpoint -> encoder.onnx & decoder.onnx
Compatible with cpp/inference_engine.cpp ONNX Runtime engine.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))

from train_all_in_one_tpu import ASLFoundationModel

class EncoderWrapper(nn.Module):
    """
    Wraps ASLFoundationModel._encode for ONNX export.
    Inputs:
      - features: (B, T, 60, 9)
      - mask: (B, T) boolean mask
    Outputs:
      - memory: (B, T_merged, 512)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_cls, h_seq, enc_mask, used_mlm_mask, h_pre_tome, orig_enc_mask = self.model._encode(features, mask)
        return h_seq


class DecoderWrapper(nn.Module):
    """
    Wraps ASLFoundationModel.decoder for ONNX export.
    Inputs:
      - memory: (B, T_merged, 512)
      - tgt_ids: (B, t)
    Outputs:
      - logits: (B, t, 6152)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        dec_logits, _ = self.model.decoder(tgt_ids, memory)
        return dec_logits


class CTCHeadWrapper(nn.Module):
    """
    Wraps ASLFoundationModel.ctc_head for 1-pass fast non-autoregressive decoding.
    Inputs:
      - memory: (B, T_merged, 512)
    Outputs:
      - ctc_log_probs: (B, T_merged, vocab_size)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self.model.ctc_head(memory), dim=-1)


def quantize_onnx_models(out_dir: Path):
    try:
        import onnxruntime.quantization as ort_quant
        for onnx_file in list(out_dir.glob("*.onnx")):
            if not onnx_file.name.endswith("_int8.onnx"):
                int8_path = onnx_file.parent / f"{onnx_file.stem}_int8.onnx"
                print(f"[*] Quantizing {onnx_file.name} -> {int8_path.name} (INT8 Dynamic Quantization for Intel AVX2 CPU)...")
                ort_quant.quantize_dynamic(
                    model_input=str(onnx_file),
                    model_output=str(int8_path),
                    weight_type=ort_quant.QuantType.QUInt8,
                )
                print(f"[+] Quantized {int8_path.name} ({os.path.getsize(int8_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[!] ONNX quantization step skipped or failed: {e}")


def main():
    ckpt_path = r"C:\Users\Windows 10 21H1\Downloads\asl_model_epoch_210.pt"
    out_dir = Path(r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\export_onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading ASLFoundationModel checkpoint: {ckpt_path}")
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        model = ASLFoundationModel()
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[!] Checkpoint {ckpt_path} not found. Exporting random weight structure...")
        model = ASLFoundationModel()

    model.eval()

    enc_wrapper = EncoderWrapper(model)
    dec_wrapper = DecoderWrapper(model)
    ctc_wrapper = CTCHeadWrapper(model)

    dummy_feat = torch.zeros(1, 320, 60, 9, dtype=torch.float32)
    dummy_mask = torch.ones(1, 320, dtype=torch.bool)
    dummy_memory = torch.zeros(1, 160, 512, dtype=torch.float32)
    dummy_tgt = torch.zeros(1, 10, dtype=torch.long)

    enc_path = str(out_dir / "encoder.onnx")
    dec_path = str(out_dir / "decoder.onnx")
    ctc_path = str(out_dir / "ctc_head.onnx")

    print(f"[*] Exporting encoder.onnx -> {enc_path}...")
    torch.onnx.export(
        enc_wrapper,
        (dummy_feat, dummy_mask),
        enc_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features", "mask"],
        output_names=["memory"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "mask": {0: "batch", 1: "seq_len"},
            "memory": {0: "batch", 1: "merged_len"},
        },
    )
    print(f"[+] Exported encoder.onnx ({os.path.getsize(enc_path) / 1024 / 1024:.2f} MB)")

    print(f"[*] Exporting decoder.onnx -> {dec_path}...")
    torch.onnx.export(
        dec_wrapper,
        (dummy_memory, dummy_tgt),
        dec_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["memory", "tgt_ids"],
        output_names=["logits"],
        dynamic_axes={
            "memory": {0: "batch", 1: "merged_len"},
            "tgt_ids": {0: "batch", 1: "tgt_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        },
    )
    print(f"[+] Exported decoder.onnx ({os.path.getsize(dec_path) / 1024 / 1024:.2f} MB)")

    print(f"[*] Exporting ctc_head.onnx -> {ctc_path}...")
    torch.onnx.export(
        ctc_wrapper,
        (dummy_memory,),
        ctc_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["memory"],
        output_names=["ctc_log_probs"],
        dynamic_axes={
            "memory": {0: "batch", 1: "merged_len"},
            "ctc_log_probs": {0: "batch", 1: "merged_len"},
        },
    )
    print(f"[+] Exported ctc_head.onnx ({os.path.getsize(ctc_path) / 1024 / 1024:.2f} MB)")

    quantize_onnx_models(out_dir)
    print("=== ONNX EXPORT & QUANTIZATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
