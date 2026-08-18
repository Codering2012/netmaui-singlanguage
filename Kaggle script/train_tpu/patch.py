
with open('train_all_in_one_tpu.py', 'r') as f:
    text = f.read()

# Fix 4: Validation and Training EOS mask
def replace_eos_mask(text):
    # training mask
    old_gloss_mask_train = """            valid_mask = gt_tokens != GlossVocabulary.PAD_ID
            has_valid_gloss = has_valid.bool()
            valid_gloss_mask = valid_mask & has_valid_gloss.unsqueeze(-1)"""
    new_gloss_mask_train = """            token_mask = (gt_tokens != GlossVocabulary.PAD_ID) & has_valid.bool().unsqueeze(-1)
            valid_gloss_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)"""
            
    text = text.replace(old_gloss_mask_train, new_gloss_mask_train)
    text = text.replace("compute_eos_loss(\n                        dec_logits,\n                        gt_tokens,\n                        valid_gloss_mask,", "compute_eos_loss(\n                        dec_logits,\n                        gt_tokens,\n                        token_mask,")
    
    # chicago training
    old_c_train = "valid_chicago_mask = chicago_valid & has_valid_chicago.unsqueeze(-1)"
    new_c_train = "chi_token_mask = chicago_valid & has_valid_chicago.unsqueeze(-1)\n                valid_chicago_mask = chi_token_mask & (chicago_gt != GlossVocabulary.EOS_ID)"
    text = text.replace(old_c_train, new_c_train)
    text = text.replace("compute_eos_loss(\n                    chicago_logits,\n                    chicago_gt,\n                    valid_chicago_mask,", "compute_eos_loss(\n                    chicago_logits,\n                    chicago_gt,\n                    chi_token_mask,")
    
    # english training
    old_e_train = "valid_english_mask = english_valid & has_valid_english.unsqueeze(-1)"
    new_e_train = "eng_token_mask = english_valid & has_valid_english.unsqueeze(-1)\n                valid_english_mask = eng_token_mask & (english_gt != GlossVocabulary.EOS_ID)"
    text = text.replace(old_e_train, new_e_train)
    text = text.replace("compute_eos_loss(\n                    english_logits,\n                    english_gt,\n                    valid_english_mask,", "compute_eos_loss(\n                    english_logits,\n                    english_gt,\n                    eng_token_mask,")

    # validation mask
    old_gloss_mask_val = """                valid_mask = (
                    (gt_tokens != GlossVocabulary.PAD_ID)
                    & (gt_tokens != GlossVocabulary.EOS_ID)
                    & has_valid_gloss.unsqueeze(-1)
                )"""
    new_gloss_mask_val = """                token_mask = (gt_tokens != GlossVocabulary.PAD_ID) & has_valid_gloss.unsqueeze(-1)
                valid_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)"""
    text = text.replace(old_gloss_mask_val, new_gloss_mask_val)
    text = text.replace("compute_eos_loss(\n                        dec_logits, gt_tokens, valid_mask", "compute_eos_loss(\n                        dec_logits, gt_tokens, token_mask")

    # chicago val
    old_c_val = """                    c_valid_mask = (
                        (chicago_gt != GlossVocabulary.PAD_ID)
                        & (chicago_gt != GlossVocabulary.EOS_ID)
                        & has_valid_chicago.unsqueeze(-1)
                    )"""
    new_c_val = """                    chi_token_mask = (chicago_gt != GlossVocabulary.PAD_ID) & has_valid_chicago.unsqueeze(-1)
                    c_valid_mask = chi_token_mask & (chicago_gt != GlossVocabulary.EOS_ID)"""
    text = text.replace(old_c_val, new_c_val)
    text = text.replace("compute_eos_loss(\n                        chicago_logits,\n                        chicago_gt,\n                        c_valid_mask,", "compute_eos_loss(\n                        chicago_logits,\n                        chicago_gt,\n                        chi_token_mask,")

    # english val
    old_e_val = """                    e_valid_mask = (
                        (english_gt != GlossVocabulary.PAD_ID)
                        & (english_gt != GlossVocabulary.EOS_ID)
                        & has_valid_english.unsqueeze(-1)
                    )"""
    new_e_val = """                    eng_token_mask = (english_gt != GlossVocabulary.PAD_ID) & has_valid_english.unsqueeze(-1)
                    e_valid_mask = eng_token_mask & (english_gt != GlossVocabulary.EOS_ID)"""
    text = text.replace(old_e_val, new_e_val)
    text = text.replace("compute_eos_loss(\n                        english_logits,\n                        english_gt,\n                        e_valid_mask,", "compute_eos_loss(\n                        english_logits,\n                        english_gt,\n                        eng_token_mask,")
    
    return text

text = replace_eos_mask(text)

# Fix 5: CrossModalInfoNCE double scaling
text = text.replace(
'''        if world_size > 1 and _XLA_AVAILABLE:
            res = res * world_size''',
''
)
text = text.replace(
'''        if world_size > 1 and IS_TPU:
            res = res * world_size''',
''
)

# Fix 6: Resume scaler
old_scaler = '''    scaler = None
    if args.precision == "float16" and "cuda" in str(device).lower():
        scaler = torch.amp.GradScaler("cuda")'''

text = text.replace(old_scaler, '')

resume_start = '''    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():'''

new_resume = '''    scaler = None
    if args.precision == "float16" and "cuda" in str(device).lower():
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():'''

text = text.replace(resume_start, new_resume)

# Fix 8: CLS uses PE[-1]
old_cls = '''        if frame_indices is not None:
            cls_fi = torch.full(
                (B, 1), -1.0, dtype=frame_indices.dtype, device=frame_indices.device
            )
            frame_indices = torch.cat([cls_fi, frame_indices], dim=1)'''
            
new_cls = '''        if frame_indices is not None:
            frame_indices = frame_indices.long() + 1
            cls_fi = torch.zeros(
                (B, 1), dtype=frame_indices.dtype, device=frame_indices.device
            )
            frame_indices = torch.cat([cls_fi, frame_indices], dim=1)'''
            
text = text.replace(old_cls, new_cls)

# Fix 9: CUDA validation mismatches
old_val_wt = '''                sample_weight = batch.get(
                    "sample_weight", torch.ones(batch["feature"].size(0), device=device)
                )'''

new_val_wt = '''                sample_weight = batch.get(
                    "sample_weight", torch.ones(batch["feature"].size(0), dtype=torch.float32, device=device)
                ).to(device)
                english_trunc_flag = batch.get(
                    "english_trunc", torch.zeros(has_valid_english.shape, dtype=torch.bool, device=device)
                ).to(device)'''
                
text = text.replace(old_val_wt, new_val_wt)

old_e_trunc = '''                if True:
                    e_trunc_c = (
                        batch.get("english_trunc", torch.tensor([False]))
                        .float()
                        .sum()
                    )'''
new_e_trunc = '''                if True:
                    e_trunc_c = (
                        english_trunc_flag
                        .float()
                        .sum()
                    )'''

text = text.replace(old_e_trunc, new_e_trunc)


with open('train_all_in_one_tpu.py', 'w') as f:
    f.write(text)
print("Patched successfully")
