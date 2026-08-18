import torch
import numpy as np
from pathlib import Path

train_dir = Path(r'E:\datasets\asl_dataset\asl_preprocessed_phase1\train')
shards = list(train_dir.glob('*.pt'))[:5]

samples = {}
for s in shards:
    try:
        data = torch.load(s, map_location='cpu')
        items = data.items() if isinstance(data, dict) else enumerate(data)
        for k, v in items:
            source = v.get('source', 'unknown')
            if source not in samples:
                samples[source] = v
            if len(samples) == 6: break
    except Exception:
        pass
    if len(samples) == 6: break

for src, sample in samples.items():
    print('\n==================================================')
    print(f'SOURCE: {src}')
    print('==================================================')
    
    # Label info
    lbl = sample.get('label', sample.get('gloss', ''))
    lbl_idx = sample.get('label_idx', -1)
    toks = sample.get('token_ids', sample.get('gloss_seq', []))
    print('Raw Label: "{lbl}" (Index: {lbl_idx})')
    print(f'Token IDs: {toks}')
    
    # Feature info
    feat = sample.get('features', sample.get('feature_array'))
    if feat is not None:
        if isinstance(feat, torch.Tensor): feat = feat.numpy()
        print(f'Feature Shape: {feat.shape}')
        if len(feat) > 0:
            first_frame = feat[0].flatten()
            print(f'First Frame Total Features: {len(first_frame)}')
            
            print('First Frame Sample (First 18 values):')
            print(np.round(first_frame[:18], 4).tolist())
            
            zeros = np.sum(first_frame == 0.0)
            print(f'Zero/Missing values in first frame: {zeros} / {len(first_frame)}')
