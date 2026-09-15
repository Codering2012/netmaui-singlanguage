import sys
sys.path.append(r'c:\Users\Windows 10 21H1\source\repos\Kaggle script\train_tpu')

import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyAttn(nn.Module):
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        return x

from train_all_in_one_tpu import Swin1DAttention, GroupedQueryEncoderAttention

attn = Swin1DAttention(DummyAttn(), window_size=128, shift_size=64)
x = torch.randn(2, 300, 256)
out = attn(x)
print('Output shape:', out.shape)
assert out.shape == x.shape
print('Swin1DAttention test passed!')
