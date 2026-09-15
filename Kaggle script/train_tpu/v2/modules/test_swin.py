import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyAttn(nn.Module):
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x: (B, L, C)
        # attn_mask: (B, L, L) or (1, L, L)
        # return x
        return x

class Swin1DAttention(nn.Module):
    def __init__(self, mha_module, window_size=128, shift_size=0):
        super().__init__()
        self.mha = mha_module
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, input_x, key_padding_mask=None, frame_indices=None):
        B, L, C = input_x.shape
        
        # Pad to multiple of window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        if pad_l > 0:
            input_x = F.pad(input_x, (0, 0, 0, pad_l))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, pad_l), value=True)
                
        # Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(input_x, shifts=-self.shift_size, dims=1)
            if key_padding_mask is not None:
                shifted_mask = torch.roll(key_padding_mask, shifts=-self.shift_size, dims=1)
            else:
                shifted_mask = None
        else:
            shifted_x = input_x
            shifted_mask = key_padding_mask

        # Partition windows
        num_windows = shifted_x.shape[1] // self.window_size
        x_windows = shifted_x.view(B * num_windows, self.window_size, C)
        
        if shifted_mask is not None:
            mask_windows = shifted_mask.view(B * num_windows, self.window_size)
        else:
            mask_windows = None

        # Masking for shifted windows to prevent cross-boundary attention
        attn_mask = None
        if self.shift_size > 0:
            # Create mask for a single sequence, then expand to B*num_windows
            img_mask = torch.zeros((1, shifted_x.shape[1], 1), device=input_x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                img_mask[:, h, :] = cnt
                cnt += 1
                
            mask_windows_attn = img_mask.view(1, num_windows, self.window_size, 1).view(num_windows, self.window_size)
            # attn_mask: (num_windows, W, W)
            attn_mask = mask_windows_attn.unsqueeze(1) - mask_windows_attn.unsqueeze(2)
            attn_mask = (attn_mask != 0) # True means do not attend
            # Expand to (B * num_windows, W, W)
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_windows, self.window_size, self.window_size)

        # Call inner attention
        # NOTE: If inner attention doesn't support custom attn_mask, we just use key_padding_mask.
        # GroupedQueryEncoderAttention expects key_padding_mask. It builds its own ttn_mask from it.
        # We can pass ttn_mask to it if we modify it.
        
        # To avoid modifying GroupedQueryEncoderAttention, if we have attn_mask, we can't easily pass it 
        # unless we modify GroupedQueryEncoderAttention to accept ttn_mask.
        
        attn_windows = self.mha(x_windows, key_padding_mask=mask_windows, attn_mask=attn_mask)
        
        # Reverse windows
        shifted_x = attn_windows.view(B, num_windows * self.window_size, C)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
            
        # Unpad
        if pad_l > 0:
            x = x[:, :L, :]
            
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyAttn(nn.Module):
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x: (B, L, C)
        # return x
        return x

from train_all_in_one_tpu import Swin1DAttention

attn = Swin1DAttention(DummyAttn(), window_size=128, shift_size=64)
x = torch.randn(2, 300, 256)
out = attn(x)
print('Output shape:', out.shape)
assert out.shape == x.shape
print('Swin1DAttention test passed!')
