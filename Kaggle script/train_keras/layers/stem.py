"""
Multi-Modal Input Stems & Cross-Modal Fusion for ASL Foundation Architecture V2 in Keras 3
Supports:
  1. 60-Keypoint 9-D Kinematics Trajectory Stem (540 dims)
  2. 19-D ASL Phonological Feature Normalization
  3. 256x256 Dual-Stream Upper-Body Visual ROI Stem
  4. Adaptive Gated Cross-Modal Visual-Kinematic Fusion
"""

from typing import Optional
import keras
from keras import layers, ops
from .norm import RMSNorm


class LandmarkTrajectoryStem(layers.Layer):
    """
    Spatiotemporal 1D Convolutional stem for kinematic landmark trajectories.
    Transforms raw [B, T, num_keypoints * channels_per_kp] -> [B, T, out_dim].
    """
    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        out_dim: int = 128,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.out_dim = out_dim
        self.is_causal = is_causal
        self.in_dim = num_keypoints * in_channels
        self.supports_masking = True

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(256, kernel_size=7, padding="same", name="conv1")
        self.norm1 = RMSNorm(256, name="norm1")
        self.conv2 = layers.Conv1D(256, kernel_size=5, padding="same", groups=256, name="conv2")
        self.conv3 = layers.Conv1D(self.out_dim, kernel_size=1, name="conv3")
        self.norm2 = RMSNorm(self.out_dim, name="norm2")
        self.out_proj = layers.Dense(self.out_dim, use_bias=True, name="out_proj")
        super().build(input_shape)

    def call(self, x, mask: Optional[any] = None):
        # Flatten [B, T, K, C] -> [B, T, K*C] if 4D
        if ops.ndim(x) == 4:
            b = ops.shape(x)[0]
            t = ops.shape(x)[1]
            x = ops.reshape(x, (b, t, -1))

        h = ops.gelu(self.norm1(self.conv1(x)))
        h = self.conv2(h)
        h = ops.gelu(self.norm2(self.conv3(h)))
        out = self.out_proj(h)

        if mask is not None:
            mask_exp = ops.expand_dims(ops.cast(mask, ops.dtype(out)), axis=-1)
            out = out * mask_exp
        return out


class VisualROI256Stem(layers.Layer):
    """
    TPU-accelerated depthwise separable visual patch stem for 256x256 upper-body ROI crops.
    Transforms [B, T, 256, 256, 3] -> [B, T, d_model] via strided conv stages and spatial pooling.
    """
    def __init__(self, d_model: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        # Stage 1: 256x256 -> 128x128
        self.conv1 = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False, name="conv1")
        self.norm1 = RMSNorm(64, name="norm1")

        # Stage 2: 128x128 -> 64x64
        self.dw2 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False, name="dw2")
        self.pw2 = layers.Conv2D(128, kernel_size=1, use_bias=False, name="pw2")
        self.norm2 = RMSNorm(128, name="norm2")

        # Stage 3: 64x64 -> 32x32
        self.dw3 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False, name="dw3")
        self.pw3 = layers.Conv2D(256, kernel_size=1, use_bias=False, name="pw3")
        self.norm3 = RMSNorm(256, name="norm3")

        # Stage 4: 32x32 -> 16x16
        self.dw4 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False, name="dw4")
        self.pw4 = layers.Conv2D(self.d_model, kernel_size=1, use_bias=False, name="pw4")
        self.norm4 = RMSNorm(self.d_model, name="norm4")

        # Spatial pooling: 16x16 -> 1 token per frame
        self.global_pool = layers.GlobalAveragePooling2D(name="global_pool")
        super().build(input_shape)

    def call(self, roi_frames):
        # roi_frames: [B, T, 256, 256, 3] or [B, T, 3, 256, 256]
        # Channels last expected in Keras: [B, T, H, W, C]
        if ops.shape(roi_frames)[2] == 3 and ops.shape(roi_frames)[-1] != 3:
            roi_frames = ops.transpose(roi_frames, (0, 1, 3, 4, 2))

        b = ops.shape(roi_frames)[0]
        t = ops.shape(roi_frames)[1]

        # Reshape to (B*T, 256, 256, 3) for batched 2D convolutions
        frames_flat = ops.reshape(roi_frames, (b * t, 256, 256, 3))
        # Normalize uint8 -> [-1, 1] float32
        frames_f = ops.cast(frames_flat, "float32") / 127.5 - 1.0

        # Pass through conv stages
        h = ops.gelu(self.norm1(self.conv1(frames_f)))
        h = ops.gelu(self.norm2(self.pw2(self.dw2(h))))
        h = ops.gelu(self.norm3(self.pw3(self.dw3(h))))
        h = ops.gelu(self.norm4(self.pw4(self.dw4(h))))

        # Pool spatial dimensions (H, W) -> d_model
        tokens = self.global_pool(h)
        return ops.reshape(tokens, (b, t, self.d_model))


class GatedCrossModalFusion(layers.Layer):
    """
    Adaptive Gated Cross-Modal Fusion combining Kinematic Landmarks and Visual ROI.
    alpha = sigmoid(W_gate [h_landmark, h_visual])
    h_fused = h_landmark + alpha * Linear(h_visual)
    """
    def __init__(self, d_model: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.gate_dense1 = layers.Dense(self.d_model, activation="gelu", name="gate_dense1")
        self.gate_dense2 = layers.Dense(self.d_model, activation="sigmoid", name="gate_dense2")
        self.visual_proj = layers.Dense(self.d_model, use_bias=False, name="visual_proj")
        self.out_norm = RMSNorm(self.d_model, name="out_norm")
        super().build(input_shape)

    def call(self, h_landmark, h_visual: Optional[any] = None):
        if h_visual is None:
            return h_landmark

        # Temporal length alignment
        t_land = ops.shape(h_landmark)[1]
        t_vis = ops.shape(h_visual)[1]
        if t_vis != t_land:
            min_t = min(t_vis, t_land)
            h_landmark = h_landmark[:, :min_t]
            h_visual = h_visual[:, :min_t]

        concat = ops.concatenate([h_landmark, h_visual], axis=-1)
        gate = self.gate_dense2(self.gate_dense1(concat))
        v_proj = self.visual_proj(h_visual)

        fused = h_landmark + gate * v_proj
        return self.out_norm(fused)
