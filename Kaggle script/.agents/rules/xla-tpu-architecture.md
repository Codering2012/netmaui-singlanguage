---
trigger: always_on
---

# PyTorch/XLA & Cloud TPU Architecture Invariants

When developing, refactoring, or optimizing PyTorch models for Cloud TPUs (specifically TPU v5e with PyTorch/XLA):

1. **NO Unrolled Python Loops with Autograd Dependencies**:
   - Python `for` loops in an XLA forward pass that slice or chunk tensors with backward gradients are unrolled during graph tracing into $N$ distinct subgraphs.
   - NEVER chunk projection layers (e.g. `lm_head`) across Python iterations. Use single fused tensor projections or XLA-native scan operators (`torch_xla.experimental.scan`).

2. **NO Gradient Checkpointing on Frozen Weights**:
   - Never enable `gradient_checkpointing_enable()` on models or submodules where all parameters are frozen (`requires_grad = False`).
   - Frozen modules have zero parameter gradients; checkpointing them provides zero activation memory savings while creating branching rematerialization graphs (`remat`) that duplicate attention buffers for each downstream consumer (causing catastrophic HBM OOM and 15+ minute compilations).
   - Always gate: `if gradient_checkpointing and not freeze_model:`.

3. **Strict TPU v5e MXU 128x128 Tile Alignment**:
   - TPU v5e Matrix Multiply Units (MXUs) compute in hardware systolic tiles of $128 \times 128$.
   - Sequence lengths ($L_{\text{prefix}} + L_{\text{text}}$) and batch sizes MUST always be exact multiples of 128 ($128, 256, 384, 512, \dots$).
   - Non-multiples (e.g., 271) cause XLA to zero-pad to the next tile boundary (384), causing severe memory bloat and padding overhead.

4. **Single-Graph Forward Invariance**:
   - Keep the computation graph static and fused. Avoid intermediate dynamic shapes, non-tensor Python branches, or frequent host-device synchronizations (`.item()`, `.cpu()`, logging tensors inside the step loop) that break XLA graph compilation.
