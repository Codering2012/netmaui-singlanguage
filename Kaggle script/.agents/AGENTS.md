### Empirical Code Audit & Anti-Hallucination Workflow
When presented with lists of bug claims, architectural critiques, or optimization proposals:
- **Audit Before Modifying**: Inspect authoritative source files and trace tensor shapes, parameters, and execution flow before making any edits.
- **Automated Self-Test Hypothesis Scripts**: Write dedicated, runnable hypothesis self-test scripts (e.g. in `scratch/test_claims.py`) to empirically test claims against live isolated tensors and functions before reaching a final verdict.
  - *Local Hardware Constraint (16GB i5-8250U)*: All local self-tests and verification scripts MUST use lightweight mock dimensions ($B \le 4$, $L \le 64$, $D \le 128$, vocab $\le 500$) and never instantiate production-scale models or high-concurrency worker pools on CPU to prevent allocator out-of-memory errors and system lockups.
- **Mathematical & Code Debunking**: For claims that are invalid, explain precisely why they are false using mathematical proofs (e.g. broadcasting rules, tensor dimensions, derivative properties), direct code references, and empirical outputs from self-test scripts.
- **Surgical Fixes for Verified Bugs**: Implement precise fixes only for claims that are verified to be true bugs, preserving overall system contracts.
- **Mandatory Empirical Verification**: Always run compilation checks (`py_compile`/`pyflakes`) and execute automated unit tests to verify zero regressions before declaring success.

### PyTorch/XLA & TPU v5e Architecture Invariants
- **No Unrolled Chunk Loops**: Do not unroll loops over tensor projections in the forward pass; unrolling causes graph cloning, remat duplication explosion (120+ cloned attention buffers), 15+ minute compilations, and device HBM OOM.
- **No Gradient Checkpointing on Frozen Models**: Never enable checkpointing when weights are frozen (`if gradient_checkpointing and not freeze_model:`).
- **128x128 Tile Alignment**: All sequence lengths ($L_{\text{prefix}} + L_{\text{text}}$) and batch dimensions must be exact multiples of 128 to match TPU v5e MXU systolic array tiles.

