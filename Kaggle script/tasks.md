# Dynamic Audit & Self-Testing Task Directory

## File 1: `train_tpu/dataset.py` (3,019 lines)

- [x] **Patch 1 (Lines 1 – 92)**: Imports, environment variables (`XLA_PYTHON_CLIENT_MEM_FRACTION`, `PJRT_ALLOCATOR_FRACTION`, threading caps), Caches, Task Routing Constants, `_SKIP_LABELS`, `normalize_vocabulary()`.
- [x] **Patch 2 (Lines 93 – 280)**: `GlossVocabulary` (Reserved tokens, offset arithmetic, encoding/decoding) & `EnglishVocabulary` (JSON load, fallback candidates, special token contract validation).
- [x] **Patch 3 (Lines 281 – 502) [MATH BLOCK: LandmarkAugmenter]**: Progressive noise curriculum augmentation, 3D spatial scaling, translation, 2D rotation matrix ($R(\theta)$), Gaussian jitter, Time stretching (linear interpolation), Temporal warping, node/finger/hand drop, Kinematics computation ($\vec{v}, \vec{a}$ using actual $\Delta t$).
- [x] **Patch 4 (Lines 503 – 562)**: `motion_aware_sample_indices()` (O(T) L1 motion energy, partition sampling) & `clear_global_dataset_caches()`.
- [x] **Patch 5 (Lines 563 – 714)**: `ASLShardedDataset` Init Part 1 (Directory resolution, Vocab loading fallback, English vocab init).
- [x] **Patch 6 (Lines 715 – 850)**: `ASLShardedDataset` Init Part 2 (ASL-LEX grammatical map loading, POS categories, Shard discovery and worker partitioning).
- [x] **Patch 7 (Lines 851 – 1000)**: `ASLShardedDataset` Init Part 3 (Metadata parsing, Shard grouping, sample indexing, item mapping).
- [x] **Patch 8 (Lines 1001 – 1150)**: `ASLShardedDataset.__getitem__` Part 1 (File loading, `.pt` shard parsing, caching logic).
- [x] **Patch 9 (Lines 1151 – 1300)**: `ASLShardedDataset.__getitem__` Part 2 (Sequence slicing, feature extraction, landmark selection).
- [x] **Patch 10 (Lines 1301 – 1450)**: `ASLShardedDataset.__getitem__` Part 3 (Augmentation pipeline integration, padding to `max_len`, target tensor assembly).
- [x] **Patch 11 (Lines 1451 – 1694)**: `ASLShardedDataset` helper methods & auxiliary features (ASL-LEX attribute tensor formatting, edge case handling).
- [x] **Patch 12 (Lines 1695 – 1772)**: `_seed_worker()` and `ShardPreservingSampler` (Distributed sampler logic for TPU ranks).
- [x] **Patch 13 (Lines 1773 – 1950)**: `ASLStreamedDataset` Init & Generator Setup Part 1 (IterableDataset initialization, streaming pipeline).
- [x] **Patch 14 (Lines 1951 – 2150)**: `ASLStreamedDataset` Generator Setup Part 2 (Shard reading, worker buffer management).
- [x] **Patch 15 (Lines 2151 – 2361)**: `ASLStreamedDataset` Iteration logic, sample batching, error recovery.
- [x] **Patch 16 (Lines 2362 – 2480)**: `create_dataloader()` & `apply_dae_corruptions()` (Denoising AutoEncoder mask corruption math).
- [x] **Patch 17 (Lines 2481 – 2736)**: `KDWDDataset` (Iterable web dataset loading, online sample stream, buffer shuffling).
- [x] **Patch 18 (Lines 2737 – 2885)**: `ASLGPC12Dataset` (GPC12 dataset parsing, frame extraction, vocab matching).
- [x] **Patch 19 (Lines 2886 – 3019)**: `Phase1MixedIterable`, `phase1_collate_fn()`, `phase2_collate_fn()` (Static shape tensor padding, batch collation for PyTorch XLA TPU compatibility).

---

## File 2: `train_tpu/train_all_in_one_tpu.py` (8,185 lines)

- [x] **Patch 20 (Lines 1 – 187)**: Monolithic header, Imports, PyTorch XLA environment setup (`LIBTPU_INIT_ARGS`, `XLA_PERSISTENT_CACHE_PATH`, `XLA_USE_BF16`), `get_xla_world_size()`, `_distributed_normalize()`, `_safe_torch_device()`.
- [x] **Patch 21 (Lines 188 – 402) [MATH BLOCK: RMSNorm, SwiGLU, RichASLLexEmbeddingTable]**:
  - `RMSNorm`: $y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$
  - `SwiGLUFFN`: $\text{SwiGLU}(x) = W_{down}(\text{SiLU}(W_{gate}x) \odot W_{up}x)$ with TPU 128-alignment.
  - `RichASLLexEmbeddingTable`: Multi-attribute embedding projection (LexicalClass, SignType, Handshape, Location, SemanticCategory, Flexion, Transparency, Iconicity).
- [x] **Patch 22 (Lines 403 – 543) [MATH BLOCK: TemporalStridedPool & DropPath]**: `drop_path()` stochastic depth, `TemporalStridedPool` strided frame reduction with causal/non-causal padding and token size tracking, `DropPath`.
- [x] **Patch 23 (Lines 544 – 722) [MATH BLOCK: RotaryPositionalEncoding & GroupedQueryEncoderAttention (GQA + MLA)]**:
  - `RotaryPositionalEncoding` (RoPE): $R_{\Theta, m}^d x_m$ with pre-cached sine/cosine and frame index alignment.
  - `GroupedQueryEncoderAttention`: DeepSeek V3 MLA Latent Compression ($c_t^{KV} = W^{DKV} h_t$), RoPE splitting, scaled dot-product attention, causal lookahead mask.
- [x] **Patch 24 (Lines 723 – 818) [MATH BLOCK: Swin1DAttention]**: Shifted 1D window attention (`torch.roll`, window partitioning, boundary masking).
- [x] **Patch 25 (Lines 819 – 933) [MATH BLOCK: SpatialTemporalSE & ConvNeXtTemporalBlock]**: Squeeze-and-Excitation channel attention, ConvNeXt 1D depthwise separable convolution ($7 \times 1$), LayerNorm, inverted bottleneck ($4\times$).
- [x] **Patch 26 (Lines 934 – 1227) [MATH BLOCK: BiMamba2SSMBlock]**: Bidirectional Mamba-2 State Space Model block ($\mathbf{h}'_t = \mathbf{A}\mathbf{h}_t + \mathbf{B}\mathbf{x}_t, \mathbf{y}_t = \mathbf{C}\mathbf{h}_t + \mathbf{D}\mathbf{x}_t$), selective scan discretization, forward/backward fusion.
- [x] **Patch 27 (Lines 1228 – 1357) [MATH BLOCK: MobileConformerBlock]**: Macaron-style FFN + MHSA/Swin + Depthwise Conv Module + FFN structure.
- [x] **Patch 28 (Lines 1358 – 1559)**: `LandmarkTrajectory1DStem`, `MaskedGroupNorm`, `VisualStem` (Spatial-temporal landmark embedding, depthwise conv, group norm, linear projection).
- [x] **Patch 29 (Lines 1560 – 1829) [MATH BLOCK: Decoder RoPE & GroupedQueryAttention]**: Decoder self-attention with RoPE, causal mask, GQA projection.
- [x] **Patch 30 (Lines 1830 – 2009) [MATH BLOCK: Decoder CrossAttention & ASLDecoderLayer]**: Cross-attention over encoder representations, feedforward sublayers, residual connections.
- [x] **Patch 31 (Lines 2010 – 2231)**: `ASLTransformerDecoder` (Multimodal decoder with text embedding, positional encoding, stacked decoder layers, linear head).
- [x] **Patch 32 (Lines 2232 – 2555) [MATH BLOCK: Auxiliary Losses - Homoscedastic, CosineLinear, CTC, InfoNCE]**:
  - `HomoscedasticLossWrapper`: Learnable task variance weighting $\mathcal{L}_{total} = \sum \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$.
  - `CosineLinear`: Normalized cosine similarity logits.
  - `CTCHead`: Logits for Connectionist Temporal Classification.
  - `CrossModalInfoNCELoss`: Scaled symmetric InfoNCE loss with cross-replica `all_gather`. $\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_v, z_t)/\tau)}{\sum \exp(\text{sim}/\tau)}$.
- [x] **Patch 33 (Lines 2556 – 2780) [MATH BLOCK: Sentence Semantic & Supervised Contrastive Losses]**:
  - `DenseSentenceSemanticLoss`: Cosine distance / MSE loss between predicted embeddings and text sentence embeddings.
  - `SupervisedContrastiveLoss`: SupCon loss over class labels $\mathcal{L}_{SupCon} = \sum_{i} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a} \exp(z_i \cdot z_a / \tau)}$.
- [x] **Patch 34 (Lines 2781 – 2916)**: `GradientReversalFunction` (Domain adversarial training $G_r(x) = x, \nabla G_r(x) = -\lambda I$), `LandmarkReconstructionHead`, `PositionalEncoding1D`.
- [x] **Patch 35 (Lines 2917 – 3150) [MATH BLOCK: ASLFoundationModel Architecture & Forward Part 1]**: Monolithic initialization, embedding projection, stem forward pass.
- [x] **Patch 36 (Lines 3151 – 3400) [MATH BLOCK: ASLFoundationModel Forward Part 2]**: Encoder forward pass through Conformer/Mamba blocks, ToMe pooling, feature masking.
- [x] **Patch 37 (Lines 3401 – 3650) [MATH BLOCK: ASLFoundationModel Forward Part 3]**: Decoder forward pass, autoregressive token generation, cross-attention fusion.
- [x] **Patch 38 (Lines 3651 – 3848) [MATH BLOCK: ASLFoundationModel Loss Computation]**: Multi-task loss aggregation, homoscedastic weighting, auxiliary loss calculation.
- [x] **Patch 39 (Lines 3849 – 4200)**: `ModelEMA` (Exponential Moving Average of model parameters $\theta_{EMA} \leftarrow \beta \theta_{EMA} + (1-\beta) \theta$), Decoding routines (Greedy, CTC Beam Search, Prefix Beam Search), Metric calculation (WER, BLEU-4, ROUGE-L).
- [x] **Patch 40 (Lines 4201 – 4700)**: Hyperparameter parsing, CLI arguments (`argparse`), device selection, TPU XLA rank initialization.
- [x] **Patch 41 (Lines 4701 – 5200)**: Dataloader initialization, vocabulary binding, model construction, EMA model setup.
- [x] **Patch 42 (Lines 5201 – 5700)**: Optimizer creation (AdamW with decoupled weight decay), Cosine Annealing scheduler with warmup equations ($\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t}{T}\pi))$).
- [x] **Patch 43 (Lines 5701 – 6300)**: Phase 1 Training Loop execution, batch iteration, XLA mark_step, gradient accumulation, loss logging.
- [x] **Patch 44 (Lines 6301 – 7000)**: Phase 1 Validation & Evaluation loop, metric computation across ranks, checkpoint saving.
- [x] **Patch 45 (Lines 7001 – 7600)**: Phase 2 Training Loop execution (Sentence fine-tuning with autoregressive decoder).
- [x] **Patch 46 (Lines 7601 – 8185)**: Phase 2 Validation, final evaluation, main entry point execution, error handling.
