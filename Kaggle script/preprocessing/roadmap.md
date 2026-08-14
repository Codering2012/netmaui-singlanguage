# OSR-ASL Production Roadmap

This roadmap turns the design spec and the local dataset analysis into a staged execution plan that is practical to ship and easy to harden.

## What Is In Scope

- Static-image alphanumeric sign recognition for the local ASL alphabet and digit datasets.
- A production-ready baseline training scaffold with reproducible config, run manifests, and artifact layout.
- A hand-vision upgrade path that includes the foundation-stack items from the blueprint.

## What Is Already In The Repo

- Local metadata files and lexicon sources.
- Dataset roots listed in `dataset_path.txt`.
- Dataset analysis and blueprint generation in `osr_asl_pipeline.py`.
- A first-pass training blueprint in `outputs/training_blueprint.json`.

## Production Principles

- Keep every run reproducible.
- Separate data preparation from model training.
- Fail fast on missing paths, schema errors, or empty datasets.
- Save immutable artifacts for every run.
- Treat quality routing, masking, and calibration as first-class features, not afterthoughts.

## Stage 0: Data Contract And Run Hygiene

Goal:
Establish a stable contract for paths, outputs, seeds, and dataset manifests.

Deliverables:

- `train.yaml`
- `train_sign_pipeline.py`
- Run-scoped artifact directories under `artifacts/runs/<run_id>`
- Config snapshots and dataset manifests for every execution

Exit criteria:

- A run can be prepared without training.
- Missing dataset roots are detected before GPU work begins.
- The same seed produces the same split on repeated runs.

## Stage 1: Baseline Image Classifier

Goal:
Ship a dependable baseline that trains on the folder-structured image datasets already present in the archive roots.

Implementation:

- Train `asl_alphabet` and `numbers` as separate experiments.
- Use a pretrained-compatible classifier head, with offline-safe random initialization by default.
- Use stratified validation splits.
- Save best and last checkpoints.
- Report top-1, top-5, and loss for validation and test manifolds.

Exit criteria:

- The baseline reaches a stable training loop with checkpointing and evaluation.
- Metrics are written to disk for each experiment.
- Inference-ready class maps are saved with the checkpoints.

## Stage 2: Hand-Centric Preprocessing

Goal:
Reduce background noise, pose drift, and crop instability before the classifier sees the image.

Foundation items:

1. Hand foundation model pretraining with DINOv2 or EVA-CLIP.
2. SAM2 hand tracking.
3. Landmark graph transformers.
4. Diffusion-based motion deblurring.

Implementation:

- Detect hands first.
- Stabilize masks across adjacent frames or adjacent samples.
- Expand the crop to preserve wrist and lower-forearm context.
- Canonicalize handedness and scale.
- Route low-quality frames through a restoration branch.

Exit criteria:

- The preprocessing layer produces cached tracklets or cleaned crops.
- The model can consume raw and normalized views side by side.

## Stage 3: Temporal And Pose Modeling

Goal:
Move from a pure image classifier to a spatiotemporal model that can reason about motion and keypoints.

Foundation items:

- VideoMAE temporal pretraining.
- Landmark graph transformers.
- Uncertainty-aware decoding.

Implementation:

- Add a temporal encoder for frame sequences or pseudo-tracklets.
- Fuse landmarks, handedness, and motion cues.
- Emit multiple hypotheses when the frame is ambiguous.
- Calibrate confidence so the system can reject low-trust predictions.

Exit criteria:

- The model can consume a short sequence or tracklet instead of a single frame.
- Confidence scores are calibrated and logged.

## Stage 4: Synthetic And Semi-Supervised Scale-Up

Goal:
Expand coverage without requiring linear growth in manual labels.

Foundation items:

- Synthetic signer generation.
- Self-training on millions of unlabeled sign clips.

Implementation:

- Generate synthetic hands, signers, occlusions, and camera conditions.
- Mix synthetic and real samples with curriculum control.
- Pseudo-label the easiest unlabeled clips first.
- Retrain iteratively on the cleanest high-confidence predictions.

Exit criteria:

- The model improves on held-out hard cases without collapsing on clean data.
- Pseudo-label quality is tracked and bounded by confidence thresholds.

## Stage 5: Context And Language Modeling

Goal:
Use sequence context to resolve visually ambiguous tokens.

Foundation items:

- Sign-language language models.
- Multi-camera training.

Implementation:

- Add a lightweight language model to rerank uncertain sequences.
- Train on multiple camera viewpoints when available.
- Preserve metadata such as view angle, signer identity, and capture source.

Exit criteria:

- The system improves ambiguous sequences without hurting clean single-frame accuracy.

## Stage 6: Production Deployment

Goal:
Make the pipeline safe to run repeatedly in a real environment.

Implementation:

- Add configuration validation.
- Add model and data versioning.
- Add drift monitoring on blur, brightness, and confidence.
- Add rollback-friendly checkpoints.
- Log the exact dataset root, config hash, and code revision for each run.

Exit criteria:

- A run can be reproduced from the saved artifacts alone.
- Operators can inspect failures without rerunning the full training job.

## Recommended Build Order

1. Land the baseline trainer and run manifest flow.
2. Add dataset scanning and evaluation on the test manifolds.
3. Introduce quality routing and tracklet caching.
4. Add landmark-aware preprocessing and pose fusion hooks.
5. Wire in self-training and synthetic data once the supervised loop is stable.
6. Add multi-camera and language-model reranking as late-stage robustness layers.

## Risks And Mitigations

- Risk: Overfitting to clean samples.
  - Mitigation: curriculum training, harder augmentations, and test-set evaluation.
- Risk: Bad crops and blur destroy accuracy.
  - Mitigation: quality routing, restoration only on low-quality samples, and cached cleaned crops.
- Risk: Reproducibility drift.
  - Mitigation: deterministic seeds, config snapshots, and immutable run directories.
- Risk: Future model complexity grows too fast.
  - Mitigation: keep the baseline classifier as the stable fallback and add one module at a time.

## Definition Of Done

- A config-driven baseline training run can complete end to end.
- The dataset manifests and run metadata are saved.
- The roadmap items are mapped to concrete implementation phases.
- The codebase can evolve toward the foundation-stack items without breaking the baseline.
