# Dataset Audit & 30 FPS Sequence Visualization Report

## 1. Dataset Technical Audit (`asl_preprocessed_phase1`)

### Path Overview
- **Kaggle Path**: `/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset/asl_preprocessed_phase1`
- **Local Path**: `E:\datasets\asl_dataset\asl_preprocessed_phase1`

---

### Vocabulary & Tokenizer Audit (`english_vocab.json`)
- **Total Vocabulary Size**: **23,473 Tokens**
- **Tokenizer Type**: Byte-Pair Encoding (BPE) Subword Tokenizer
- **Special Token Registry**:
  - `PAD_ID = 0` (`<pad>`)
  - `UNK_ID = 1` (`<unk>`)
  - `BOS_ID = 2` (`<bos>`)
  - `EOS_ID = 3` (`<eos>`)

---

### Multi-Modal Landmark Tensor Schema
Every sign language gesture sequence is pre-processed into a 9-channel feature tensor:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times T \times V}$$

- **Spatial Keypoints ($V = 540$)**:
  - **Pose Skeleton**: 33 Keypoints (Upper body, shoulders, elbows, wrists)
  - **Face Mesh**: 468 Keypoints (Facial expressions, lips, mouth movement)
  - **Left Hand**: 21 Keypoints (Wrist, knuckles, finger joints)
  - **Right Hand**: 21 Keypoints (Wrist, knuckles, finger joints)
- **Feature Channels ($C = 9$)**:
  - `[0:3]`: 3D Spatial Coordinates $(x, y, z)$
  - `[3:6]`: 1st Order Temporal Velocity ($\Delta x, \Delta y, \Delta z$)
  - `[6:9]`: 2nd Order Temporal Acceleration ($\Delta^2 x, \Delta^2 y, \Delta^2 z$)
- **Frame Rate & Temporal Bounds**:
  - **Sampling Frequency**: **30 FPS** (33.3 milliseconds per frame)
  - **Sequence Length ($T$)**: 384 frames (up to 12.8 seconds of video)

---

## 2. 30 FPS Rendered Landmark Sample Frames

Generated 30 FPS visualizations are saved directly in the `INSTRUCTIONS/landmark_samples_30fps/` directory:

- [frame_01.png](file:///c:/Users/Windows%2010%2021H1/source/repos/Kaggle%20script/INSTRUCTIONS/landmark_samples_30fps/frame_01.png) (Frame 1 @ 30 FPS - Gesture Start)
- [frame_05.png](file:///c:/Users/Windows%2010%2021H1/source/repos/Kaggle%20script/INSTRUCTIONS/landmark_samples_30fps/frame_05.png) (Frame 5 @ 30 FPS - Hand Raising Phase)
- [frame_15.png](file:///c:/Users/Windows%2010%2021H1/source/repos/Kaggle%20script/INSTRUCTIONS/landmark_samples_30fps/frame_15.png) (Frame 15 @ 30 FPS - Mid-gesture Hand & Finger Articulation)
- [frame_30.png](file:///c:/Users/Windows%2010%2021H1/source/repos/Kaggle%20script/INSTRUCTIONS/landmark_samples_30fps/frame_30.png) (Frame 30 @ 30 FPS - Gesture Completion)
- [asl_sequence_30fps.gif](file:///c:/Users/Windows%2010%2021H1/source/repos/Kaggle%20script/INSTRUCTIONS/landmark_samples_30fps/asl_sequence_30fps.gif) (Full 30 FPS Animated GIF Sequence)

---

### Frame Keypoint Color Guide
- **Green Points / Skeleton**: Left Hand (21 MediaPipe Keypoints)
- **Cyan Points / Skeleton**: Right Hand (21 MediaPipe Keypoints)
- **Purple Lines**: Upper Body Pose Skeleton (Shoulders, Elbows, Wrists, Torso)
- **Blue Points**: Face Mesh Contour (Lips, Eyes, Eyebrows)
