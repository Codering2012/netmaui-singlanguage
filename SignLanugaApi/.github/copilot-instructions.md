# Copilot Instructions

## Project Guidelines
- Prefer a stateless web API architecture for ASL recognition: keep temporal smoothing and MediaPipe extraction on the client, and keep the C# API focused on fast ONNX inference only.