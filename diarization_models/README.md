# Diarization Models (Speaker Embeddings)

The app uses speaker embeddings for diarization (identifying who speaks when).

## Auto-download
The model downloads automatically when you enable diarization (set Max speakers > 1 in the UI).

## Manual download (if auto-download fails)

**WeSpeaker ResNet34 (trained on VoxCeleb):**
```bash
cd diarization_models
curl -L -o ecapa-voxceleb.onnx \
	https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx
```

**Model details:**
- **Source:** hbredin/wespeaker-voxceleb-resnet34-LM
- **Format:** ONNX (CPU-compatible)
- **File:** speaker-embedding.onnx → saved as ecapa-voxceleb.onnx

## Fallback behavior
If the model is missing or diarization dependencies aren't installed, the app falls back to single-speaker mode automatically.
