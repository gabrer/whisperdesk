Place diarization model assets here.

## Required model (not included):

Download the ECAPA-TDNN speaker embedding model:

**Option 1: Direct download**
```bash
cd diarization_models
curl -L -o ecapa-voxceleb.onnx https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.onnx
```

**Option 2: Manual download**
1. Visit: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/main
2. Download `embedding_model.onnx`
3. Rename to `ecapa-voxceleb.onnx`
4. Place in this folder (`diarization_models/`)

## Config file:
- `vad_config.json` — Voice Activity Detection settings (already included)

## Note:
If the ECAPA model is missing, diarization will fall back to single-speaker mode automatically.
