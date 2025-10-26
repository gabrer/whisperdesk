# WhisperDesk (Offline)

**Portable, offline transcription for Windows 11** using Whisper (faster-whisper/CT2), optional CUDA acceleration, CPU fallback, lightweight diarization (up to 3 speakers), batch processing, and TXT/DOCX output with segment timestamps.

## End users (no Python needed)
1. Unzip the package.
2. Ensure the folder contains:
   - `WhisperDesk.exe`
   - `models/whisper-*-ct2/` (at least one model)
   - `diarization_models/ecapa-voxceleb.onnx`
3. Double-click `WhisperDesk.exe`.
4. Drag-and-drop `.wav` files or choose a folder. Click **Transcribe**.

Desktop GUI for offline transcription using Whisper (CTranslate2) with optional speaker diarization.

End users only need the compiled WhisperDesk.exe plus the `models/` and `diarization_models/` folders. Developers run the Python source and can build the .exe via `build.ps1` (Windows).

## Features

- PySide6 GUI for selecting audio/video files and viewing transcripts
- faster-whisper transcription using local CT2 models if present
- Optional diarization pipeline (stubbed initially)
- Export to TXT and DOCX

## Project layout

```
WhisperDesk/
├─ app.py                  # GUI entry point (PySide6)
├─ transcription.py        # Whisper (faster-whisper) pipeline
├─ diarization.py          # VAD + ECAPA (ONNX) + clustering (stub initially)
├─ exporters.py            # TXT and DOCX exporters
├─ device.py               # Device/RAM/VRAM detection
├─ settings.py             # Settings model, load/save (presets.json)
├─ utils.py                # Helpers: hashing, time formatting, logging
├─ requirements.txt        # Dev-only; for running/building from source
├─ build.ps1               # PyInstaller build for Windows
├─ presets.json            # Default app config
├─ models/                 # ct2 Whisper models: tiny/base/small/medium/large-v3
├─ diarization_models/     # ecapa-voxceleb.onnx (not included) + VAD config
└─ logs/                   # Created at runtime
```

## Quick start (developer)

1) Create a virtual environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Place Whisper CT2 models (optional but recommended for offline)

- Create folders like `models/whisper-small-ct2/` and put the converted CTranslate2 model files there.
- If not present, faster-whisper may download models by name (internet required).

3) (Optional) Place diarization model

- Put `ecapa-voxceleb.onnx` in `diarization_models/`.
- Adjust `diarization_models/vad_config.json` if needed.

4) Run the app

```bash
python app.py
```

## Build Windows .exe (developer)

On Windows with Python and the venv activated:

```powershell
./build.ps1 -Clean
```

Outputs to `dist/WhisperDesk/WhisperDesk.exe` and bundles `models/`, `diarization_models/`, and `presets.json`.

## License

MIT — see `LICENSE`.


## Developers (to build the .exe)
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File build.ps1