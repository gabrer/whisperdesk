# WhisperDesk on Windows 11 (Portable/Offline)

This guide shows how to build a portable Windows app, optionally fully offline.

## 1) Prerequisites

- Windows 11
- Python 3.11 x64
- PowerShell
- (Optional, recommended) Virtual environment

## 2) Setup

```powershell
# From project root
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to build fully offline later, finish all downloads now while online.

## 3) Prepare offline models (optional but recommended)

To avoid network access at runtime, place models in these folders before building:

- Whisper CTranslate2 models
  - Put folders like `whisper-small-ct2`, `whisper-large-v3-ct2`, etc. under `models/`
  - Example structure:
    ```
    models/
      whisper-small-ct2/
        config.json
        model.bin
        ...
    ```
- Diarization ONNX (WeSpeaker ResNet34)
  - Save as `diarization_models/ecapa-voxceleb.onnx`

If models are missing at runtime, the app can download them (internet required).

## 4) Build (no installer)

Use the PowerShell script to create a portable app under `dist/`.

- One-folder build (recommended for portability and antivirus friendliness):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir
```

- Single EXE (one-file):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onefile
```

- Include SpeechBrain (torch+torchaudio) for high-quality diarization (bigger build):

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -IncludeSpeechBrain
```

> Tip: You can also use `whisperdesk.spec` with `pyinstaller --noconfirm whisperdesk.spec`. Set `INCLUDE_SPEECHBRAIN=1` in the environment to bundle SpeechBrain.

## 5) Run

- Onedir build: run `dist/WhisperDesk/WhisperDesk.exe`
- Onefile build: run `dist/WhisperDesk.exe`

## 6) Offline notes

- If Whisper model folders are present under `models/`, transcription works offline.
- If `diarization_models/ecapa-voxceleb.onnx` is present, ONNX diarization works offline.
- SpeechBrain mode requires torch+torchaudio. If not bundled, choose "WeSpeaker ONNX (light)" or disable diarization.

## 7) Troubleshooting

- If Windows SmartScreen warns, click More info → Run anyway.
- If antivirus quarantines the EXE, prefer the one-folder build.
- If torchaudio import errors occur in SpeechBrain mode, rebuild with `-IncludeSpeechBrain`.
- If audio resampling fails, install `scipy` (already in requirements.txt).

## 8) Customization

- Default settings are in `presets.json`.
- Put favorite Whisper models under `models/`.
- Logs are written to the `logs/` folder next to the app.
