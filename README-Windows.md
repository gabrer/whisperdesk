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

- **One-folder build with SpeechBrain (default, recommended)**:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir
```

This includes SpeechBrain and PyTorch for best diarization quality (~500MB additional size).

- **One-folder build WITHOUT SpeechBrain (smaller)**:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir -ExcludeSpeechBrain
```

This uses only WeSpeaker ONNX for diarization (lighter, good quality).

- **Single EXE (one-file)**:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onefile
```

> **Note:** SpeechBrain is now **included by default** for high-quality speaker diarization. The build will be larger (~500MB) but provides the best results. If you select "SpeechBrain ECAPA" in the GUI and it's not bundled, the app will automatically fall back to WeSpeaker ONNX.

> **Tip:** You can also use `whisperdesk.spec` with `pyinstaller --noconfirm whisperdesk.spec`. Set `INCLUDE_SPEECHBRAIN=0` to exclude SpeechBrain and reduce build size.

## 5) Run

- Onedir build: run `dist/WhisperDesk/WhisperDesk.exe`
- Onefile build: run `dist/WhisperDesk.exe`

## 6) Offline notes

- If Whisper model folders are present under `models/`, transcription works offline.
- If `diarization_models/ecapa-voxceleb.onnx` is present, WeSpeaker ONNX diarization works offline.
- **SpeechBrain mode** (included by default): Downloads speaker embedding models on first use to `%LOCALAPPDATA%\WhisperDesk\hf-cache`. Once cached, works offline.
- If you excluded SpeechBrain during build, choose "WeSpeaker ONNX (light)" for diarization.

## 7) Troubleshooting

- If Windows SmartScreen warns, click More info → Run anyway.
- If antivirus quarantines the EXE, prefer the one-folder build.
- **SpeechBrain is now included by default** - no special steps needed for high-quality diarization.
- If audio resampling fails, install `scipy` (already in requirements.txt).

### SpeechBrain compatibility issues

**Symptoms:** Error when diarization starts:

```
hf_hub_download() got an unexpected keyword argument 'use_auth_token'
```

**Cause:** Older SpeechBrain versions (1.0.0) use deprecated `use_auth_token` parameter incompatible with newer `huggingface_hub`.

**Solution:** The code now includes an automatic runtime compatibility patch that handles this. However, for best results, upgrade your build environment:

```powershell
pip install --upgrade "speechbrain>=1.0.1" "huggingface_hub>=0.22.0"
pip install -r requirements.txt --upgrade
```

Then rebuild. **No Hugging Face token is required** for SpeechBrain models - they're public.

**Note:** If you're using an already-built executable and can't rebuild, the runtime patch should handle this automatically on next run.

### Model downloads hang or freeze

**Symptoms:** App freezes during model download, especially in bundled .exe builds.

**Cause:** Windows Defender (or other antivirus) real-time scanning locks files during download, causing indefinite hangs.

**Solution:** Add Windows Defender exclusion for the cache folder:

```powershell
# Run PowerShell as Administrator
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\WhisperDesk"
```

This tells Windows Defender to skip scanning the WhisperDesk cache folder, preventing file locks during downloads.

**Alternative:** Pre-download models on a different machine and copy them to `models/` folder before building the bundle.

### Model downloads and cache

- When a model isn't found under `models/`, WhisperDesk will download it to a per-user cache:
  - `%LOCALAPPDATA%\WhisperDesk\hf-cache` on Windows
  - `~/Library/Caches/WhisperDesk/hf-cache` on macOS
  - `~/.cache/WhisperDesk/hf-cache` on Linux
- This avoids permission issues and antivirus interference in bundled apps.
- Some Hugging Face repositories (for example `faster-whisper-large-v3-turbo`) require accepting a license before download. Generate an access token on huggingface.co and set it via `WHISPERDESK_HF_TOKEN` (or the standard `HUGGINGFACE_HUB_TOKEN`) so the Windows bundle can authenticate when fetching those models.
- Whisper v3 family (e.g., `large-v3`, `large-v3-turbo`, `distil-large-v3`) needs `faster-whisper` 1.0.5+ and `ctranslate2` 4.7+. Update your build venv (`pip install -U faster-whisper ctranslate2`) before running `build.ps1` so the packaged exe includes the newer 128-mel feature extractor.
- If network downloads are slow or blocked, you can pre-download models on a machine with internet and copy the resulting `models--<org>--<repo>` folder from the cache listed above.
- For maximum portability (offline), place CT2 model folders like `whisper-small-ct2` directly under `models/` before building.

### Whisper v3 shape mismatch (80 vs 128 mel features)

**Symptoms:** Error when using Whisper v3 models (large-v3, distil-large-v3, large-v3-turbo):

```
Invalid input features shape: expected an input with shape (1, 128, 3000), but got an input with shape (1, 80, 3000) instead
```

**Cause:** Whisper v3 models require 128 mel features. This error means the Windows build has older `faster-whisper` or `ctranslate2` wheels that extract only 80 features.

**Solution:** Upgrade dependencies in your build environment before building:

```powershell
python -m pip install --upgrade pip wheel
pip install --upgrade "faster-whisper>=1.1.0" "ctranslate2>=4.6.0"
pip install -r requirements.txt --upgrade
```

Then rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir
```

**Note:** Other Whisper models (base, small, medium, large-v2) use 80 mel features and work fine. This only affects v3 models.

## 8) Customization

- Default settings are in `presets.json`.
- Put favorite Whisper models under `models/`.
- Logs are written to the `logs/` folder next to the app.
