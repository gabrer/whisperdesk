<#
  WhisperDesk Windows builder
  - Requires: Python 3.11, venv with dependencies (pip install -r requirements.txt)
  - Optional: place Whisper CTranslate2 models under .\models\ (e.g., whisper-small-ct2) for offline build
  - Optional: place diarization ONNX under .\diarization_models\ecapa-voxceleb.onnx for offline diarization

  Usage examples:
    # One-folder build (recommended for offline portability)
    powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir

    # One-file build (single exe)
    powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onefile

    # Include SpeechBrain (torch+torchaudio) for high-quality diarization
    powershell -ExecutionPolicy Bypass -File build.ps1 -IncludeSpeechBrain
#>

param(
  [ValidateSet('onedir','onefile')]
  [string]$Mode = 'onedir',
  [string]$Name = 'WhisperDesk',
  [switch]$IncludeSpeechBrain,
  # When set, build with a visible console (errors will show in console)
  [switch]$Console
)

$ErrorActionPreference = 'Stop'

# Clean
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path build) { Remove-Item build -Recurse -Force }

$modeSwitch = if ($Mode -eq 'onefile') { '--onefile' } else { '--onedir' }

# Required data files
$dataArgs = @(
  '--add-data', 'presets.json;.'
)
if (Test-Path 'models') { $dataArgs += @('--add-data', 'models;models') }
if (Test-Path 'diarization_models') { $dataArgs += @('--add-data', 'diarization_models;diarization_models') }

# Package collection (ensure runtime resources are included)
$collectArgs = @(
  '--collect-all', 'faster_whisper',
  '--collect-all', 'ctranslate2',
  '--collect-all', 'onnxruntime',
  '--collect-all', 'sklearn',
  '--collect-all', 'soundfile'
)
if ($IncludeSpeechBrain) {
  $collectArgs += @('--collect-all', 'speechbrain', '--collect-all', 'torch', '--collect-all', 'torchaudio')
}

pyinstaller --noconfirm `
  $modeSwitch `
  --name $Name `
  $(if ($Console) { '--console' } else { '--noconsole' }) `
  @dataArgs `
  @collectArgs `
  app.py

Write-Host "`nBuild complete. Output: dist/$Name*"
