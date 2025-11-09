<#
  WhisperDesk Windows builder
  - Requires: Python 3.11, venv with dependencies (pip install -r requirements.txt)
  - Optional: place Whisper CTranslate2 models under .\models\ (e.g., whisper-small-ct2) for offline build
  - Optional: place diarization ONNX under .\diarization_models\ecapa-voxceleb.onnx for offline diarization

  Usage examples:
    # One-folder build with SpeechBrain (default, recommended)
    powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onedir

    # One-file build
    powershell -ExecutionPolicy Bypass -File build.ps1 -Mode onefile

    # Build WITHOUT SpeechBrain (smaller, WeSpeaker ONNX only)
    powershell -ExecutionPolicy Bypass -File build.ps1 -ExcludeSpeechBrain
#>

param(
  [ValidateSet('onedir','onefile')]
  [string]$Mode = 'onedir',
  [string]$Name = 'WhisperDesk',
  # SpeechBrain now enabled by default for best diarization quality
  [switch]$IncludeSpeechBrain = $true,
  [switch]$ExcludeSpeechBrain,
  # When set, build with a visible console (errors will show in console)
  [switch]$Console
)

$ErrorActionPreference = 'Stop'

# Handle SpeechBrain flag (excluded if explicitly requested)
if ($ExcludeSpeechBrain) {
  $IncludeSpeechBrain = $false
}

Write-Host "Build Configuration:"
Write-Host "  Mode: $Mode"
Write-Host "  SpeechBrain: $(if ($IncludeSpeechBrain) { 'Enabled (high-quality diarization)' } else { 'Disabled (WeSpeaker ONNX only)' })"
Write-Host ""

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
  '--collect-all', 'soundfile',
  '--collect-all', 'huggingface_hub',
  '--collect-all', 'httpx',
  '--collect-all', 'certifi'
)

# SpeechBrain and PyTorch (enabled by default for best diarization)
if ($IncludeSpeechBrain) {
  Write-Host "Including SpeechBrain and PyTorch dependencies..."
  $collectArgs += @(
    '--collect-all', 'speechbrain',
    '--collect-all', 'torch',
    '--collect-all', 'torchaudio',
    '--collect-all', 'tqdm',
    '--collect-all', 'hyperpyyaml',
    '--collect-all', 'joblib',
    '--collect-all', 'sentencepiece'
  )
  # Hidden imports for SpeechBrain internal modules
  $collectArgs += @(
    '--hidden-import', 'speechbrain.inference.speaker',
    '--hidden-import', 'speechbrain.pretrained',
    '--hidden-import', 'speechbrain.dataio.dataio',
    '--hidden-import', 'speechbrain.dataio.dataset',
    '--hidden-import', 'speechbrain.dataio.encoder',
    '--hidden-import', 'speechbrain.processing.features',
    '--hidden-import', 'torch.nn.functional',
    '--hidden-import', 'torch.utils.data'
  )
}

pyinstaller --noconfirm `
  $modeSwitch `
  --name $Name `
  $(if ($Console) { '--console' } else { '--noconsole' }) `
  @dataArgs `
  @collectArgs `
  app.py

Write-Host "`nBuild complete. Output: dist/$Name*"
