# Requires: Python 3.11, venv created, dependencies installed per requirements.txt
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1

$ErrorActionPreference = "Stop"

# Clean
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path build) { Remove-Item build -Recurse -Force }

# PyInstaller
pyinstaller --noconfirm `
  --onefile `
  --name WhisperDesk `
  --noconsole `
  --add-data "models;models" `
  --add-data "diarization_models;diarization_models" `
  --add-data "presets.json;." `
  app.py

Write-Host "\nBuild complete. Output: dist/WhisperDesk.exe"
