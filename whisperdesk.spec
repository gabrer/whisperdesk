# -*- mode: python ; coding: utf-8 -*-

# Optional PyInstaller spec for advanced builds. The PowerShell build script can be used instead.
# To use: pyinstaller --noconfirm whisperdesk.spec

import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect resources for key packages
hiddenimports = []
datas, binaries, hiddenimports_fw = collect_all('faster_whisper')
datas += collect_all('ctranslate2')[0]
datas += collect_all('onnxruntime')[0]
datas += collect_all('sklearn')[0]
datas += collect_all('soundfile')[0]
binaries += collect_all('requests')[1]
datas += collect_all('urllib3')[0]
datas += collect_all('certifi')[0]
hiddenimports += hiddenimports_fw
hiddenimports += ['requests', 'urllib3', 'charset_normalizer']

# SpeechBrain: Enable by default for best diarization quality (set INCLUDE_SPEECHBRAIN=0 to disable)
include_speechbrain = os.environ.get('INCLUDE_SPEECHBRAIN', '1') == '1'
if include_speechbrain:
    print("Including SpeechBrain and PyTorch for high-quality diarization...")
    d2, b2, h2 = collect_all('speechbrain')
    datas += d2
    binaries += b2
    hiddenimports += h2

    d3, b3, h3 = collect_all('torch')
    datas += d3
    binaries += b3
    hiddenimports += h3

    d4, b4, h4 = collect_all('torchaudio')
    datas += d4
    binaries += b4
    hiddenimports += h4

    # Additional dependencies for SpeechBrain
    for pkg in ['tqdm', 'hyperpyyaml', 'joblib', 'sentencepiece', 'yaml', 'ruamel.yaml']:
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception:
            pass

    # Hidden imports for SpeechBrain internal modules and dependencies
    hiddenimports += [
        'speechbrain.inference.speaker',
        'speechbrain.pretrained',
        'speechbrain.dataio.dataio',
        'speechbrain.dataio.dataset',
        'speechbrain.dataio.encoder',
        'speechbrain.processing.features',
        'speechbrain.utils.fetching',
        'speechbrain.utils.data_utils',
        'torch.nn.functional',
        'torch.utils.data',
        'requests',
        'urllib3',
        'charset_normalizer',
    ]
else:
    print("Building without SpeechBrain (WeSpeaker ONNX only for diarization)")

# Include app data
if os.path.exists('presets.json'):
    datas.append(('presets.json', '.'))
if os.path.isdir('models'):
    datas.append(('models', 'models'))
if os.path.isdir('diarization_models'):
    datas.append(('diarization_models', 'diarization_models'))

a = Analysis([
    'app.py',
],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='WhisperDesk',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='WhisperDesk')
