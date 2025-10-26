"""Lightweight diarization: VAD -> embeddings (ECAPA-ONNX) -> k-means (K<=3) -> assign to segments.

This module defers heavy imports so the application can import even if optional
dependencies are not installed. When diarization is disabled (max_speakers <= 1)
or dependencies are missing, it returns a single-speaker fallback.
"""
import logging
import os
from typing import List, Dict, Any, Tuple

from utils import diarization_root


def ensure_ecapa_model(progress_callback=None) -> bool:
    """Ensure the ECAPA ONNX model exists locally; download if missing.

    Uses huggingface_hub for robust downloads, falls back to urllib if needed.
    Returns True if available (already present or downloaded), False on failure.
    progress_callback: optional callable (msg: str, pct: int)
    """
    root = diarization_root()
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, 'ecapa-voxceleb.onnx')
    if os.path.isfile(target):
        return True

    # Try huggingface_hub first: download any .onnx from the repo snapshot and select one
    try:
        if progress_callback:
            progress_callback('Downloading diarization model…', 10)
        from huggingface_hub import snapshot_download
        snap_dir = snapshot_download(
            repo_id='speechbrain/spkrec-ecapa-voxceleb',
            revision='main',
            allow_patterns=['*.onnx', '**/*.onnx'],
            local_dir=root,
            local_dir_use_symlinks=False,
        )
        # Find any .onnx file in the snapshot dir
        candidates = []
        for dirpath, _, filenames in os.walk(snap_dir):
            for fn in filenames:
                if fn.lower().endswith('.onnx'):
                    candidates.append(os.path.join(dirpath, fn))
        if not candidates:
            raise FileNotFoundError('No .onnx file found in speechbrain/spkrec-ecapa-voxceleb')
        # Prefer a file containing 'embedding' in the name, else pick the first
        chosen = None
        for c in candidates:
            name = os.path.basename(c).lower()
            if 'embedding' in name or 'ecapa' in name:
                chosen = c
                break
        if not chosen:
            chosen = candidates[0]
        # Copy to expected target name
        try:
            import shutil as _shutil
            _shutil.copyfile(chosen, target)
        except Exception as _e:
            logging.warning('Failed to copy downloaded model (%s); using snapshot file directly.', _e)
            target = chosen
        if progress_callback:
            progress_callback('Diarization model ready.', 100)
        return True
    except Exception as e:
        logging.info('huggingface_hub download failed or not available: %s', e)

    # Fallback: plain HTTP download via urllib
    try:
        import urllib.request
        import shutil
        import tempfile
        if progress_callback:
            progress_callback('Downloading diarization model…', 20)
        url = 'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.onnx?download=true'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            total = resp.length or 0
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                read = 0
                chunk = 1024 * 64
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    tmp.write(buf)
                    read += len(buf)
                    if total and progress_callback:
                        pct = min(95, 20 + int(read * 80 / total))
                        progress_callback('Downloading diarization model…', pct)
                tmp_path = tmp.name
        shutil.move(tmp_path, target)
        if progress_callback:
            progress_callback('Diarization model ready.', 100)
        return True
    except Exception as e:
        logging.warning('Failed to download ECAPA ONNX model via HTTP: %s', e)
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        if progress_callback:
            progress_callback('Diarization model download failed; continuing without diarization.', 100)
        return False


class VADSegmenter:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_ms: int = 30):
        import webrtcvad  # defer import

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2  # 16-bit PCM
        self.frame_ms = frame_ms

    def segment_pcm16(self, pcm: bytes) -> List[Tuple[float, float]]:
        # Returns voiced segments as (start_sec, end_sec)
        voiced = []
        cur_start = None
        t = 0.0
        step = self.frame_ms / 1000.0
        for i in range(0, len(pcm), self.frame_bytes):
            frame = pcm[i:i+self.frame_bytes]
            if len(frame) < self.frame_bytes:
                break
            is_voiced = self.vad.is_speech(frame, self.sample_rate)
            if is_voiced and cur_start is None:
                cur_start = t
            elif not is_voiced and cur_start is not None:
                voiced.append((cur_start, t))
                cur_start = None
            t += step
        if cur_start is not None:
            voiced.append((cur_start, t))
        return voiced


def load_wav_mono16(wav_path: str):
    """Return (audio_np_int16, sample_rate). Imports numpy locally.
    
    Converts audio to mono 16-bit PCM at 16kHz for VAD compatibility.
    """
    import numpy as np
    try:
        import soundfile as sf
        # Use soundfile for robust loading and resampling
        data, sr = sf.read(wav_path, dtype='int16')
        
        # Convert stereo to mono if needed
        if len(data.shape) == 2:
            data = data.mean(axis=1).astype(np.int16)
        
        # Resample to 16kHz if needed (webrtcvad requires 8k, 16k, 32k, or 48k)
        if sr not in [8000, 16000, 32000, 48000]:
            logging.warning("Audio sample rate %d Hz not supported by VAD; resampling to 16kHz", sr)
            try:
                from scipy import signal
                target_sr = 16000
                num_samples = int(len(data) * target_sr / sr)
                data = signal.resample(data, num_samples).astype(np.int16)
                sr = target_sr
            except ImportError:
                logging.error("scipy not installed; cannot resample. Install with: pip install scipy")
                raise
        
        return data, sr
    except ImportError:
        # Fallback to wave module if soundfile not available
        import wave
        logging.warning("soundfile not installed; using wave module (less robust). Install with: pip install soundfile")
        with wave.open(wav_path, 'rb') as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            if w.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit audio, got {w.getsampwidth() * 8}-bit")
            pcm = w.readframes(w.getnframes())
        
        data = np.frombuffer(pcm, dtype=np.int16)
        if ch == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Check sample rate
        if sr not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate {sr} Hz not supported by VAD. Use 8k, 16k, 32k, or 48k Hz.")
        
        return data, sr


class ECAPAEmbedder:
    def __init__(self):
        import onnxruntime as ort  # defer import

        model_path = os.path.join(diarization_root(), 'ecapa-voxceleb.onnx')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Missing ECAPA ONNX model at {model_path}\n"
                "Download it from: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.onnx?download=true\n"
                "Save as: diarization_models/ecapa-voxceleb.onnx"
            )
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _mfcc(self, audio, _sr: int):
        # Minimal MFCC/log-mel features; replace with your preproc that matches ONNX export.
        # For skeleton, we use a simple log-mel filterbank with librosa-like steps, but without librosa.
        # This is a placeholder; in production align with the ECAPA pipeline used for export.
        import numpy as np
        import numpy.fft as fft
        win = 400
        hop = 160
        frames = []
        for i in range(0, len(audio)-win, hop):
            frame = audio[i:i+win] * np.hanning(win)
            spec = np.abs(fft.rfft(frame, n=512))
            frames.append(np.log1p(spec))
        feats = np.stack(frames, axis=0) if frames else np.zeros((1, 257), dtype=np.float32)
        return feats.astype(np.float32)

    def embed_segment(self, audio, sr: int, start: float, end: float):
        import numpy as np
        s = max(0, int(start * sr))
        e = min(len(audio), int(end * sr))
        frag = audio[s:e].astype(np.float32) / 32768.0
        feats = self._mfcc(frag, sr)
        inp = feats[None, ...]  # (1, T, F)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: inp})
        emb = outs[0].mean(axis=1).squeeze()
        return emb / (np.linalg.norm(emb) + 1e-10)


def diarize(wav_path: str, max_speakers: int = 3) -> List[Tuple[float, float, int]]:
    """Returns list of (start, end, speaker_id)."""
    logging.info("Diarizing: %s", wav_path)
    audio, sr = load_wav_mono16(wav_path)

    # If diarization is disabled or single speaker requested, return a single segment.
    if max_speakers <= 1:
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # Defer heavy deps; if missing, return single-speaker fallback with a warning.
    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning("Diarization dependencies missing (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # VAD
    try:
        vad = VADSegmenter(aggressiveness=2, sample_rate=sr, frame_ms=30)
        pcm = audio.tobytes()
        voiced = vad.segment_pcm16(pcm)
    except Exception as e:
        logging.warning("VAD processing failed (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    if not voiced:
        return [(0.0, len(audio)/sr, 0)]

    # Embeddings
    try:
        emb = ECAPAEmbedder()
        vecs = []
        for (s, e) in voiced:
            vecs.append(emb.embed_segment(audio, sr, s, e))
        X = np.stack(vecs, axis=0)
    except FileNotFoundError as e:
        logging.warning("ECAPA model not found; returning single-speaker. %s", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]
    except Exception as e:
        logging.warning("Embedding extraction failed (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # Choose K<=max_speakers via silhouette (fallback K=1)
    def best_k(X, kmax):
        if X.shape[0] < 2:
            return 1
        from sklearn.metrics import silhouette_score
        best_score, bestk = -1.0, 1
        for k in range(1, min(kmax, X.shape[0]) + 1):
            if k == 1:
                score = -1.0
            else:
                km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
                score = silhouette_score(X, km.labels_)
            if score > best_score:
                best_score, bestk = score, k
        return bestk

    K = best_k(X, max_speakers)
    if K == 1:
        labels = np.zeros((X.shape[0],), dtype=int)
    else:
        km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
        labels = km.labels_

    # Assign speakers to segments
    diarized = []
    for (seg, spk) in zip(voiced, labels):
        diarized.append((seg[0], seg[1], int(spk)))
    return diarized


def assign_speakers_to_asr(asr_segments: List[Dict[str, Any]], diarized_segments: List[Tuple[float, float, int]]):
    """For each ASR segment, pick the speaker whose diarized time overlaps most."""
    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    for seg in asr_segments:
        s0, s1 = seg["start"], seg["end"]
        best_id, best_ov = 0, -1.0
        for (d0, d1, spk) in diarized_segments:
            ov = overlap(s0, s1, d0, d1)
            if ov > best_ov:
                best_ov = ov
                best_id = spk
        seg["speaker"] = best_id
    return asr_segments
