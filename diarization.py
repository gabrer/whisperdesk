"""Lightweight diarization: VAD -> embeddings -> clustering -> assign to segments.

Supports two engines:
- 'speechbrain': High-quality ECAPA embeddings via SpeechBrain (requires torch)
- 'wespeaker': Lightweight ONNX embeddings (fallback, no torch needed)

This module defers heavy imports so the application can import even if optional
dependencies are not installed. When diarization is disabled (max_speakers <= 1)
or dependencies are missing, it returns a single-speaker fallback.
"""
import logging
import os
from typing import List, Dict, Any, Tuple, Optional

from utils import diarization_root, models_root


def ensure_ecapa_model(progress_callback=None) -> bool:
    """Ensure the speaker embedding ONNX model exists locally; download if missing.

    Uses WeSpeaker ResNet34 from hbredin/wespeaker-voxceleb-resnet34-LM.
    Returns True if available (already present or downloaded), False on failure.
    progress_callback: optional callable (msg: str, pct: int)
    """
    root = diarization_root()
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, 'ecapa-voxceleb.onnx')
    if os.path.isfile(target):
        return True

    # Helper: basic validation to catch HTML/partial downloads
    def _looks_like_onnx(path: str) -> bool:
        try:
            sz = os.path.getsize(path)
            if sz < 100_000:  # very small -> likely HTML error page
                return False
            with open(path, 'rb') as f:
                head = f.read(64)
            # crude check for HTML
            if head.strip().lower().startswith(b'<!doctype') or head.strip().lower().startswith(b'<html'):
                return False
            return True
        except Exception:
            return False

    # Try huggingface_hub first for clean download
    try:
        if progress_callback:
            progress_callback('Downloading diarization model…', 10)
        from huggingface_hub import hf_hub_download
        dl_path = hf_hub_download(
            repo_id='hbredin/wespeaker-voxceleb-resnet34-LM',
            filename='speaker-embedding.onnx',
            revision='main',
        )
        # Copy to our expected location
        import shutil as _shutil
        _shutil.copyfile(dl_path, target)
        if not _looks_like_onnx(target):
            raise RuntimeError('Downloaded file does not look like a valid ONNX model')
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
        url = 'https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx?download=true'
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
        if not _looks_like_onnx(target):
            raise RuntimeError('HTTP-downloaded file is not a valid ONNX (possibly an HTML page).')
        if progress_callback:
            progress_callback('Diarization model ready.', 100)
        return True
    except Exception as e:
        logging.warning('Failed to download speaker embedding model via HTTP: %s', e)
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
                f"Missing speaker embedding ONNX model at {model_path}\n"
                "Download it from: https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/resolve/main/speaker-embedding.onnx\n"
                "Save as: diarization_models/ecapa-voxceleb.onnx"
            )
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _mfcc(self, audio, _sr: int):
        """Compute 80-dim log-Mel filterbank features (fbank) expected by WeSpeaker models.
        Window: 25 ms (400 samples at 16k), hop: 10 ms (160 samples), n_fft=512, n_mels=80.
        """
        import numpy as np
        import numpy.fft as fft

        n_fft = 512
        n_mels = 80
        win = int(0.025 * _sr)  # 25 ms
        hop = int(0.010 * _sr)  # 10 ms
        if win < 1:
            win = 400
        if hop < 1:
            hop = 160

        # Precompute Mel filterbank
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0**(mel / 2595.0) - 1.0)

        fmin = 0.0
        fmax = _sr / 2.0
        mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num=n_mels + 2)
        hz = mel_to_hz(mels)
        bins = np.floor((n_fft + 1) * hz / _sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m0, f_m1, f_m2 = bins[m - 1], bins[m], bins[m + 1]
            f_m0 = max(f_m0, 0)
            f_m2 = min(f_m2, n_fft // 2)
            if f_m1 > f_m0:
                fb[m - 1, f_m0:f_m1] = (np.arange(f_m0, f_m1) - f_m0) / max(1, (f_m1 - f_m0))
            if f_m2 > f_m1:
                fb[m - 1, f_m1:f_m2] = (f_m2 - np.arange(f_m1, f_m2)) / max(1, (f_m2 - f_m1))

        # Framing and STFT power spectrum
        frames = []
        hann = np.hanning(win).astype(np.float32)
        for i in range(0, max(0, len(audio) - win + 1), hop):
            frame = audio[i:i + win].astype(np.float32) * hann
            spec = np.abs(fft.rfft(frame, n=n_fft))**2  # power spectrum
            mel = fb @ spec[: (n_fft // 2 + 1)]
            mel = np.maximum(mel, 1e-10)
            frames.append(np.log(mel))

        feats = np.stack(frames, axis=0) if frames else np.zeros((1, n_mels), dtype=np.float32)
        return feats.astype(np.float32)

    def embed_segment(self, audio, sr: int, start: float, end: float):
        import numpy as np
        s = max(0, int(start * sr))
        e = min(len(audio), int(end * sr))
        frag = audio[s:e].astype(np.float32) / 32768.0

        # Check for silent/empty segments
        if len(frag) < 100 or np.abs(frag).max() < 1e-6:
            # Return a small random embedding for silent segments
            return np.random.randn(192).astype(np.float32) * 0.01

        feats = self._mfcc(frag, sr)
        feats = np.nan_to_num(feats, copy=False, nan=0.0, posinf=1e5, neginf=-1e5)
        inp = feats[None, ...]  # (1, T, 80)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: inp})
        arr = outs[0]
        # Robustly collapse to a 1-D embedding vector
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            emb_vec = arr
        elif arr.ndim == 2:
            # assume (T, D) or (D, T); pick time mean to get (D,)
            # choose axis with larger size as time dimension
            time_axis = 0 if arr.shape[0] >= arr.shape[1] else 1
            emb_vec = arr.mean(axis=time_axis)
        else:
            # flatten time dimensions, keep last as feature dim
            emb_vec = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        emb_vec = np.nan_to_num(emb_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if embedding is zero vector
        norm = np.linalg.norm(emb_vec)
        if norm < 1e-8:
            # Return small random embedding instead of zero
            return np.random.randn(len(emb_vec)).astype(np.float32) * 0.01

        return emb_vec / norm


def diarize(wav_path: str, max_speakers: int = 3, engine: str = 'speechbrain') -> List[Tuple[float, float, int]]:
    """Returns list of (start, end, speaker_id).

    Args:
        wav_path: path to WAV file
        max_speakers: maximum number of speakers (1-3)
        engine: 'speechbrain' (high quality) or 'wespeaker' (lightweight ONNX fallback)
    """
    logging.info("Diarizing: %s (engine=%s)", wav_path, engine)
    audio, sr = load_wav_mono16(wav_path)

    # If diarization is disabled or single speaker requested, return a single segment.
    if max_speakers <= 1:
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # Defer heavy deps; if missing, return single-speaker fallback with a warning.
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as e:
        logging.warning("NumPy missing (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # VAD to find voiced segments
    try:
        vad = VADSegmenter(aggressiveness=2, sample_rate=sr, frame_ms=30)
        pcm = audio.tobytes()
        voiced = vad.segment_pcm16(pcm)
    except Exception as e:
        logging.warning("VAD processing failed (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    if not voiced:
        return [(0.0, len(audio)/sr, 0)]

    # Try SpeechBrain engine first if requested
    if engine == 'speechbrain':
        try:
            embeddings = _extract_embeddings_speechbrain(audio, sr, voiced)
        except Exception as e:
            # Check if this is a frozen build without SpeechBrain bundled
            error_msg = str(e)
            if "cannot find the path" in error_msg.lower() and ("speechbrain" in error_msg.lower() or "_internal" in error_msg.lower()):
                logging.info("SpeechBrain not bundled in this build; using WeSpeaker ONNX fallback")
            else:
                logging.warning("SpeechBrain embedding failed (%s); falling back to WeSpeaker ONNX.", e)
            embeddings = None
    else:
        embeddings = None

    # Fallback to WeSpeaker ONNX if SpeechBrain unavailable or failed
    if embeddings is None:
        try:
            embeddings = _extract_embeddings_wespeaker(audio, sr, voiced)
        except Exception as e:
            logging.warning("WeSpeaker embedding failed (%s); returning single-speaker.", e)
            return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # Cluster embeddings
    try:
        labels = _cluster_embeddings(embeddings, max_speakers)
    except Exception as e:
        logging.warning("Clustering failed (%s); returning single-speaker.", e)
        return [(0.0, len(audio)/sr if sr else 0.0, 0)]

    # Assign speakers to segments
    diarized = []
    for (seg, spk) in zip(voiced, labels):
        diarized.append((seg[0], seg[1], int(spk)))

    # Apply smoothing to reduce spurious speaker changes
    diarized = _smooth_speaker_labels(diarized, min_duration=0.3, median_window=3)

    return diarized


def _extract_embeddings_speechbrain(audio, sr: int, voiced_segments: List[Tuple[float, float]]):
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN.

    Returns: np.ndarray of shape (n_segments, embedding_dim)
    """
    import numpy as np
    import torch

    # CRITICAL: Apply compatibility patch BEFORE importing SpeechBrain
    # SpeechBrain 1.0.0 uses 'use_auth_token', newer huggingface_hub expects 'token'
    # Must patch before SpeechBrain imports huggingface_hub
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download
        import inspect

        # Check if we need the patch (token param exists but use_auth_token doesn't)
        sig = inspect.signature(hf_hub_download)
        needs_patch = 'token' in sig.parameters and 'use_auth_token' not in sig.parameters

        if needs_patch:
            # Store the original hf_hub_download function
            _original_hf_hub_download = hf_hub_download

            def _patched_hf_hub_download(*args, **kwargs):
                """Wrapper that translates use_auth_token -> token for backward compatibility."""
                # If use_auth_token is passed, rename it to token
                if 'use_auth_token' in kwargs:
                    kwargs['token'] = kwargs.pop('use_auth_token')
                return _original_hf_hub_download(*args, **kwargs)

            # Patch at module level - must happen before SpeechBrain import
            huggingface_hub.hf_hub_download = _patched_hf_hub_download
            # Also replace in sys.modules to catch all references
            import sys
            if 'huggingface_hub' in sys.modules:
                sys.modules['huggingface_hub'].hf_hub_download = _patched_hf_hub_download

            logging.info("[Diarization] Applied huggingface_hub compatibility patch (use_auth_token->token)")
    except Exception as e:
        # Patch failed, might not be needed or already compatible
        logging.debug("[Diarization] Compatibility patch not applied: %s", e)

    # NOW import SpeechBrain - it will see the patched hf_hub_download
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        raise ImportError(f"Failed to import SpeechBrain: {e}")

    # Determine savedir - check multiple locations
    # 1. Try models/speechbrain_ecapa (standard location)
    # 2. Try models/models--speechbrain--spkrec-ecapa-voxceleb (HF cache format)
    # 3. Use temp directory as fallback
    savedir_candidates = [
        os.path.join(models_root(), "speechbrain_ecapa"),
        os.path.join(models_root(), "models--speechbrain--spkrec-ecapa-voxceleb", "snapshots"),
    ]

    savedir = None
    for candidate in savedir_candidates:
        if os.path.isdir(candidate):
            # For HF snapshot directories, find the latest snapshot
            if "snapshots" in candidate:
                try:
                    snapshots = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
                    if snapshots:
                        # Use the first (and typically only) snapshot
                        savedir = os.path.join(candidate, snapshots[0])
                        logging.info("[Diarization] Found SpeechBrain model in HF cache: %s", savedir)
                        break
                except Exception as e:
                    logging.debug("[Diarization] Failed to check HF snapshots: %s", e)
                    continue
            else:
                savedir = candidate
                logging.info("[Diarization] Found SpeechBrain model at: %s", savedir)
                break

    # If no existing model found, use default location and let SpeechBrain download
    if savedir is None:
        savedir = os.path.join(models_root(), "speechbrain_ecapa")
        logging.info("[Diarization] No existing SpeechBrain model found, will download to: %s", savedir)
        model_exists = False
    else:
        # Check if model files actually exist
        model_exists = all(
            os.path.exists(os.path.join(savedir, f))
            for f in ["hyperparams.yaml", "embedding_model.ckpt"]
        )
        if model_exists:
            logging.info("[Diarization] Model files verified in: %s", savedir)
        else:
            logging.info("[Diarization] Model directory exists but files incomplete, will download")

    # Load pretrained ECAPA model
    try:
        if model_exists:
            # Load from local files only (offline mode)
            logging.info("[Diarization] Loading SpeechBrain model from local files (offline)")
            classifier = EncoderClassifier.from_hparams(
                source=savedir,  # Use local path as source for offline loading
                savedir=savedir
            )
        else:
            # Download from HuggingFace Hub
            logging.info("[Diarization] Downloading SpeechBrain model from HuggingFace Hub")
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=savedir
            )
    except Exception as e:
        logging.error("[Diarization] Failed to load SpeechBrain model from %s: %s", savedir, e)
        raise

    embeddings = []
    for idx, (start, end) in enumerate(voiced_segments):
        try:
            s = max(0, int(start * sr))
            e = min(len(audio), int(end * sr))
            segment = audio[s:e].astype(np.float32) / 32768.0

            # Skip silent/empty segments
            if len(segment) < 100 or np.abs(segment).max() < 1e-6:
                # Use a small random embedding for silent segments
                emb_np = np.random.randn(192).astype(np.float32) * 0.01
            else:
                # SpeechBrain expects torch tensor
                waveform = torch.from_numpy(segment).unsqueeze(0)  # (1, samples)

                with torch.no_grad():
                    emb = classifier.encode_batch(waveform)
                    emb_np = emb.squeeze().cpu().numpy()

            embeddings.append(emb_np)
        except Exception as e:
            # If segment embedding fails, use random embedding to avoid breaking entire diarization
            logging.warning("[Diarization] Failed to extract embedding for segment %d (%.2f-%.2f): %s. Using random fallback.",
                          idx, start, end, e)
            emb_np = np.random.randn(192).astype(np.float32) * 0.01
            embeddings.append(emb_np)

    if not embeddings:
        raise RuntimeError("No embeddings extracted from any voiced segment")

    return np.stack(embeddings, axis=0)


def _extract_embeddings_wespeaker(audio, sr: int, voiced_segments: List[Tuple[float, float]]):
    """Extract speaker embeddings using WeSpeaker ONNX model (fallback).

    Returns: np.ndarray of shape (n_segments, embedding_dim)
    """
    import numpy as np

    emb = ECAPAEmbedder()
    vecs = []
    for (s, e) in voiced_segments:
        vecs.append(emb.embed_segment(audio, sr, s, e))

    return np.stack(vecs, axis=0)


def _cluster_embeddings(embeddings, max_speakers: int):
    """Cluster embeddings using Agglomerative Clustering with cosine affinity.

    Returns: np.ndarray of speaker labels for each segment
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n_samples = embeddings.shape[0]

    if n_samples < 2:
        return np.zeros((n_samples,), dtype=int)

    # Normalize embeddings for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Check for zero vectors that would cause clustering to fail
    zero_mask = norms.squeeze() < 1e-8
    if zero_mask.all():
        # All embeddings are zero - return single speaker
        logging.warning("All embeddings are zero vectors; returning single-speaker.")
        return np.zeros((n_samples,), dtype=int)

    # Replace zero vectors with small random noise to prevent cosine errors
    embeddings_safe = embeddings.copy()
    if zero_mask.any():
        logging.warning(f"Found {zero_mask.sum()} zero embedding vectors; replacing with noise.")
        for i in np.where(zero_mask)[0]:
            embeddings_safe[i] = np.random.randn(embeddings.shape[1]).astype(np.float32) * 0.01

    # Re-normalize after fixing zero vectors
    norms = np.linalg.norm(embeddings_safe, axis=1, keepdims=True)
    embeddings_norm = embeddings_safe / (norms + 1e-10)

    # Find optimal number of clusters using silhouette score
    best_score = -1.0
    best_k = 1

    for k in range(1, min(max_speakers, n_samples) + 1):
        if k == 1:
            continue

        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='cosine',
            linkage='average'
        ).fit(embeddings_norm)

        score = silhouette_score(embeddings_norm, clustering.labels_, metric='cosine')

        if score > best_score:
            best_score = score
            best_k = k

    if best_k == 1:
        return np.zeros((n_samples,), dtype=int)

    # Final clustering with best k
    clustering = AgglomerativeClustering(
        n_clusters=best_k,
        metric='cosine',
        linkage='average'
    ).fit(embeddings_norm)

    return clustering.labels_


def _smooth_speaker_labels(segments: List[Tuple[float, float, int]],
                           min_duration: float = 0.3,
                           median_window: int = 3) -> List[Tuple[float, float, int]]:
    """Apply temporal smoothing to speaker labels.

    Args:
        segments: list of (start, end, speaker_id)
        min_duration: minimum segment duration; merge shorter segments
        median_window: window size for median filtering

    Returns: smoothed list of (start, end, speaker_id)
    """
    if not segments or len(segments) < 2:
        return segments

    import numpy as np

    # Step 1: Merge very short segments into neighbors
    merged = []
    for i, (start, end, spk) in enumerate(segments):
        duration = end - start
        if duration < min_duration and merged:
            # Merge into previous segment
            prev_start, prev_end, prev_spk = merged[-1]
            merged[-1] = (prev_start, end, prev_spk)
        else:
            merged.append((start, end, spk))

    if len(merged) < median_window:
        return merged

    # Step 2: Apply median filter to speaker labels
    labels = np.array([spk for (_, _, spk) in merged])
    smoothed_labels = labels.copy()

    half_window = median_window // 2
    for i in range(len(labels)):
        window_start = max(0, i - half_window)
        window_end = min(len(labels), i + half_window + 1)
        window_labels = labels[window_start:window_end]
        # Use mode (most common label) instead of median for speaker IDs
        unique, counts = np.unique(window_labels, return_counts=True)
        smoothed_labels[i] = unique[np.argmax(counts)]

    # Reconstruct segments with smoothed labels
    result = []
    for (start, end, _), new_spk in zip(merged, smoothed_labels):
        result.append((start, end, int(new_spk)))

    # Step 3: Merge consecutive segments with same speaker
    if not result:
        return result

    final = [result[0]]
    for (start, end, spk) in result[1:]:
        prev_start, prev_end, prev_spk = final[-1]
        if spk == prev_spk:
            # Merge with previous
            final[-1] = (prev_start, end, spk)
        else:
            final.append((start, end, spk))

    return final


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
