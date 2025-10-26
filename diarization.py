"""Lightweight diarization: VAD -> embeddings (ECAPA-ONNX) -> k-means (K<=3) -> assign to segments.

This module defers heavy imports so the application can import even if optional
dependencies are not installed. When diarization is disabled (max_speakers <= 1)
or dependencies are missing, it returns a single-speaker fallback.
"""
import logging
import os
from typing import List, Dict, Any, Tuple

from utils import diarization_root


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
    """Return (audio_np_int16, sample_rate). Imports numpy locally."""
    import numpy as np
    import wave
    with wave.open(wav_path, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        assert w.getsampwidth() == 2, "Expect 16-bit WAV"
        pcm = w.readframes(w.getnframes())
    # no need to parse struct manually; numpy handles the buffer
    data = np.frombuffer(pcm, dtype=np.int16)
    if ch == 2:
        data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return data, sr


class ECAPAEmbedder:
    def __init__(self):
        import onnxruntime as ort  # defer import

        model_path = os.path.join(diarization_root(), 'ecapa-voxceleb.onnx')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing ECAPA ONNX model at {model_path}")
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
    vad = VADSegmenter(aggressiveness=2, sample_rate=sr, frame_ms=30)
    pcm = audio.tobytes()
    voiced = vad.segment_pcm16(pcm)
    if not voiced:
        return [(0.0, len(audio)/sr, 0)]

    # Embeddings
    emb = ECAPAEmbedder()
    vecs = []
    for (s, e) in voiced:
        vecs.append(emb.embed_segment(audio, sr, s, e))
    X = np.stack(vecs, axis=0)

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
