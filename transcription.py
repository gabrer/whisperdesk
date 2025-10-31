import logging
import os
import platform
from typing import List, Dict, Any, Optional, TypedDict, Callable

# ...existing code...
# Remove the top-level import to avoid failing during module import
# from faster_whisper import WhisperModel
from utils import models_root

class Word(TypedDict):
    start: float
    end: float
    word: str


class Segment(TypedDict, total=False):
    start: float
    end: float
    text: str
    words: Optional[List[Word]]


__all__ = ["Transcriber", "Segment", "Word"]

class Transcriber:
    def __init__(self, model_name: str, device_mode: str = "auto", language_hint: str = "en", word_timestamps: bool = False, num_workers: int = 1, progress_callback: Optional[Callable[[str, int], None]] = None):
        # Import here so app import succeeds even if dependency is missing
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "faster-whisper is not installed or failed to import. "
                "Install dev deps: pip install -r requirements.txt"
            ) from e
        # Resolve model path or remote id. If local dir is missing, fall back to remote name
        # so faster-whisper can download the model from Hugging Face on first use.
        self.model_dir = os.path.join(models_root(), model_name)

        def to_remote_name(dir_like: str) -> str:
            # Map folder-like names to faster-whisper ids
            mapping = {
                "whisper-tiny-ct2": "tiny",
                "whisper-base-ct2": "base",
                "whisper-small-ct2": "small",
                "whisper-medium-ct2": "medium",
                # Large family
                "whisper-large-v2-ct2": "large-v2",
                "whisper-large-v3-ct2": "large-v3",
                "whisper-large-v3-turbo-ct2": "large-v3-turbo",
                # English-only
                "whisper-small-en-ct2": "small.en",
                "whisper-medium-en-ct2": "medium.en",
                # Distilled
                "whisper-distil-large-v3-ct2": "distil-large-v3",
            }
            return mapping.get(dir_like, dir_like)

        model_id = self.model_dir if os.path.isdir(self.model_dir) else to_remote_name(model_name)

        # Determine device and compute type
        is_macos = platform.system() == "Darwin"
        is_windows = platform.system() == "Windows"

        # Check if CUDA is available for GPU acceleration
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            # If torch is not available or fails, assume no CUDA
            pass

        if device_mode == "gpu":
            device = "cuda"
            compute_type = "float16"
        elif device_mode == "cpu":
            device = "cpu"
            # Use int8 on macOS/Windows for better compatibility
            compute_type = "int8" if (is_macos or is_windows) else "int8_float16"
        else:  # auto
            if cuda_available:
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                # Use int8 on macOS/Windows CPU for better compatibility
                compute_type = "int8" if (is_macos or is_windows) else "int8_float16"

        # Report progress if callback provided
        if progress_callback:
            progress_callback("Initializing model...", 0)
            if not os.path.isdir(self.model_dir):
                progress_callback("Downloading model (this may take a few minutes)...", 10)

        # Force faster-whisper/HF Hub to download models to our local models/ folder
        # Set download_root to ensure models are stored in models/ instead of user cache
        download_root = models_root()
        os.makedirs(download_root, exist_ok=True)
        # Also direct Hugging Face caches to the same folder for consistency
        os.environ["HF_HOME"] = download_root
        os.environ["HUGGINGFACE_HUB_CACHE"] = download_root
        logging.info(
            "Model source: %s | local_dir_exists=%s | download_root=%s",
            model_id,
            os.path.isdir(self.model_dir),
            download_root,
        )

        logging.info("Loading model: %s (device=%s, compute_type=%s)", model_id, device, compute_type)
        # Use configurable number of workers for CTranslate2 (1 = safest, 2-4 = faster but may conflict)
        self.model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
            cpu_threads=1 if device == "cpu" else 4,
            download_root=download_root,
        )

        if progress_callback:
            progress_callback("Model loaded.", 100)

        self.language = language_hint
        self.word_timestamps = word_timestamps

    def transcribe(self, wav_path: str) -> Dict[str, Any]:
        # ...existing code...
        logging.info("Transcribing: %s", wav_path)
        # Disable VAD to avoid onnxruntime crashes on some platforms
        segments, info = self.model.transcribe(
            wav_path,
            language=self.language,
            word_timestamps=self.word_timestamps,
            vad_filter=False
        )
        out_segments: List[Segment] = []
        for seg in segments:
            out_segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": (
                    [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in getattr(seg, 'words', [])]
                    if self.word_timestamps and getattr(seg, 'words', None) else None
                )
            })
        return {
            "language": getattr(info, 'language', None),
            "duration": float(getattr(info, 'duration', 0.0)),
            "segments": out_segments
        }
