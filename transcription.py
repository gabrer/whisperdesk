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
        # Resolve model path or remote id. Prefer fully local models (offline) when available.
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

        # Helper: check a directory contains expected CT2 files
        def _looks_like_ct2_dir(path: str) -> bool:
            required = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]
            try:
                return all(os.path.isfile(os.path.join(path, f)) for f in required)
            except Exception:
                return False

        # Try 1: explicit local ct2 directory (e.g., models/whisper-small-ct2)
        model_id: str
        if os.path.isdir(self.model_dir) and _looks_like_ct2_dir(self.model_dir):
            model_id = self.model_dir
            os.environ["HF_HUB_OFFLINE"] = "1"  # enforce offline usage
        else:
            # Try 2: locate HF snapshot in our models/ cache and use it offline
            remote_short = to_remote_name(model_name)  # e.g., 'small', 'large-v2'
            org = "Systran"
            repo = f"faster-whisper-{remote_short}"
            cache_dir_name = f"models--{org}--{repo}"
            cache_root = os.path.join(models_root(), cache_dir_name, "snapshots")
            local_snapshot_dir = None
            try:
                if os.path.isdir(cache_root):
                    # Pick the most recent snapshot folder that contains required files
                    candidates = [
                        os.path.join(cache_root, d) for d in os.listdir(cache_root)
                        if os.path.isdir(os.path.join(cache_root, d))
                    ]
                    # Sort by modification time, newest first
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    for cand in candidates:
                        if _looks_like_ct2_dir(cand):
                            local_snapshot_dir = cand
                            break
            except Exception:
                local_snapshot_dir = None

            if local_snapshot_dir:
                model_id = local_snapshot_dir
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                # Fallback to remote id (online)
                model_id = remote_short
                # allow online checks
                os.environ.pop("HF_HUB_OFFLINE", None)

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
            # If we resolved to a remote id (not a local directory), warn about long download
            if not os.path.isabs(model_id):
                progress_callback("Downloading model (this may take several minutes for large models)...", 10)

        # Force faster-whisper/HF Hub to download models to our local models/ folder
        # Set download_root to ensure models are stored in models/ instead of user cache
        download_root = models_root()
        os.makedirs(download_root, exist_ok=True)
        # Also direct Hugging Face caches to the same folder for consistency
        os.environ["HF_HOME"] = download_root
        os.environ["HUGGINGFACE_HUB_CACHE"] = download_root
        # Prefer accelerated transfer if available (no-op if package not installed)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        logging.info(
            "Model source: %s | local_dir_exists=%s | download_root=%s",
            model_id,
            os.path.isdir(self.model_dir),
            download_root,
        )

        logging.info("Loading model: %s (device=%s, compute_type=%s)", model_id, device, compute_type)

        # Track actual device/compute_type used after initialization
        self.active_device = device
        self.active_compute_type = compute_type

        # Use configurable number of workers for CTranslate2 (1 = safest, 2-4 = faster but may conflict)
        # Try multiple compute types with fallback for GPU compatibility
        compute_types_to_try = [compute_type]

        # Add fallbacks for GPU mode if float16 fails
        if device == "cuda" and compute_type == "float16":
            compute_types_to_try.extend(["int8_float32", "int8_float16", "int8"])

        last_error = None
        loaded = False
        if device == "cuda":
            for ct in compute_types_to_try:
                try:
                    logging.info("Attempting to load model with compute_type=%s", ct)
                    self.model = WhisperModel(
                        model_id,
                        device=device,
                        compute_type=ct,
                        num_workers=num_workers,
                        cpu_threads=1 if device == "cpu" else 4,
                        download_root=download_root,
                    )
                    if ct != compute_type:
                        logging.warning("Fell back to compute_type=%s (original %s not supported)", ct, compute_type)
                    loaded = True
                    self.active_device = device
                    self.active_compute_type = ct
                    break  # Success, exit loop
                except ValueError as e:
                    if "compute type" in str(e).lower():
                        logging.warning("compute_type=%s not supported on GPU, trying fallback...", ct)
                        last_error = e
                        continue
                    else:
                        # Different error on GPU, record and break to CPU fallback
                        last_error = e
                        break

            if not loaded:
                # Attempt CPU fallback
                cpu_ct = "int8" if (is_macos or is_windows) else "int8_float16"
                logging.warning(
                    "GPU initialization failed (%s). Falling back to CPU with compute_type=%s.",
                    str(last_error) if last_error else "unknown error",
                    cpu_ct,
                )
                if progress_callback:
                    progress_callback("GPU unavailable/unsupported; falling back to CPU...", 15)
                self.model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type=cpu_ct,
                    num_workers=num_workers,
                    cpu_threads=1,
                    download_root=download_root,
                )
                loaded = True
                logging.warning("CPU fallback succeeded (compute_type=%s)", cpu_ct)
                self.active_device = "cpu"
                self.active_compute_type = cpu_ct
        else:
            # Non-GPU path: just load with the chosen CPU compute type
            self.model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                num_workers=num_workers,
                cpu_threads=1,
                download_root=download_root,
            )
            self.active_device = device
            self.active_compute_type = compute_type

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
