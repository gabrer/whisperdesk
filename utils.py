import hashlib
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict
import platform
from functools import lru_cache


# Paths
def app_root() -> str:
    """Absolute path to the application root.

    Handles PyInstaller onefile by using sys._MEIPASS when present.
    """
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        return str(Path(bundle_dir).resolve())
    # When running from source
    return str(Path(__file__).resolve().parent)


def models_root() -> str:
    return _resolve_resource_dir("models", "WHISPERDESK_MODELS_DIR")


def diarization_root() -> str:
    return _resolve_resource_dir("diarization_models", "WHISPERDESK_DIARIZATION_DIR")


def hf_cache_root() -> str:
    r"""Return a per-user, writable cache directory for Hugging Face models.

    Rationale:
    - In frozen apps (e.g., PyInstaller on Windows), writing into the app folder can fail
      or be blocked by AV/scanners. Using a user cache dir avoids permission issues.
    - Mirrors platform conventions:
      - Windows: %LOCALAPPDATA%\\WhisperDesk\\hf-cache
      - macOS: ~/Library/Caches/WhisperDesk/hf-cache
      - Linux: ~/.cache/WhisperDesk/hf-cache
    """
    system = platform.system()
    try:
        if system == "Windows":
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
            logging.info("[FileSystem] Windows detected - LOCALAPPDATA=%s, base=%s",
                        os.environ.get("LOCALAPPDATA", "NOT_SET"), base)
            cache_dir = os.path.join(base, "WhisperDesk", "hf-cache")
        elif system == "Darwin":  # macOS
            base = os.path.expanduser("~/Library/Caches")
            logging.info("[FileSystem] macOS detected - base=%s", base)
            cache_dir = os.path.join(base, "WhisperDesk", "hf-cache")
        else:  # Linux and others
            base = os.path.expanduser("~/.cache")
            logging.info("[FileSystem] Linux/other detected - base=%s", base)
            cache_dir = os.path.join(base, "WhisperDesk", "hf-cache")

        logging.info("[FileSystem] Computed HF cache directory: %s", cache_dir)
        logging.info("[FileSystem] Cache directory exists: %s", os.path.exists(cache_dir))

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logging.info("[FileSystem] Created/verified cache directory: %s", cache_dir)
        logging.info("[FileSystem] Cache directory is writable: %s", os.access(cache_dir, os.W_OK))
        logging.info("[FileSystem] Cache directory is readable: %s", os.access(cache_dir, os.R_OK))

        return cache_dir
    except Exception as e:
        # Fallback to app-local cache if anything goes wrong
        logging.error("[FileSystem] Failed to create user cache directory: %s", str(e), exc_info=True)
        fallback = os.path.join(app_root(), "hf-cache")
        logging.warning("[FileSystem] Falling back to app-local cache: %s", fallback)
        try:
            Path(fallback).mkdir(parents=True, exist_ok=True)
            logging.info("[FileSystem] Created fallback cache directory: %s", fallback)
            logging.info("[FileSystem] Fallback cache is writable: %s", os.access(fallback, os.W_OK))
        except Exception as e2:
            logging.error("[FileSystem] Failed to create fallback cache: %s", str(e2), exc_info=True)
        return fallback


# Logging
def setup_logging(level: Any = logging.INFO) -> str:
    """Configure root logging and return the log file path.

    Accepts either an int level or a string like "INFO".
    Logs to console and to logs/whisperdesk.log
    """
    if isinstance(level, str):
        level_obj = getattr(logging, level.upper(), logging.INFO)
    else:
        level_obj = int(level)

    logger = logging.getLogger()
    logger.setLevel(level_obj)

    # Ensure logs directory
    logs_dir = os.path.join(app_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "whisperdesk.log")

    # Clear existing handlers to avoid duplicate logs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Console handlers: INFO to stdout, WARN+ to stderr
    ch_out = logging.StreamHandler(stream=sys.stdout)
    ch_out.setLevel(min(level_obj, logging.INFO))
    ch_out.setFormatter(fmt)
    logger.addHandler(ch_out)

    ch_err = logging.StreamHandler(stream=sys.stderr)
    ch_err.setLevel(logging.WARNING)
    ch_err.setFormatter(fmt)
    logger.addHandler(ch_err)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level_obj)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_path


# JSON helpers
def load_json(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(default) if default is not None else {}


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Misc helpers
def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 hash of a file efficiently.

    Note: For integrity/tamper checks only; not for security-sensitive uses.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def format_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS (hours omitted if zero)."""
    total = int(round(float(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{m:02d}:{s:02d}" if h == 0 else f"{h}:{m:02d}:{s:02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def _resolve_resource_dir(subdir: str, env_var: str) -> str:
    """Locate bundled resources in both source runs and frozen builds."""
    candidates: list[Path] = []

    # Explicit override via environment variable (primarily for testing/support)
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    # Standard locations relative to the import root
    base_dir = Path(app_root())
    candidates.append(base_dir / subdir)

    try:
        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / subdir)
    except Exception:
        pass

    # When running from a PyInstaller bundle, data may live next to the executable
    if getattr(sys, "frozen", False):
        try:
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / subdir)
        except Exception:
            pass

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            meipass_path = Path(meipass)
            candidates.append(meipass_path / subdir)
            candidates.append(meipass_path.parent / subdir)

    # Fallback to current working directory to aid local testing
    candidates.append(Path.cwd() / subdir)

    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.exists():
                return key
        except OSError:
            continue

    # Nothing found; default to app_root/subdir so the caller knows where to place files.
    return str(base_dir / subdir)
