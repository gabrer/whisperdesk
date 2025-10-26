import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict


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
    return os.path.join(app_root(), "models")


def diarization_root() -> str:
    return os.path.join(app_root(), "diarization_models")


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

    ch = logging.StreamHandler()
    ch.setLevel(level_obj)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

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
