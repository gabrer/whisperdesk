#!/usr/bin/env python3
"""Download SpeechBrain ECAPA weights for offline builds."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from huggingface_hub import snapshot_download

REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
REQUIRED_FILES: Sequence[str] = (
    "hyperparams.yaml",
    "classifier.ckpt",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
    "label_encoder.ckpt",
    "custom.py",
)


def has_required_files(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, fname)) for fname in REQUIRED_FILES)


def ensure_speechbrain(target_dir: str, force: bool = False) -> None:
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    if not force and has_required_files(target_dir):
        logging.info("SpeechBrain weights already present at %s", target_dir)
        return

    logging.info("Downloading SpeechBrain ECAPA model (%s) to %s", REPO_ID, target_dir)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        cache_dir=None,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(REQUIRED_FILES),
    )
    logging.info("SpeechBrain weights downloaded successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=os.path.join("models", "speechbrain_ecapa"),
        help="Destination directory for the SpeechBrain checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    try:
        ensure_speechbrain(args.target, args.force)
    except Exception as exc:  # pragma: no cover - build-time helper
        logging.error("Failed to download SpeechBrain weights: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
