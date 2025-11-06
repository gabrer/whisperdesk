import logging
import os
import platform
from typing import List, Dict, Any, Optional, TypedDict, Callable, Tuple

# ...existing code...
# Remove the top-level import to avoid failing during module import
# from faster_whisper import WhisperModel
from utils import models_root, hf_cache_root

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

        def repo_candidates_for_remote(remote_short: str) -> List[str]:
            """Return ordered repo_id candidates (org/repo) for a remote model name."""
            overrides: Dict[str, List[str]] = {
                # Newer large v3 turbo builds are hosted under mobiuslabsgmbh/dropbox mirrors.
                "large-v3-turbo": [
                    "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                    "dropbox-dash/faster-whisper-large-v3-turbo",
                ],
            }
            base_repo = f"Systran/faster-whisper-{remote_short}"
            candidates = overrides.get(remote_short, []).copy()
            if base_repo not in candidates:
                candidates.append(base_repo)
            return candidates

        def split_repo_id(repo_id: str) -> Tuple[str, str]:
            if "/" not in repo_id:
                raise ValueError(f"Invalid repo_id '{repo_id}' (expected 'org/name')")
            org, repo = repo_id.split("/", 1)
            return org, repo

        def cache_dir_name_for_repo(repo_id: str) -> str:
            org, repo = split_repo_id(repo_id)
            return f"models--{org}--{repo}"

        # Helper: check a directory contains expected CT2 files
        def _looks_like_ct2_dir(path: str) -> bool:
            """CT2 Whisper dir is valid if it has config.json, model.bin, tokenizer.json, and a vocabulary file."""
            try:
                logging.debug("[FileSystem] Checking if %s looks like CT2 dir", path)
                has_config = os.path.isfile(os.path.join(path, "config.json"))
                has_model = os.path.isfile(os.path.join(path, "model.bin"))
                has_tokenizer = os.path.isfile(os.path.join(path, "tokenizer.json"))
                has_vocab_txt = os.path.isfile(os.path.join(path, "vocabulary.txt"))
                has_vocab_json = os.path.isfile(os.path.join(path, "vocabulary.json"))
                has_vocab = has_vocab_txt or has_vocab_json
                result = has_config and has_model and has_tokenizer and has_vocab
                logging.debug(
                    "[FileSystem] %s is CT2 dir: %s (config=%s, model=%s, tokenizer=%s, vocab_txt=%s, vocab_json=%s)",
                    path, result, has_config, has_model, has_tokenizer, has_vocab_txt, has_vocab_json
                )
                return result
            except Exception as e:
                logging.warning("[FileSystem] Error checking CT2 dir %s: %s", path, str(e))
                return False

        # Resolve model path or remote id. Prefer fully local models (offline) when available.
        self.model_dir = os.path.join(models_root(), model_name)
        remote_short = to_remote_name(model_name)
        repo_candidates = repo_candidates_for_remote(remote_short)
        download_repo_id: List[str] = [repo_candidates[0]]  # Mutable for closures
        selected_repo_id: Optional[str] = None

        # Check if this is a Whisper v3 model (requires 128 mel features)
        is_v3_model = remote_short in {"large-v3", "distil-large-v3", "large-v3-turbo"}
        if is_v3_model:
            try:
                from faster_whisper import __version__ as fw_ver  # type: ignore
                import ctranslate2  # type: ignore
                ct2_ver = getattr(ctranslate2, "__version__", "0.0.0")
                logging.info("[ModelInit] Whisper v3 model detected. Versions: faster-whisper=%s, ctranslate2=%s", fw_ver, ct2_ver)
                # Simple version check: ensure we have at least 1.0 and 4.6
                if fw_ver.split('.')[0] == '0' or (ct2_ver.split('.')[0] == '4' and int(ct2_ver.split('.')[1]) < 6):
                    raise RuntimeError(
                        f"Whisper v3 models require faster-whisper>=1.0.0 and ctranslate2>=4.6.0 for 128 mel features. "
                        f"Current: faster-whisper={fw_ver}, ctranslate2={ct2_ver}. "
                        f"Please upgrade: pip install --upgrade 'faster-whisper>=1.1.0' 'ctranslate2>=4.6.0'"
                    )
            except ImportError:
                pass  # Already handled by earlier import check

        # Try 1: explicit local ct2 directory (e.g., models/whisper-small-ct2)
        model_id: str
        logging.info("[ModelInit] Checking model_dir: %s", self.model_dir)
        logging.info("[FileSystem] model_dir exists: %s", os.path.exists(self.model_dir))
        logging.info("[FileSystem] model_dir is directory: %s", os.path.isdir(self.model_dir))

        if os.path.isdir(self.model_dir) and _looks_like_ct2_dir(self.model_dir):
            model_id = self.model_dir
            logging.info("[ModelInit] Using local CT2 directory: %s", model_id)
            os.environ["HF_HUB_OFFLINE"] = "1"  # enforce offline usage
        else:
            logging.info("[ModelInit] Local CT2 directory not found, searching caches")
            # Try 2: locate HF snapshot in our models/ cache OR user HF cache and use it offline
            local_snapshot_dir = None
            try:
                for repo_id in repo_candidates:
                    download_repo_id[0] = repo_id
                    org, repo = split_repo_id(repo_id)
                    cache_dir_name = cache_dir_name_for_repo(repo_id)
                    candidate_roots = [
                        os.path.join(models_root(), cache_dir_name, "snapshots"),
                        os.path.join(hf_cache_root(), cache_dir_name, "snapshots"),
                    ]
                    logging.info("[ModelInit] Searching for cached snapshots for %s in: %s", repo_id, candidate_roots)

                    for cache_root in candidate_roots:
                        logging.info("[FileSystem] Checking cache root: %s", cache_root)
                        logging.info("[FileSystem] Cache root exists: %s", os.path.exists(cache_root))
                        logging.info("[FileSystem] Cache root is directory: %s", os.path.isdir(cache_root))

                        if os.path.isdir(cache_root):
                            # Pick the most recent snapshot folder that contains required files
                            try:
                                dir_contents = os.listdir(cache_root)
                                logging.info("[FileSystem] Cache root contains %d items: %s",
                                           len(dir_contents), dir_contents[:10])  # Log first 10 items
                            except Exception as e:
                                logging.error("[FileSystem] Failed to list cache_root %s: %s", cache_root, str(e))
                                continue

                            candidates = [
                                os.path.join(cache_root, d) for d in dir_contents
                                if os.path.isdir(os.path.join(cache_root, d))
                            ]
                            logging.info("[FileSystem] Found %d snapshot directories", len(candidates))

                            # Sort by modification time, newest first
                            try:
                                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                                logging.debug("[FileSystem] Sorted candidates by mtime")
                            except Exception as e:
                                logging.warning("[FileSystem] Failed to sort by mtime: %s", str(e))

                            for cand in candidates:
                                logging.debug("[FileSystem] Checking candidate: %s", cand)
                                if _looks_like_ct2_dir(cand):
                                    local_snapshot_dir = cand
                                    selected_repo_id = repo_id
                                    logging.info("[ModelInit] Found valid snapshot for %s: %s", repo_id, cand)
                                    break
                        if local_snapshot_dir:
                            break
                    if local_snapshot_dir:
                        break
            except Exception as e:
                logging.error("[ModelInit] Error searching for cached snapshots: %s", str(e), exc_info=True)
                local_snapshot_dir = None

            if local_snapshot_dir:
                model_id = local_snapshot_dir
                if selected_repo_id is None:
                    selected_repo_id = download_repo_id[0]
                else:
                    download_repo_id[0] = selected_repo_id
                logging.info("[ModelInit] Using cached snapshot: %s", model_id)
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                logging.info("[ModelInit] No cached snapshot found, will download from HuggingFace")
                # Fallback to remote id (online) — but prefetch via huggingface_hub to avoid internal hangs
                # Use snapshot_download into our per-user cache with symlinks disabled
                try:
                    if progress_callback:
                        progress_callback("Preparing model download…", 0)
                    logging.info("[Trace] BEFORE import snapshot_download")
                    from huggingface_hub import snapshot_download  # type: ignore
                    logging.info("[Trace] AFTER import snapshot_download")
                    cache_dir = hf_cache_root()
                    logging.info("[Download] Starting snapshot_download to cache_dir: %s", cache_dir)
                    logging.info("[Download] repo candidates: %s", repo_candidates)
                    logging.info("[Download] local_dir_use_symlinks: False (Windows compatibility)")
                    logging.info("[Download] max_workers: 1 (reduce file contention)")
                    hf_token = (
                        os.environ.get("WHISPERDESK_HF_TOKEN")
                        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                        or os.environ.get("HF_TOKEN")
                    )

                    # Clean up incomplete downloads to avoid lock files and stale state
                    import shutil
                    for repo_id in repo_candidates:
                        repo_cache_dir = os.path.join(cache_dir, cache_dir_name_for_repo(repo_id))
                        snapshot_dir = os.path.join(repo_cache_dir, "snapshots")
                        if os.path.isdir(snapshot_dir):
                            logging.info("[Download] Checking for incomplete snapshots in: %s", snapshot_dir)
                            try:
                                for snap in os.listdir(snapshot_dir):
                                    snap_path = os.path.join(snapshot_dir, snap)
                                    if os.path.isdir(snap_path):
                                        if not _looks_like_ct2_dir(snap_path):
                                            logging.warning("[Download] Removing incomplete snapshot: %s", snap_path)
                                            try:
                                                shutil.rmtree(snap_path, ignore_errors=True)
                                                logging.info("[Download] Removed incomplete snapshot successfully")
                                            except Exception as rm_err:
                                                logging.error("[Download] Failed to remove incomplete snapshot: %s", str(rm_err))
                                        else:
                                            logging.info("[Download] Snapshot %s appears complete, keeping", snap)
                            except Exception as cleanup_err:
                                logging.warning("[Download] Error during cleanup: %s", str(cleanup_err))

                    # Start a lightweight monitor on Windows to log cache growth during download
                    monitor_stop = None
                    monitor_thread = None
                    try:
                        if platform.system() == "Windows":
                            import threading, time
                            monitor_stop = threading.Event()
                            def _monitor_cache_growth():
                                last_bytes = -1
                                last_path = None
                                while not monitor_stop.is_set():
                                    total = 0
                                    newest_path = None
                                    newest_mtime = -1.0
                                    try:
                                        current_repo_dir = os.path.join(cache_dir, cache_dir_name_for_repo(download_repo_id[0]))
                                        if os.path.isdir(current_repo_dir):
                                            for dirpath, _, filenames in os.walk(current_repo_dir):
                                                for fn in filenames:
                                                    fp = os.path.join(dirpath, fn)
                                                    try:
                                                        sz = os.path.getsize(fp)
                                                        total += sz
                                                        mt = os.path.getmtime(fp)
                                                        if mt > newest_mtime:
                                                            newest_mtime = mt
                                                            newest_path = fp
                                                    except Exception:
                                                        pass
                                    except Exception:
                                        pass
                                    if total != last_bytes:
                                        current_repo_dir = os.path.join(cache_dir, cache_dir_name_for_repo(download_repo_id[0]))
                                        if newest_path and newest_path != last_path:
                                            try:
                                                logging.info("[Download] Cache size under %s: %.2f MB (latest file: %s size=%.2f MB)",
                                                             current_repo_dir, total / (1024*1024), newest_path, (os.path.getsize(newest_path) / (1024*1024)))
                                            except Exception:
                                                logging.info("[Download] Cache size under %s: %.2f MB (latest file: %s)",
                                                             current_repo_dir, total / (1024*1024), newest_path)
                                        else:
                                            logging.info("[Download] Cache size under %s: %.2f MB", current_repo_dir, total / (1024*1024))
                                        last_bytes = total
                                        last_path = newest_path
                                    monitor_stop.wait(15.0)

                            monitor_thread = threading.Thread(target=_monitor_cache_growth, name="DLMonitor", daemon=True)
                            monitor_thread.start()
                    except Exception as mon_err:
                        logging.debug("[Download] Monitor not started: %s", str(mon_err))

                    # Heartbeat: log while inside snapshot_download to pinpoint stalls
                    hb_stop = None
                    hb_thread = None
                    try:
                        import threading, time
                        hb_stop = threading.Event()

                        def _heartbeat():
                            start_t = time.time()
                            ix = 0
                            while not hb_stop.is_set():
                                elapsed = int(time.time() - start_t)
                                logging.info("[Trace] Inside snapshot_download... %ds elapsed (tick %d)", elapsed, ix)
                                ix += 1
                                hb_stop.wait(5.0)

                        hb_thread = threading.Thread(target=_heartbeat, name="DLHeartbeat", daemon=True)
                        hb_thread.start()
                    except Exception as _:
                        pass

                    logging.info("[Trace] ENTER snapshot_download call")

                    # Use custom downloader on Windows for better control and diagnostics
                    # Falls back to snapshot_download on other platforms
                    snapshot_path: Optional[str] = None
                    download_success = False

                    try:
                        if platform.system() == "Windows":
                            logging.info("[Download] Using manual file-by-file downloader on Windows")
                            try:
                                # Prefer requests if available; otherwise fall back to httpx
                                try:
                                    import requests  # type: ignore
                                    http_impl = "requests"
                                except Exception:
                                    import httpx  # type: ignore
                                    http_impl = "httpx"

                                from huggingface_hub import hf_hub_url  # type: ignore
                                import certifi  # type: ignore

                                if hf_token:
                                    logging.info("[Download] Using Hugging Face token from environment for authenticated downloads")
                                headers = {"Authorization": f"Bearer {hf_token.strip()}"} if hf_token else {}
                                ca_bundle = certifi.where()

                                class ManualDownloadAuthError(Exception):
                                    """Raised when repository access requires authentication."""

                                def _manual_download_single(repo_id: str) -> str:
                                    download_repo_id[0] = repo_id
                                    org, repo = split_repo_id(repo_id)
                                    logging.info("[Download] Manual download targeting repo: %s", repo_id)

                                    repo_cache_dir = os.path.join(cache_dir, cache_dir_name_for_repo(repo_id))
                                    os.makedirs(repo_cache_dir, exist_ok=True)

                                    # Determine revision hash for the repo
                                    api_url = f"https://huggingface.co/api/models/{repo_id}/revision/main"
                                    logging.info("[Download] Fetching revision info from: %s (via %s)", api_url, http_impl)
                                    if http_impl == "requests":
                                        rev_resp = requests.get(api_url, timeout=30, verify=ca_bundle, headers=headers)  # type: ignore
                                        if rev_resp.status_code in (401, 403):
                                            raise ManualDownloadAuthError(
                                                "Authentication required to access this model. Set HUGGINGFACE_HUB_TOKEN or WHISPERDESK_HF_TOKEN."
                                            )
                                        rev_resp.raise_for_status()
                                        rev_json = rev_resp.json()
                                    else:
                                        import httpx  # type: ignore
                                        with httpx.Client(verify=ca_bundle, timeout=30, follow_redirects=True) as client:  # type: ignore
                                            rev_resp = client.get(api_url, headers=headers, follow_redirects=True)
                                            if rev_resp.status_code in (401, 403):
                                                raise ManualDownloadAuthError(
                                                    "Authentication required to access this model. Set HUGGINGFACE_HUB_TOKEN or WHISPERDESK_HF_TOKEN."
                                                )
                                            rev_resp.raise_for_status()
                                            rev_json = rev_resp.json()

                                    revision_hash = rev_json.get("sha", "main")
                                    logging.info("[Download] Resolved revision hash: %s", revision_hash)

                                    # Query file list to decide which artifacts exist
                                    files_api = f"https://huggingface.co/api/models/{repo_id}?expand=files"
                                    try:
                                        if http_impl == "requests":
                                            files_resp = requests.get(files_api, timeout=30, verify=ca_bundle, headers=headers)  # type: ignore
                                            files_resp.raise_for_status()
                                            files_json = files_resp.json()
                                        else:
                                            import httpx  # type: ignore
                                            with httpx.Client(verify=ca_bundle, timeout=30, follow_redirects=True) as client:  # type: ignore
                                                files_resp = client.get(files_api, headers=headers, follow_redirects=True)
                                                files_resp.raise_for_status()
                                                files_json = files_resp.json()
                                        siblings = files_json.get("siblings", []) or files_json.get("files", [])
                                        available_names = {s.get("rfilename") or s.get("path") or s.get("filename") for s in siblings}
                                        logging.info("[Download] Available repository files: %s", sorted([n for n in available_names if n]))
                                    except Exception as e:
                                        logging.warning("[Download] Could not fetch file list for %s: %s (will attempt defaults)", repo_id, str(e))
                                        available_names = None

                                    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", revision_hash)
                                    os.makedirs(snapshot_dir, exist_ok=True)
                                    logging.info("[Download] Prepared snapshot directory: %s", snapshot_dir)

                                    refs_dir = os.path.join(repo_cache_dir, "refs")
                                    os.makedirs(refs_dir, exist_ok=True)
                                    main_ref = os.path.join(refs_dir, "main")
                                    with open(main_ref, "w", encoding="utf-8") as ref_fp:
                                        ref_fp.write(revision_hash)
                                    logging.info("[Download] Updated refs/main to: %s", revision_hash)

                                    desired = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "vocabulary.json"]
                                    if available_names:
                                        files_to_download = [f for f in desired if f in available_names]
                                        missing = [f for f in desired if f not in available_names]
                                        if missing:
                                            logging.info("[Download] Skipping missing files (not in repo): %s", missing)
                                    else:
                                        files_to_download = desired
                                    for filename in files_to_download:
                                        file_path = os.path.join(snapshot_dir, filename)

                                        if os.path.isfile(file_path):
                                            file_size = os.path.getsize(file_path)
                                            logging.info("[Download] File %s already exists (%.2f MB), skipping", filename, file_size / (1024 * 1024))
                                            continue

                                        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision_hash)
                                        logging.info("[Download] Downloading %s from %s via %s", filename, url, http_impl)

                                        if http_impl == "requests":
                                            with requests.get(url, stream=True, timeout=60, verify=ca_bundle, headers=headers) as r:  # type: ignore
                                                if r.status_code in (401, 403):
                                                    raise ManualDownloadAuthError(
                                                        "Authentication required to download model artifacts. Set HUGGINGFACE_HUB_TOKEN or WHISPERDESK_HF_TOKEN."
                                                    )
                                                if r.status_code == 404:
                                                    logging.warning("[Download] %s not found in repo %s, skipping", filename, repo_id)
                                                    continue
                                                r.raise_for_status()
                                                total_size = int(r.headers.get("content-length", 0))
                                                logging.info("[Download] %s size: %.2f MB", filename, total_size / (1024 * 1024))
                                                downloaded = 0
                                                with open(file_path, "wb") as f:
                                                    for chunk in r.iter_content(chunk_size=8192):
                                                        if chunk:
                                                            f.write(chunk)
                                                            downloaded += len(chunk)
                                                            if downloaded % (1024 * 1024) == 0:
                                                                pct = (downloaded / total_size * 100) if total_size else 0
                                                                logging.info("[Download] %s: %.2f MB / %.2f MB (%.1f%%)",
                                                                             filename, downloaded / (1024 * 1024),
                                                                             total_size / (1024 * 1024), pct)
                                        else:
                                            import httpx  # type: ignore
                                            with httpx.Client(verify=ca_bundle, timeout=60, follow_redirects=True) as client:  # type: ignore
                                                with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
                                                    if r.status_code in (401, 403):
                                                        raise ManualDownloadAuthError(
                                                            "Authentication required to download model artifacts. Set HUGGINGFACE_HUB_TOKEN or WHISPERDESK_HF_TOKEN."
                                                        )
                                                    if r.status_code == 404:
                                                        logging.warning("[Download] %s not found in repo %s, skipping", filename, repo_id)
                                                        continue
                                                    r.raise_for_status()
                                                    total_size = int(r.headers.get("content-length", 0))
                                                    logging.info("[Download] %s size: %.2f MB", filename, total_size / (1024 * 1024))
                                                    downloaded = 0
                                                    with open(file_path, "wb") as f:
                                                        for chunk in r.iter_bytes():
                                                            if chunk:
                                                                f.write(chunk)
                                                                downloaded += len(chunk)
                                                                if downloaded % (1024 * 1024) == 0:
                                                                    pct = (downloaded / total_size * 100) if total_size else 0
                                                                    logging.info("[Download] %s: %.2f MB / %.2f MB (%.1f%%)",
                                                                                 filename, downloaded / (1024 * 1024),
                                                                                 total_size / (1024 * 1024), pct)

                                        final_size = os.path.getsize(file_path)
                                        logging.info("[Download] Completed %s: %.2f MB written", filename, final_size / (1024 * 1024))

                                    return snapshot_dir

                                manual_errors: List[str] = []
                                for repo_id in repo_candidates:
                                    try:
                                        snapshot_dir = _manual_download_single(repo_id)
                                        snapshot_path = snapshot_dir
                                        download_success = True
                                        selected_repo_id = repo_id
                                        logging.info("[Download] Manual download completed successfully for %s", repo_id)
                                        break
                                    except ManualDownloadAuthError as auth_err:
                                        logging.warning("[Download] Manual download requires authentication for %s: %s", repo_id, str(auth_err))
                                        manual_errors.append(f"{repo_id}: {auth_err}")
                                    except Exception as manual_err:
                                        logging.error("[Download] Manual download failed for %s: %s", repo_id, str(manual_err), exc_info=True)
                                        manual_errors.append(f"{repo_id}: {manual_err}")

                                if not download_success and manual_errors:
                                    logging.error("[Download] Manual download failed for all repo candidates: %s", "; ".join(manual_errors))

                            except Exception as manual_err:
                                logging.error("[Download] Manual download encountered an unexpected error: %s", str(manual_err), exc_info=True)
                                snapshot_path = None
                                download_success = False

                        # Fallback to snapshot_download if manual download not attempted or failed
                        if not download_success:
                            logging.info("[Download] Using snapshot_download (non-Windows or manual failed)")
                            try:
                                # Wrap in thread with timeout to prevent indefinite hangs
                                import threading
                                result_container: List[Optional[str]] = [None]
                                error_container: List[Optional[Exception]] = [None]
                                snapshot_errors: List[str] = []

                                def _download_with_snapshot(repo_id: str) -> None:
                                    try:
                                        result_container[0] = snapshot_download(
                                            repo_id=repo_id,
                                            cache_dir=cache_dir,
                                            local_files_only=False,
                                            max_workers=1,
                                            local_dir_use_symlinks=False,
                                            revision="main",
                                            resume_download=True,
                                            token=hf_token,
                                        )
                                    except Exception as e:
                                        error_container[0] = e

                                for repo_id in repo_candidates:
                                    download_repo_id[0] = repo_id
                                    selected_repo_id = repo_id
                                    result_container[0] = None
                                    error_container[0] = None
                                    download_thread = threading.Thread(target=_download_with_snapshot, args=(repo_id,), name="SnapshotDL")
                                    download_thread.start()
                                    download_thread.join(timeout=120)  # 2 minute timeout

                                    if download_thread.is_alive():
                                        logging.error("[Download] snapshot_download timed out after 120s for %s", repo_id)
                                        snapshot_errors.append(f"{repo_id}: timeout")
                                        continue

                                    if error_container[0] is not None:
                                        logging.error("[Download] snapshot_download failed for %s: %s", repo_id, str(error_container[0]), exc_info=True)
                                        snapshot_errors.append(f"{repo_id}: {error_container[0]}")
                                        continue

                                    snapshot_path = result_container[0]
                                    if snapshot_path:
                                        logging.info("[Download] snapshot_download completed for %s: %s", repo_id, snapshot_path)
                                        download_success = True
                                        break

                                if not download_success and snapshot_errors:
                                    raise RuntimeError(f"snapshot_download failed for all candidates: {'; '.join(snapshot_errors)}")

                            except Exception as snap_err:
                                logging.error("[Download] snapshot_download failed: %s", str(snap_err), exc_info=True)
                                raise
                    finally:
                        logging.info("[Trace] RETURN snapshot_download call")
                        # Stop monitor
                        try:
                            if monitor_stop is not None:
                                monitor_stop.set()
                            if monitor_thread is not None:
                                monitor_thread.join(timeout=5)
                        except Exception:
                            pass
                        # Stop heartbeat
                        try:
                            if hb_stop is not None:
                                hb_stop.set()
                            if hb_thread is not None:
                                hb_thread.join(timeout=5)
                        except Exception:
                            pass

                    if snapshot_path:
                        logging.info("[Download] snapshot_download returned path: %s", snapshot_path)
                        logging.info("[FileSystem] Snapshot path exists: %s", os.path.exists(snapshot_path))
                        logging.info("[FileSystem] Snapshot path is directory: %s", os.path.isdir(snapshot_path))

                        # Verify contents and set as model_id
                        if os.path.isdir(snapshot_path):
                            if _looks_like_ct2_dir(snapshot_path):
                                model_id = snapshot_path
                                logging.info("[ModelInit] Downloaded snapshot is valid CT2 dir: %s", model_id)
                                os.environ["HF_HUB_OFFLINE"] = "1"  # after prefetch, prefer offline
                            else:
                                logging.info("[ModelInit] Snapshot root is not CT2 dir, searching subdirectories")
                                # Sometimes snapshot_download returns root; find CT2 folder inside
                                found_ct2 = False
                                for root, dirs, files in os.walk(snapshot_path):
                                    logging.debug("[FileSystem] Walking: %s (dirs: %s, files: %d)",
                                                root, dirs[:5], len(files))
                                    if _looks_like_ct2_dir(root):
                                        model_id = root
                                        logging.info("[ModelInit] Found CT2 dir inside snapshot: %s", model_id)
                                        os.environ["HF_HUB_OFFLINE"] = "1"
                                        found_ct2 = True
                                        break
                                if not found_ct2:
                                    logging.warning("[ModelInit] No CT2 dir found in downloaded snapshot, using remote id")
                    else:
                        logging.error("[Download] snapshot_path is None after download")

                    # If still not set, fall back to remote id
                    if 'model_id' not in locals():
                        logging.warning("[ModelInit] model_id not set after download, falling back to remote")
                        model_id = remote_short
                        os.environ.pop("HF_HUB_OFFLINE", None)
                except Exception as prefetch_err:
                    logging.error("[ModelInit] snapshot_download failed: %s", str(prefetch_err), exc_info=True)
                    logging.warning("[ModelInit] Falling back to internal downloader with remote id")
                    model_id = remote_short
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
        download_progress_active = False
        download_progress_counter = [0]  # Mutable list to avoid closure issues
        download_timer = None
        download_last_size = [0]  # Track previous size for speed calculation
        download_last_time = [0.0]  # Track previous time for speed calculation

        def update_download_progress():
            """Periodically update download progress with download speed indicator."""
            try:
                logging.debug("[DownloadProgress] update_download_progress called, counter=%d, active=%s",
                             download_progress_counter[0], download_progress_active)

                if not download_progress_active:
                    logging.debug("[DownloadProgress] Progress inactive, returning")
                    return

                # Calculate actual download speed by monitoring cache directory
                speed_str = ""
                try:
                    import time
                    # Get the download cache directory for this model
                    cache_path = os.path.join(models_root(), cache_dir_name_for_repo(download_repo_id[0]))

                    logging.debug("[DownloadProgress] Checking cache path: %s (exists=%s)",
                                 cache_path, os.path.isdir(cache_path))

                    if os.path.isdir(cache_path):
                        # Calculate total size of files in the cache directory
                        total_size = 0
                        file_count = 0
                        for dirpath, dirnames, filenames in os.walk(cache_path):
                            for filename in filenames:
                                filepath = os.path.join(dirpath, filename)
                                try:
                                    file_size = os.path.getsize(filepath)
                                    total_size += file_size
                                    file_count += 1
                                except Exception as file_err:
                                    logging.debug("[DownloadProgress] Failed to get size of %s: %s", filepath, str(file_err))

                        logging.debug("[DownloadProgress] Total size: %d bytes (%d files)", total_size, file_count)

                        current_time = time.time()
                        if download_progress_counter[0] > 0:  # Not the first iteration
                            time_diff = current_time - download_last_time[0]
                            size_diff = total_size - download_last_size[0]

                            logging.debug("[DownloadProgress] Speed calc: size_diff=%d bytes, time_diff=%.2f sec",
                                        size_diff, time_diff)

                            if time_diff > 0 and size_diff >= 0:
                                # Calculate speed in bytes per second
                                speed_bps = size_diff / time_diff

                                logging.debug("[DownloadProgress] Calculated speed: %.2f B/s", speed_bps)

                                # Format speed in human-readable form
                                if speed_bps >= 1024 * 1024 * 1024:  # GB/s
                                    speed_str = f" ({speed_bps / (1024 * 1024 * 1024):.1f} GB/s)"
                                elif speed_bps >= 1024 * 1024:  # MB/s
                                    speed_str = f" ({speed_bps / (1024 * 1024):.1f} MB/s)"
                                elif speed_bps >= 1024:  # KB/s
                                    speed_str = f" ({speed_bps / 1024:.1f} KB/s)"
                                else:
                                    speed_str = f" ({speed_bps:.0f} B/s)"
                            else:
                                # No speed yet, show animated dots
                                logging.debug("[DownloadProgress] No valid speed calculation, using dots")
                                dots = "." * ((download_progress_counter[0] % 4))
                                spaces = " " * (3 - len(dots))
                                speed_str = f"{dots}{spaces}"
                        else:
                            logging.debug("[DownloadProgress] First iteration, initializing tracking")
                            # First iteration - show animated dots while initializing
                            dots = "." * ((download_progress_counter[0] % 4))
                            spaces = " " * (3 - len(dots))
                            speed_str = f"{dots}{spaces}"

                        # Update tracking variables
                        download_last_size[0] = total_size
                        download_last_time[0] = current_time
                    else:
                        logging.debug("[DownloadProgress] Cache directory not found yet, using animated dots")
                        dots = "." * ((download_progress_counter[0] % 4))
                        spaces = " " * (3 - len(dots))
                        speed_str = f"{dots}{spaces}"
                except Exception as e:
                    # If speed calculation fails, fall back to animated dots
                    logging.warning("[DownloadProgress] Speed calculation failed: %s", str(e), exc_info=True)
                    dots = "." * ((download_progress_counter[0] % 4))
                    spaces = " " * (3 - len(dots))
                    speed_str = f"{dots}{spaces}"

                download_progress_counter[0] += 1
                if progress_callback:
                    msg = f"Downloading model{speed_str}"
                    logging.debug("[DownloadProgress] Calling progress_callback with: %s", msg)
                    try:
                        # Use a fixed progress value instead of cycling to avoid false sense of progress
                        # We don't know the actual download progress, so just indicate activity
                        progress_callback(msg, 15)
                    except Exception as cb_err:
                        logging.warning("[DownloadProgress] progress_callback failed: %s", str(cb_err))

                # Schedule next update
                import threading
                nonlocal download_timer
                logging.debug("[DownloadProgress] Scheduling next timer in 0.5s")
                download_timer = threading.Timer(0.5, update_download_progress)
                download_timer.daemon = True
                download_timer.start()
                logging.debug("[DownloadProgress] Timer started successfully")
            except Exception as outer_err:
                logging.error("[DownloadProgress] Critical error in update_download_progress: %s", str(outer_err), exc_info=True)

        if progress_callback:
            progress_callback("Initializing model...", 0)
            # If we resolved to a remote id (not a local directory), warn about long download
            if not os.path.isabs(model_id):
                logging.info("[ModelInit] Starting download progress tracker for remote model: %s", model_id)
                download_progress_active = True
                update_download_progress()
            else:
                logging.info("[ModelInit] Using local model, no download needed: %s", model_id)

        # Direct faster-whisper/HF Hub downloads to a per-user cache folder
        download_root = hf_cache_root()
        os.makedirs(download_root, exist_ok=True)
        # Also direct Hugging Face caches to the same folder for consistency
        os.environ["HF_HOME"] = download_root
        os.environ["HUGGINGFACE_HUB_CACHE"] = download_root
        # Explicitly disable hf_transfer to avoid requiring Rust-based package (cross-platform compatibility)
        # hf_transfer requires Rust compiler and may not work on all Windows/macOS setups
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)  # Remove if set externally
        logging.info(
            "Model source: %s | local_dir_exists=%s | download_root=%s",
            model_id,
            os.path.isdir(self.model_dir),
            download_root,
        )

        logging.info("Loading model: %s (device=%s, compute_type=%s)", model_id, device, compute_type)
        logging.info("[ModelInit] ============ BEGINNING MODEL LOAD ============")
        logging.info("[ModelInit] System: %s", platform.system())
        logging.info("[ModelInit] Model ID: %s", model_id)
        logging.info("[ModelInit] Device: %s", device)
        logging.info("[ModelInit] Compute Type: %s", compute_type)
        logging.info("[ModelInit] Num Workers: %d", num_workers)
        logging.info("[ModelInit] Download Root: %s", download_root)

        # Track actual device/compute_type used after initialization
        self.active_device = device
        self.active_compute_type = compute_type

        # Use configurable number of workers for CTranslate2 (1 = safest, 2-4 = faster but may conflict)
        # Try multiple compute types with fallback for GPU compatibility
        compute_types_to_try = [compute_type]

        # Add fallbacks for GPU mode if float16 fails
        if device == "cuda" and compute_type == "float16":
            compute_types_to_try.extend(["int8_float32", "int8_float16", "int8"])
            logging.info("[ModelInit] GPU mode - will try compute types: %s", compute_types_to_try)

        last_error = None
        loaded = False
        logging.info("[ModelInit] Starting model initialization (CUDA=%s)", device == "cuda")
        if device == "cuda":
            logging.info("[ModelInit] Entering GPU initialization loop")
            for ct in compute_types_to_try:
                try:
                    logging.info("[ModelInit] -------- Attempting GPU model load --------")
                    logging.info("[ModelInit] Attempting to load GPU model with compute_type=%s", ct)
                    logging.info("[ModelInit] WhisperModel params: model_id=%s, device=%s, compute_type=%s, num_workers=%d, cpu_threads=%d, download_root=%s",
                                model_id, device, ct, num_workers, 1 if device == "cpu" else 4, download_root)
                    logging.info("[ModelInit] About to call WhisperModel() constructor...")
                    self.model = WhisperModel(
                        model_id,
                        device=device,
                        compute_type=ct,
                        num_workers=num_workers,
                        cpu_threads=1 if device == "cpu" else 4,
                        download_root=download_root,
                    )
                    logging.info("[ModelInit] WhisperModel() constructor returned successfully")
                    logging.info("[ModelInit] WhisperModel successfully loaded with compute_type=%s", ct)
                    if ct != compute_type:
                        logging.warning("Fell back to compute_type=%s (original %s not supported)", ct, compute_type)
                    loaded = True
                    self.active_device = device
                    self.active_compute_type = ct
                    logging.info("[ModelInit] GPU model loaded successfully, breaking loop")
                    break  # Success, exit loop
                except ValueError as e:
                    logging.error("[ModelInit] ValueError during GPU model load: %s", str(e))
                    if "compute type" in str(e).lower():
                        logging.warning("compute_type=%s not supported on GPU, trying fallback...", ct)
                        last_error = e
                        continue
                    else:
                        # Different error on GPU, record and break to CPU fallback
                        logging.error("[ModelInit] Non-compute-type ValueError on GPU, will try CPU fallback")
                        last_error = e
                        break
                except Exception as e:
                    logging.error("[ModelInit] Unexpected exception during GPU model load: %s", str(e), exc_info=True)
                    last_error = e
                    break

            logging.info("[ModelInit] GPU initialization loop complete (loaded=%s)", loaded)
            if not loaded:
                # Attempt CPU fallback
                logging.info("[ModelInit] -------- Attempting CPU fallback --------")
                cpu_ct = "int8" if (is_macos or is_windows) else "int8_float16"
                logging.warning(
                    "GPU initialization failed (%s). Falling back to CPU with compute_type=%s.",
                    str(last_error) if last_error else "unknown error",
                    cpu_ct,
                )
                if progress_callback:
                    progress_callback("GPU unavailable/unsupported; falling back to CPU...", 15)
                logging.info("[ModelInit] Attempting CPU fallback with compute_type=%s", cpu_ct)
                logging.info("[ModelInit] WhisperModel params: model_id=%s, device=cpu, compute_type=%s, num_workers=%d, cpu_threads=1, download_root=%s",
                            model_id, cpu_ct, num_workers, download_root)
                logging.info("[ModelInit] About to call WhisperModel() constructor (CPU fallback)...")
                self.model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type=cpu_ct,
                    num_workers=num_workers,
                    cpu_threads=1,
                    download_root=download_root,
                )
                logging.info("[ModelInit] WhisperModel() constructor returned (CPU fallback)")
                loaded = True
                logging.info("[ModelInit] CPU fallback succeeded (compute_type=%s)", cpu_ct)
                self.active_device = "cpu"
                self.active_compute_type = cpu_ct
        else:
            # Non-GPU path: just load with the chosen CPU compute type
            logging.info("[ModelInit] -------- Direct CPU model load (non-GPU path) --------")
            logging.info("[ModelInit] Loading CPU model directly with compute_type=%s", compute_type)
            logging.info("[ModelInit] WhisperModel params: model_id=%s, device=%s, compute_type=%s, num_workers=%d, cpu_threads=1, download_root=%s",
                        model_id, device, compute_type, num_workers, download_root)
            logging.info("[ModelInit] About to call WhisperModel() constructor (direct CPU)...")
            self.model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                num_workers=num_workers,
                cpu_threads=1,
                download_root=download_root,
            )
            logging.info("[ModelInit] WhisperModel() constructor returned (direct CPU)")
            logging.info("[ModelInit] CPU model successfully loaded")
            self.active_device = device
            self.active_compute_type = compute_type
            logging.info("[ModelInit] CPU model initialization complete")

        # Stop download progress timer if it was started
        logging.info("[ModelInit] ============ MODEL LOAD COMPLETE ============")
        logging.info("[ModelInit] Stopping download progress timer (active=%s, timer=%s)", download_progress_active, download_timer is not None)
        download_progress_active = False
        if download_timer:
            try:
                logging.info("[ModelInit] Cancelling download timer...")
                download_timer.cancel()
                logging.info("[ModelInit] Download timer cancelled successfully")
            except Exception as timer_err:
                logging.warning("[ModelInit] Failed to cancel download timer: %s", str(timer_err))
        else:
            logging.info("[ModelInit] No download timer to cancel")

        if progress_callback:
            logging.info("[ModelInit] Model initialization complete, calling progress_callback with 'Model loaded.'")
            progress_callback("Model loaded.", 100)
        else:
            logging.info("[ModelInit] No progress_callback provided")

        logging.info("[ModelInit] Setting language=%s, word_timestamps=%s", language_hint, word_timestamps)

        logging.info("[ModelInit] Setting language=%s, word_timestamps=%s", language_hint, word_timestamps)
        self.language = language_hint
        self.word_timestamps = word_timestamps
        logging.info("[ModelInit] ============ TRANSCRIBER INIT COMPLETE ============")

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
