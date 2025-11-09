import os
import threading
from typing import Dict, Any, Optional

from transcription import Transcriber
from diarization import diarize, assign_speakers_to_asr
from exporters import export_txt, export_docx
from utils import models_root  # noqa: F401 (reserved for future use)

# Process-global / thread-local caches for Transcriber
_TR_PROCESS = None
_TR_THREAD_LOCAL = threading.local()


def _build_output_paths(wav: str, model_name: str, out_dir_cfg: str) -> Dict[str, str]:
    # Determine output base path (respect configured output directory if set)
    out_dir = (out_dir_cfg or "").strip()
    if out_dir:
        if not os.path.isabs(out_dir):
            out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        # Use same directory as input file
        out_dir = os.path.dirname(os.path.abspath(wav))

    # Build filename: fileName_YYYYMMDD_HHMM_modelName.ext
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    original_name = os.path.splitext(os.path.basename(wav))[0]
    model_short = model_name.replace("whisper-", "").replace("-ct2", "")
    # New naming: [fileName_yearMonthDay_time_modelName]
    base_name = f"{original_name}_{timestamp}_{model_short}"

    out_txt = os.path.join(out_dir, base_name + ".txt")
    out_docx = os.path.join(out_dir, base_name + ".docx")
    return {"out_dir": out_dir, "out_txt": out_txt, "out_docx": out_docx}


# ------------- Multiprocessing support -------------

def mp_initializer(model_name: str, device_mode: str, language_hint: str,
                   word_timestamps: bool, num_workers: int):
    """Initializer for process pool workers.

    Creates and caches a Transcriber per process to avoid re-loading the model
    on every task, and constrains intra-op threads to avoid oversubscription.
    """
    import logging
    global _TR_PROCESS

    logging.info("[MPInit] ========== PROCESS INITIALIZER STARTED ==========")
    logging.info("[MPInit] Process PID: %d", os.getpid())
    logging.info("[MPInit] model_name: %s", model_name)
    logging.info("[MPInit] device_mode: %s", device_mode)
    logging.info("[MPInit] language_hint: %s", language_hint)
    logging.info("[MPInit] word_timestamps: %s", word_timestamps)
    logging.info("[MPInit] num_workers: %d", num_workers)

    # Configure inner threading to avoid contention across processes
    try:
        cores = os.cpu_count() or 4
        per_proc_threads = max(1, cores // max(1, int(num_workers or 1)))
        os.environ['OMP_NUM_THREADS'] = str(per_proc_threads)
        os.environ['MKL_NUM_THREADS'] = str(per_proc_threads)
        logging.info("[MPInit] Configured threading: OMP_NUM_THREADS=%s, MKL_NUM_THREADS=%s (cores=%d)",
                    per_proc_threads, per_proc_threads, cores)
    except Exception as e:
        logging.warning("[MPInit] Failed to configure threading: %s, using defaults", str(e))
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        logging.info("[MPInit] Using default threading: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1")

    logging.info("[MPInit] Creating Transcriber instance...")
    try:
        _TR_PROCESS = Transcriber(
            model_name=model_name,
            device_mode=device_mode,
            language_hint=language_hint,
            word_timestamps=word_timestamps,
            num_workers=num_workers,
            progress_callback=None,
        )
        logging.info("[MPInit] Transcriber instance created successfully")
        logging.info("[MPInit] ========== PROCESS INITIALIZER COMPLETE ==========")
    except Exception as e:
        logging.error("[MPInit] Failed to create Transcriber: %s", str(e))
        import traceback
        logging.error("[MPInit] Traceback:\n%s", traceback.format_exc())
        raise


def mp_transcribe_and_export(task: dict) -> dict:
    """Worker function that runs inside a process of the pool.

    Reads the task dict to extract parameters, transcribes the audio using the
    global Transcriber from the initializer, then exports the result to disk.
    Passes through exceptions as-is so as_completed can handle or re-raise.
    """
    import logging
    global _TR_PROCESS

    logging.info("[MPWorker] ========== TASK STARTED ==========")
    logging.info("[MPWorker] Process PID: %d", os.getpid())
    logging.info("[MPWorker] Task audio_file: %s", task.get('audio_file', 'N/A'))

    if _TR_PROCESS is None:
        logging.error("[MPWorker] _TR_PROCESS is None - initializer failed!")
        raise RuntimeError("Process pool worker was not initialised correctly")

    logging.info("[MPWorker] _TR_PROCESS is available")

    audio_file = task['audio_file']
    base, ext = os.path.splitext(audio_file)
    export_cfg = task['export_cfg']
    txt_file = f"{base}.txt"
    srt_file = f"{base}.srt"

    logging.info("[MPWorker] Export config: txt_enable=%s, srt_enable=%s",
                export_cfg.txt_enable, export_cfg.srt_enable)

    # Run actual transcription
    logging.info("[MPWorker] Starting transcription...")
    from transcription import Transcriber, TranscriptSegment
    tr: Transcriber = _TR_PROCESS
    try:
        result = tr.transcribe_file(audio_file)
        logging.info("[MPWorker] Transcription complete: %d segments, language=%s",
                    len(result.segments), result.language)
    except Exception as e:
        logging.error("[MPWorker] Transcription failed: %s", str(e))
        import traceback
        logging.error("[MPWorker] Traceback:\n%s", traceback.format_exc())
        raise

    # Export results
    logging.info("[MPWorker] Starting export...")
    try:
        if export_cfg.txt_enable:
            exporters.export_txt([result], txt_file)
            logging.info("[MPWorker] Exported TXT to: %s", txt_file)
        if export_cfg.srt_enable:
            exporters.export_srt([result], srt_file)
            logging.info("[MPWorker] Exported SRT to: %s", srt_file)
    except Exception as e:
        logging.error("[MPWorker] Export failed: %s", str(e))
        import traceback
        logging.error("[MPWorker] Traceback:\n%s", traceback.format_exc())
        raise

    # Return summary
    summary = {
        'audio_file': audio_file,
        'text': result.text,
        'segments': len(result.segments),
        'language': result.language,
    }
    logging.info("[MPWorker] Task complete, returning summary")
    logging.info("[MPWorker] ========== TASK COMPLETE ==========")
    return summary


# ------------- Threading support -------------

def _thread_get_transcriber(task: Dict[str, Any]) -> Transcriber:
    tr = getattr(_TR_THREAD_LOCAL, "tr", None)
    if tr is None:
        tr = Transcriber(
            model_name=task["model_name"],
            device_mode=task["device_mode"],
            language_hint=task["language_hint"],
            word_timestamps=task["word_timestamps"],
            num_workers=task["num_workers"],
            progress_callback=None,
        )
        _TR_THREAD_LOCAL.tr = tr
    return tr


def thread_transcribe_and_export(task: Dict[str, Any]) -> Optional[str]:
    import logging
    logging.info("[ThreadWorker] ========== TASK STARTED ==========")
    logging.info("[ThreadWorker] Thread ID: %s", threading.current_thread().name)
    logging.info("[ThreadWorker] Task wav: %s", task.get("wav", "N/A"))
    logging.info("[ThreadWorker] Task model_name: %s", task.get("model_name", "N/A"))
    logging.info("[ThreadWorker] Task device_mode: %s", task.get("device_mode", "N/A"))
    logging.info("[ThreadWorker] Task output_formats: %s", task.get("output_formats", []))

    try:
        logging.info("[ThreadWorker] Getting/creating transcriber...")
        tr = _thread_get_transcriber(task)
        logging.info("[ThreadWorker] Transcriber obtained successfully")

        wav = task["wav"]
        paths = _build_output_paths(wav, task["model_name"], task["output_dir"])
        logging.info("[ThreadWorker] Output paths: txt=%s, docx=%s",
                    paths.get("out_txt"), paths.get("out_docx"))

        logging.info("[ThreadWorker] Starting ASR transcription...")
        asr = tr.transcribe(wav)
        logging.info("[ThreadWorker] ASR complete: %d segments, duration=%.2f",
                    len(asr.get("segments", [])), float(asr.get("duration", 0.0)))

        logging.info("[ThreadWorker] Diarization engine: %s, max_speakers: %d",
                    task["diarization_engine"], task["diarization_max_speakers"])

        if task["diarization_engine"] == 'none' or task["diarization_max_speakers"] <= 1:
            duration = float(asr.get("duration", 0.0))
            diar = [(0.0, duration, 0)]
            logging.info("[ThreadWorker] Skipping diarization (single speaker mode)")
        else:
            logging.info("[ThreadWorker] Running diarization...")
            diar = diarize(wav, max_speakers=task["diarization_max_speakers"], engine=task["diarization_engine"])
            logging.info("[ThreadWorker] Diarization complete: %d segments", len(diar))

        logging.info("[ThreadWorker] Assigning speakers to ASR segments...")
        segs = assign_speakers_to_asr(asr["segments"], diar)

        spk_ids = sorted(set([s.get("speaker", 0) for s in segs]))
        include_speakers = (
            task["diarization_engine"] != 'none'
            and task["diarization_max_speakers"] > 1
            and len(spk_ids) > 1
        )
        speaker_map = {i: f"Speaker {i+1}" for i in spk_ids}
        logging.info("[ThreadWorker] Speaker assignment complete: %d unique speakers", len(spk_ids))

        include_timestamps = bool(task.get("word_timestamps", False))
        logging.info("[ThreadWorker] Export settings: include_speakers=%s, include_timestamps=%s",
                    include_speakers, include_timestamps)

        if "txt" in task["output_formats"]:
            logging.info("[ThreadWorker] Exporting TXT...")
            export_txt(paths["out_txt"], segs, speaker_map, include_speakers=include_speakers, include_timestamps=include_timestamps)
            logging.info("[ThreadWorker] TXT export complete: %s", paths["out_txt"])
        if "docx" in task["output_formats"]:
            logging.info("[ThreadWorker] Exporting DOCX...")
            export_docx(paths["out_docx"], segs, speaker_map, include_speakers=include_speakers, include_timestamps=include_timestamps)
            logging.info("[ThreadWorker] DOCX export complete: %s", paths["out_docx"])

        logging.info("[ThreadWorker] Task complete successfully")
        logging.info("[ThreadWorker] ========== TASK COMPLETE ==========")
        return None
    except Exception as e:
        logging.error("[ThreadWorker] Task failed with exception: %s", str(e))
        import traceback
        logging.error("[ThreadWorker] Traceback:\n%s", traceback.format_exc())
        logging.info("[ThreadWorker] ========== TASK FAILED ==========")
        return str(e)
