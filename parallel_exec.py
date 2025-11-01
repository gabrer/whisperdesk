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

    # Build filename: YYYYMMDD_HHMM_originalName_modelName.ext
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    original_name = os.path.splitext(os.path.basename(wav))[0]
    model_short = model_name.replace("whisper-", "").replace("-ct2", "")
    # New naming: [YYYYMMDD_HHMM_modelname_fileName]
    base_name = f"{timestamp}_{model_short}_{original_name}"

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
    global _TR_PROCESS
    # Limit inner threading to reduce contention across processes
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    _TR_PROCESS = Transcriber(
        model_name=model_name,
        device_mode=device_mode,
        language_hint=language_hint,
        word_timestamps=word_timestamps,
        num_workers=num_workers,
        progress_callback=None,
    )


def mp_transcribe_and_export(task: Dict[str, Any]) -> Optional[str]:
    """Run the full pipeline for a single file inside a process pool worker.

    Returns:
        None on success, or an error string on failure.
    """
    try:
        tr = _TR_PROCESS
        if tr is None:
            # Fallback (should not happen if initializer ran): build a local transcriber
            tr = Transcriber(
                model_name=task["model_name"],
                device_mode=task["device_mode"],
                language_hint=task["language_hint"],
                word_timestamps=task["word_timestamps"],
                num_workers=task["num_workers"],
                progress_callback=None,
            )

        wav = task["wav"]
        paths = _build_output_paths(wav, task["model_name"], task["output_dir"])

        asr = tr.transcribe(wav)
        # diarization
        if task["diarization_engine"] == 'none' or task["diarization_max_speakers"] <= 1:
            duration = float(asr.get("duration", 0.0))
            diar = [(0.0, duration, 0)]
        else:
            diar = diarize(wav, max_speakers=task["diarization_max_speakers"], engine=task["diarization_engine"])
        segs = assign_speakers_to_asr(asr["segments"], diar)

        # Determine if we should include speaker labels
        spk_ids = sorted(set([s.get("speaker", 0) for s in segs]))
        include_speakers = (
            task["diarization_engine"] != 'none'
            and task["diarization_max_speakers"] > 1
            and len(spk_ids) > 1
        )
        speaker_map = {i: f"Speaker {i+1}" for i in spk_ids}

        # export
        if "txt" in task["output_formats"]:
            export_txt(paths["out_txt"], segs, speaker_map, include_speakers=include_speakers)
        if "docx" in task["output_formats"]:
            export_docx(paths["out_docx"], segs, speaker_map, include_speakers=include_speakers)
        return None
    except Exception as e:
        return str(e)


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
    try:
        tr = _thread_get_transcriber(task)
        wav = task["wav"]
        paths = _build_output_paths(wav, task["model_name"], task["output_dir"])

        asr = tr.transcribe(wav)
        if task["diarization_engine"] == 'none' or task["diarization_max_speakers"] <= 1:
            duration = float(asr.get("duration", 0.0))
            diar = [(0.0, duration, 0)]
        else:
            diar = diarize(wav, max_speakers=task["diarization_max_speakers"], engine=task["diarization_engine"])
        segs = assign_speakers_to_asr(asr["segments"], diar)

        spk_ids = sorted(set([s.get("speaker", 0) for s in segs]))
        include_speakers = (
            task["diarization_engine"] != 'none'
            and task["diarization_max_speakers"] > 1
            and len(spk_ids) > 1
        )
        speaker_map = {i: f"Speaker {i+1}" for i in spk_ids}

        if "txt" in task["output_formats"]:
            export_txt(paths["out_txt"], segs, speaker_map, include_speakers=include_speakers)
        if "docx" in task["output_formats"]:
            export_docx(paths["out_docx"], segs, speaker_map, include_speakers=include_speakers)
        return None
    except Exception as e:
        return str(e)
