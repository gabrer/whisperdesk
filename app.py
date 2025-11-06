import os
import sys
import logging
from typing import List
import threading
import signal

from PySide6.QtCore import QThread, Signal, QUrl, Qt
from datetime import datetime
from PySide6.QtGui import (
    QStandardItemModel, QStandardItem, QDesktopServices,
    QGuiApplication, QFont, QPalette, QColor
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QListWidget, QListWidgetItem, QComboBox, QCheckBox, QTabWidget,
    QFormLayout, QLineEdit, QAbstractItemView, QProgressBar, QFrame
)

from utils import setup_logging, models_root, app_root, hf_cache_root
from device import device_banner
from settings import Settings
from transcription import Transcriber
from diarization import diarize, assign_speakers_to_asr, ensure_ecapa_model
from exporters import export_txt, export_docx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from parallel_exec import mp_initializer, mp_transcribe_and_export, thread_transcribe_and_export


class Worker(QThread):
    file_done = Signal(str)
    file_error = Signal(str, str)
    all_done = Signal()
    progress_update = Signal(str, int)  # (message, percentage 0-100)
    device_info = Signal(str)  # e.g., "cpu (int8)" or "cuda (float16)"

    def __init__(self, files: List[str], cfg: Settings, model_name: str, parent=None):
        super().__init__(parent)
        self.files = files
        self.cfg = cfg
        self.model_name = model_name
        self.cancelled = False

    def run(self):
        def on_progress(msg: str, pct: int):
            self.progress_update.emit(msg, pct)

        try:
            on_progress("Initializing model (this may download files if needed)...", 0)
            # Set environment variable to avoid OpenMP conflicts with Qt
            os.environ['OMP_NUM_THREADS'] = '1'

            # Windows-specific fix: Use a timeout for model initialization
            # to prevent infinite hangs during download
            tr = None
            init_error = None

            def init_model():
                nonlocal tr, init_error
                try:
                    tr = Transcriber(
                        model_name=self.model_name,
                        device_mode=self.cfg.device_mode,
                        language_hint=self.cfg.language_hint,
                        word_timestamps=self.cfg.word_timestamps,
                        num_workers=self.cfg.num_workers,
                        progress_callback=on_progress
                    )
                except Exception as e:
                    init_error = e

            # Run model initialization in a thread with timeout
            init_thread = threading.Thread(target=init_model, daemon=True)
            init_thread.start()
            init_thread.join(timeout=600)  # 10 minute timeout

            if init_thread.is_alive():
                # Timeout occurred
                raise TimeoutError(
                    f"Model initialization timed out after 10 minutes. "
                    f"This usually means the download is stuck. "
                    f"Please manually download the model to {models_root()} "
                    f"or use a model that's already available locally."
                )

            if init_error:
                raise init_error

            if tr is None:
                raise RuntimeError("Model initialization failed for unknown reason")

            # Inform UI about the actual runtime device in use
            try:
                info = f"{getattr(tr, 'active_device', self.cfg.device_mode)} ({getattr(tr, 'active_compute_type', 'unknown')})"
                self.device_info.emit(info)
            except Exception:
                pass
            on_progress("Model ready.", 100)
            # Pre-fetch diarization model if user requested multi-speaker diarization and engine is enabled
            if self.cfg.diarization_max_speakers > 1 and self.cfg.diarization_engine != 'none':
                ensure_ecapa_model(progress_callback=on_progress)
        except Exception as e:
            import traceback
            error_msg = str(e)

            # Provide helpful message for common download timeout issues
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_msg = (
                    f"Model download timed out. This can happen with slow connections or network issues. "
                    f"Try: (1) Check your internet connection, (2) Manually download the model to {models_root()}, "
                    f"or (3) Use a smaller model. Original error: {error_msg}"
                )
            elif "huggingface" in error_msg.lower() or "hf" in error_msg.lower():
                error_msg = (
                    f"HuggingFace download failed. Try manually downloading the model to {models_root()}. "
                    f"Original error: {error_msg}"
                )

            logging.error("Model initialization failed: %s\n%s", error_msg, traceback.format_exc())
            self.file_error.emit("<init>", error_msg)
            self.all_done.emit()
            return

        # Show that processing is starting
        on_progress(f"Processing {len(self.files)} file(s)...", 1)

        processed = set()
        parallel_executed = False

        # Decide parallelization strategy: prefer multiprocessing on CPU; avoid parallel on single GPU
        active_device = getattr(tr, 'active_device', self.cfg.device_mode)
        use_parallel = (
            active_device == 'cpu' and
            self.cfg.num_workers > 1 and
            len(self.files) > 1
        )

        if use_parallel:
            # Build per-file task payloads
            tasks = []
            for wav in self.files:
                tasks.append({
                    "wav": wav,
                    "model_name": self.model_name,
                    # Force CPU for parallel workers to prevent GPU contention
                    "device_mode": 'cpu',
                    "language_hint": self.cfg.language_hint,
                    "word_timestamps": self.cfg.word_timestamps,
                    # Avoid over-subscription: 1 worker inside each process/thread
                    "num_workers": 1,
                    "output_dir": (self.cfg.output_dir or "").strip(),
                    "output_formats": list(self.cfg.output_formats or ["txt"]),
                    "diarization_engine": self.cfg.diarization_engine,
                    "diarization_max_speakers": int(self.cfg.diarization_max_speakers),
                })

            self.progress_update.emit(
                f"Processing {len(tasks)} files in parallel on CPU (workers: {self.cfg.num_workers})…",
                1,
            )
            # Attempt multiprocessing first
            try:
                with ProcessPoolExecutor(
                    max_workers=self.cfg.num_workers,
                    initializer=mp_initializer,
                    initargs=(self.model_name, 'cpu', self.cfg.language_hint, self.cfg.word_timestamps, 1),
                ) as ex:
                    fut_map = {ex.submit(mp_transcribe_and_export, t): t["wav"] for t in tasks}
                    for fut in as_completed(fut_map):
                        wav = fut_map[fut]
                        if self.cancelled:
                            break
                        try:
                            err = fut.result()
                        except Exception as e:
                            err = str(e)
                        if err:
                            self.file_error.emit(wav, err)
                        else:
                            self.file_done.emit(wav)
                        processed.add(wav)
                parallel_executed = True
            except Exception as e:
                logging.warning("Multiprocessing failed (%s). Falling back to threads.", e)
                try:
                    with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                        fut_map = {ex.submit(thread_transcribe_and_export, t): t["wav"] for t in tasks}
                        for fut in as_completed(fut_map):
                            wav = fut_map[fut]
                            if self.cancelled:
                                break
                            try:
                                err = fut.result()
                            except Exception as e2:
                                err = str(e2)
                            if err:
                                self.file_error.emit(wav, err)
                            else:
                                self.file_done.emit(wav)
                            processed.add(wav)
                    parallel_executed = True
                except Exception as e2:
                    logging.warning("Thread pool failed (%s). Falling back to sequential.", e2)

        # If not using parallel (GPU or single worker) OR pools failed → sequential for remaining files
        if (not use_parallel) or (use_parallel and not parallel_executed):
            for wav in self.files:
                if wav in processed:
                    continue
                if self.cancelled:
                    break
                try:
                    # Show which file is being processed
                    on_progress(f"Processing: {os.path.basename(wav)}...", 0)
                    # Reuse the threaded worker task to run sequentially in this thread
                    task = {
                        "wav": wav,
                        "model_name": self.model_name,
                        "device_mode": self.cfg.device_mode,
                        "language_hint": self.cfg.language_hint,
                        "word_timestamps": self.cfg.word_timestamps,
                        "num_workers": self.cfg.num_workers,
                        "output_dir": (self.cfg.output_dir or "").strip(),
                        "output_formats": list(self.cfg.output_formats or ["txt"]),
                        "diarization_engine": self.cfg.diarization_engine,
                        "diarization_max_speakers": int(self.cfg.diarization_max_speakers),
                    }
                    err = thread_transcribe_and_export(task)
                    if err:
                        self.file_error.emit(wav, err)
                    else:
                        self.file_done.emit(wav)
                except Exception as e:
                    self.file_error.emit(wav, str(e))

        self.all_done.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperDesk")
        self.resize(1200, 800)

        self.cfg = Settings()
        self.log_path = setup_logging(self.cfg.log_level)

        # Main layout with proper margins
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left column - main content
        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        # Progress card - MOVED TO TOP
        progress_card = QFrame()
        progress_card.setObjectName("card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        progress_layout.setSpacing(8)

        self.progress_label = QLabel("Ready to transcribe")
        self.progress_label.setObjectName("body")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)

        left_column.addWidget(progress_card)

        # Header card - Device info (now below progress)
        header_card = QFrame()
        header_card.setObjectName("card")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(8)

        self.status_label = QLabel(device_banner(is_gpu=(self.cfg.device_mode != 'cpu')))
        self.status_label.setObjectName("subtitle")
        header_layout.addWidget(self.status_label)

        self.device_runtime_label = QLabel("Runtime device: —")
        self.device_runtime_label.setObjectName("caption")
        header_layout.addWidget(self.device_runtime_label)

        left_column.addWidget(header_card)

        # File list card
        file_card = QFrame()
        file_card.setObjectName("card")
        file_card_layout = QVBoxLayout(file_card)
        file_card_layout.setContentsMargins(16, 16, 16, 16)
        file_card_layout.setSpacing(12)

        # File list header
        file_header = QLabel("Audio Files")
        file_header.setObjectName("cardTitle")
        file_card_layout.addWidget(file_header)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        file_card_layout.addWidget(self.file_list, 1)

        # File action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setObjectName("btn_secondary")
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setObjectName("btn_secondary")
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        file_card_layout.addLayout(btn_row)

        left_column.addWidget(file_card, 1)

        # Controls card
        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_card_layout = QVBoxLayout(controls_card)
        controls_card_layout.setContentsMargins(16, 16, 16, 16)
        controls_card_layout.setSpacing(12)

        # Controls header
        controls_header = QLabel("Configuration")
        controls_header.setObjectName("cardTitle")
        controls_card_layout.addWidget(controls_header)

        # Model selection row
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self._rebuild_model_combo()
        pref = f"whisper-{self.cfg.model_preference}-ct2"
        texts = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        if pref in texts:
            self.model_combo.setCurrentText(pref)

        self.btn_fav = QPushButton("★")
        self.btn_fav.setFixedWidth(36)
        self.btn_fav.setObjectName("btn_icon")
        self.btn_fav.setToolTip("Toggle favorite")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_fav.clicked.connect(self._toggle_favorite)
        self._on_model_changed(self.model_combo.currentText())

        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.btn_fav)
        controls_card_layout.addLayout(model_row)

        # Output formats row
        format_row = QHBoxLayout()
        format_row.setSpacing(12)
        format_label = QLabel("Output:")
        self.chk_txt = QCheckBox("TXT")
        self.chk_txt.setChecked("txt" in self.cfg.output_formats)
        self.chk_docx = QCheckBox("DOCX")
        self.chk_docx.setChecked("docx" in self.cfg.output_formats)
        format_row.addWidget(format_label)
        format_row.addWidget(self.chk_txt)
        format_row.addWidget(self.chk_docx)
        format_row.addStretch()
        controls_card_layout.addLayout(format_row)

        left_column.addWidget(controls_card)

        # Action buttons row
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_transcribe = QPushButton("Start Transcription")
        self.btn_transcribe.setObjectName("btn_primary")
        self.btn_transcribe.setMinimumHeight(36)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(36)

        self.btn_open_output = QPushButton("Open Output")
        self.btn_open_output.setObjectName("btn_secondary")

        self.btn_open_logs = QPushButton("Open Logs")
        self.btn_open_logs.setObjectName("btn_secondary")

        action_row.addWidget(self.btn_transcribe, 2)
        action_row.addWidget(self.btn_stop, 1)
        action_row.addWidget(self.btn_open_output)
        action_row.addWidget(self.btn_open_logs)

        left_column.addLayout(action_row)

        # Add left column to main layout
        main_layout.addLayout(left_column, 2)

        # Right column - Settings panel
        right_column = QVBoxLayout()
        right_column.setSpacing(16)

        # Settings tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_easy_tab(), "Basic Settings")
        tabs.addTab(self._build_advanced_tab(), "Advanced Settings")
        right_column.addWidget(tabs)

        # Footer: University copyrights (bottom of right panel)
        right_column.addStretch(1)
        current_year = datetime.now().year
        copyright_label = QLabel(
            f"\u00A9 {current_year} University of Warwick\n"
            f"\u00A9 {current_year} University of Leeds\n"
            "All rights reserved."
        )
        copyright_label.setObjectName("caption")
        copyright_label.setWordWrap(True)
        try:
            # Small, subtle footer aligned to bottom-left, selectable text
            copyright_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            copyright_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        except Exception:
            pass
        right_column.addWidget(copyright_label)

        # Add right column to main layout
        main_layout.addLayout(right_column, 1)

        # Update parallel hint
        try:
            self.device_combo.currentTextChanged.connect(self._update_parallel_hint)
            self.num_workers_combo.currentIndexChanged.connect(self._update_parallel_hint)
            self._update_parallel_hint()
        except Exception:
            pass

        # Connect signals
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_remove.clicked.connect(self.remove_selected_files)
        self.btn_clear.clicked.connect(self.file_list.clear)
        self.btn_transcribe.clicked.connect(self.start_transcription)
        self.btn_stop.clicked.connect(self.stop_transcription)
        self.btn_open_output.clicked.connect(self.open_output_folder)
        self.btn_open_logs.clicked.connect(self.open_logs)

        self.worker = None
        self.total_files = 0
        self.completed_count = 0
        self.error_count = 0
        self._init_failed = False

    def closeEvent(self, event):
        """Handle window close event - ensure all threads are properly terminated."""
        # Stop the worker thread if it's running
        if self.worker and self.worker.isRunning():
            logging.info("Application closing - stopping worker thread...")
            self.worker.cancelled = True
            # Give the worker a moment to finish current file
            self.worker.wait(3000)  # Wait up to 3 seconds
            if self.worker.isRunning():
                # Force terminate if still running after timeout
                logging.warning("Worker thread did not stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait(1000)  # Wait for termination

        # Accept the close event
        event.accept()
        logging.info("Application closed.")

    def _available_models(self) -> List[str]:
        root = models_root()
        known = [
            "whisper-tiny-ct2",
            "whisper-base-ct2",
            "whisper-small-ct2",
            "whisper-medium-ct2",
            # English-only variants
            "whisper-small-en-ct2",
            "whisper-medium-en-ct2",
            # Large variants
            "whisper-large-v2-ct2",
            "whisper-large-v3-ct2",
            "whisper-large-v3-turbo-ct2",
            # Distilled
            "whisper-distil-large-v3-ct2",
        ]
        local = []
        if os.path.isdir(root):
            local = [d for d in os.listdir(root) if d.startswith('whisper-')]
        # union while preserving known order first, then any extra local folders
        seen = set()
        result = []
        for name in known + [d for d in local if d not in known]:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def _rebuild_model_combo(self):
        # Build a grouped list with favorites and all models
        all_models = self._available_models()
        favs = [m for m in self.cfg.favorite_models if m in all_models]
        others = [m for m in all_models if m not in favs]

        model = QStandardItemModel()
        def add_header(text: str):
            item = QStandardItem(text)
            # Disable selection for headers (avoid Qt enum usage to keep it simple)
            try:
                item.setEnabled(False)
                item.setSelectable(False)
            except Exception:
                pass
            model.appendRow(item)

        if favs:
            add_header("— Favorites —")
            for m in favs:
                model.appendRow(QStandardItem(m))
            add_header("— All Models —")
        for m in others:
            model.appendRow(QStandardItem(m))

        self.model_combo.setModel(model)

    def _on_model_changed(self, text: str):
        # Update star icon depending on favorite status
        if not text or text.startswith("— "):
            return
        starred = text in (self.cfg.favorite_models or [])
        self.btn_fav.setText("★" if starred else "☆")

    def _toggle_favorite(self):
        text = self.model_combo.currentText()
        if not text or text.startswith("— "):
            return
        favs = set(self.cfg.favorite_models or [])
        if text in favs:
            favs.remove(text)
        else:
            favs.add(text)
        self.cfg.favorite_models = sorted(favs)
        self.cfg.save()
        self._rebuild_model_combo()
        self._on_model_changed(text)

    def _update_parallel_hint(self):
        try:
            dev = self.device_combo.currentText().strip().lower()
            workers_text = self.num_workers_combo.currentText().strip()
            workers = int(workers_text.split()[0]) if workers_text else 1
            enabled = (dev == "cpu" and workers > 1)
            self.parallel_hint.setText("Parallel mode (CPU)" if enabled else "Disabled")
        except Exception:
            # Best-effort; ignore errors
            pass

    def _build_easy_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        # Language hint
        self.lang_combo = QComboBox()
        self.lang_combo.setEditable(False)
        # List of languages supported by Whisper
        languages = [
            ("auto", "Auto-detect"),
            ("af", "Afrikaans"),
            ("am", "Amharic"),
            ("ar", "Arabic"),
            ("as", "Assamese"),
            ("az", "Azerbaijani"),
            ("ba", "Bashkir"),
            ("be", "Belarusian"),
            ("bg", "Bulgarian"),
            ("bn", "Bengali"),
            ("bo", "Tibetan"),
            ("br", "Breton"),
            ("bs", "Bosnian"),
            ("ca", "Catalan"),
            ("cs", "Czech"),
            ("cy", "Welsh"),
            ("da", "Danish"),
            ("de", "German"),
            ("el", "Greek"),
            ("en", "English"),
            ("es", "Spanish"),
            ("et", "Estonian"),
            ("eu", "Basque"),
            ("fa", "Persian"),
            ("fi", "Finnish"),
            ("fo", "Faroese"),
            ("fr", "French"),
            ("gl", "Galician"),
            ("gu", "Gujarati"),
            ("ha", "Hausa"),
            ("haw", "Hawaiian"),
            ("he", "Hebrew"),
            ("hi", "Hindi"),
            ("hr", "Croatian"),
            ("ht", "Haitian Creole"),
            ("hu", "Hungarian"),
            ("hy", "Armenian"),
            ("id", "Indonesian"),
            ("is", "Icelandic"),
            ("it", "Italian"),
            ("ja", "Japanese"),
            ("jw", "Javanese"),
            ("ka", "Georgian"),
            ("kk", "Kazakh"),
            ("km", "Khmer"),
            ("kn", "Kannada"),
            ("ko", "Korean"),
            ("la", "Latin"),
            ("lb", "Luxembourgish"),
            ("ln", "Lingala"),
            ("lo", "Lao"),
            ("lt", "Lithuanian"),
            ("lv", "Latvian"),
            ("mg", "Malagasy"),
            ("mi", "Maori"),
            ("mk", "Macedonian"),
            ("ml", "Malayalam"),
            ("mn", "Mongolian"),
            ("mr", "Marathi"),
            ("ms", "Malay"),
            ("mt", "Maltese"),
            ("my", "Myanmar"),
            ("ne", "Nepali"),
            ("nl", "Dutch"),
            ("nn", "Norwegian Nynorsk"),
            ("no", "Norwegian"),
            ("oc", "Occitan"),
            ("pa", "Punjabi"),
            ("pl", "Polish"),
            ("ps", "Pashto"),
            ("pt", "Portuguese"),
            ("ro", "Romanian"),
            ("ru", "Russian"),
            ("sa", "Sanskrit"),
            ("sd", "Sindhi"),
            ("si", "Sinhala"),
            ("sk", "Slovak"),
            ("sl", "Slovenian"),
            ("sn", "Shona"),
            ("so", "Somali"),
            ("sq", "Albanian"),
            ("sr", "Serbian"),
            ("su", "Sundanese"),
            ("sv", "Swedish"),
            ("sw", "Swahili"),
            ("ta", "Tamil"),
            ("te", "Telugu"),
            ("tg", "Tajik"),
            ("th", "Thai"),
            ("tk", "Turkmen"),
            ("tl", "Tagalog"),
            ("tr", "Turkish"),
            ("tt", "Tatar"),
            ("uk", "Ukrainian"),
            ("ur", "Urdu"),
            ("uz", "Uzbek"),
            ("vi", "Vietnamese"),
            ("yi", "Yiddish"),
            ("yo", "Yoruba"),
            ("yue", "Cantonese"),
            ("zh", "Chinese"),
        ]
        for code, name in languages:
            self.lang_combo.addItem(f"{name} ({code})", code)
        # Set current language from config
        current_lang = self.cfg.language_hint or "en"
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == current_lang:
                self.lang_combo.setCurrentIndex(i)
                break
        f.addRow("Language:", self.lang_combo)
        # Output folder selector
        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.cfg.output_dir)
        self.btn_output_browse = QPushButton("Browse…")
        out_row.addWidget(self.output_dir_edit, 1)
        out_row.addWidget(self.btn_output_browse)
        f.addRow("Output folder:", out_row)
        self.btn_output_browse.clicked.connect(self.choose_output_dir)
        return w

    def _build_advanced_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        # Word timestamps
        self.chk_word_ts = QCheckBox()
        self.chk_word_ts.setChecked(self.cfg.word_timestamps)
        f.addRow("Word-level timestamps:", self.chk_word_ts)
        # Diarization cap
        self.max_spk = QComboBox()
        self.max_spk.addItems(["1", "2", "3"])
        self.max_spk.setCurrentText(str(self.cfg.diarization_max_speakers))
        f.addRow("Max speakers:", self.max_spk)
        # Diarization engine
        self.diar_engine = QComboBox()
        self.diar_engine.addItems(["Disabled (Whisper only)", "SpeechBrain ECAPA (best)", "WeSpeaker ONNX (light)"])
        eng_to_idx = {"none": 0, "speechbrain": 1, "wespeaker": 2}
        self.diar_engine.setCurrentIndex(eng_to_idx.get(self.cfg.diarization_engine, 1))
        f.addRow("Diarization engine:", self.diar_engine)
        # Device override
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "gpu", "cpu"])
        self.device_combo.setCurrentText(self.cfg.device_mode)
        f.addRow("Device:", self.device_combo)
        # Number of workers
        self.num_workers_combo = QComboBox()
        self.num_workers_combo.addItems(["1 (safest)", "2", "3", "4 (fastest)"])
        self.num_workers_combo.setCurrentIndex(self.cfg.num_workers - 1)
        self.num_workers_combo.setToolTip("Number of parallel CTranslate2 workers. 1 is safest; higher values may be faster but can conflict with Qt on some systems.")
        f.addRow("Worker threads:", self.num_workers_combo)
        # Parallel mode hint (visible when device=cpu and workers>1)
        self.parallel_hint = QLabel("Disabled")
        f.addRow("Parallel:", self.parallel_hint)
        return w

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select WAV files", app_root(), "WAV files (*.wav)")
        for fp in files:
            self.file_list.addItem(QListWidgetItem(fp))

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", app_root())
        if folder:
            for name in os.listdir(folder):
                if name.lower().endswith('.wav'):
                    self.file_list.addItem(QListWidgetItem(os.path.join(folder, name)))

    def remove_selected_files(self):
        """Remove selected file(s) from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", app_root())
        if folder:
            self.output_dir_edit.setText(folder)

    def start_transcription(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            self.progress_label.setText("⚠️ Please add WAV files first.")
            return

        # Persist current settings
        self.cfg.language_hint = self.lang_combo.currentData() or "en"
        self.cfg.word_timestamps = self.chk_word_ts.isChecked()
        self.cfg.diarization_max_speakers = int(self.max_spk.currentText())
        idx = self.diar_engine.currentIndex()
        if idx == 0:
            self.cfg.diarization_engine = 'none'
        elif idx == 1:
            self.cfg.diarization_engine = 'speechbrain'
        else:
            self.cfg.diarization_engine = 'wespeaker'
        self.cfg.device_mode = self.device_combo.currentText()
        self.cfg.output_dir = self.output_dir_edit.text().strip()
        # Parse num_workers from combo (text is like "1 (safest)" or "4 (fastest)")
        workers_text = self.num_workers_combo.currentText()
        self.cfg.num_workers = int(workers_text.split()[0])
        fmts = []
        if self.chk_txt.isChecked():
            fmts.append("txt")
        if self.chk_docx.isChecked():
            fmts.append("docx")
        self.cfg.output_formats = fmts or ["txt"]
        self.cfg.save()

        model_name = self.model_combo.currentText()
        if not model_name or model_name.startswith("— "):
            self.progress_label.setText("⚠️ Please select a valid model.")
            return
        # Inform users when selection will trigger a download
        model_dir = os.path.join(models_root(), model_name)
        if not os.path.isdir(model_dir):
            self.progress_label.setText(
                f"⚠️ Model '{model_name}' not found locally. "
                f"Auto-download may hang on Windows. "
                f"Please manually download to: {models_root()} "
                f"or use an already downloaded model."
            )

        # Cap workers for very large models and warn if memory appears low
        try:
            model_lower = model_name.lower()
            # Heuristic cap for large models
            if "large" in model_lower and self.cfg.num_workers > 2:
                self.cfg.num_workers = 2
                try:
                    self.num_workers_combo.setCurrentIndex(self.cfg.num_workers - 1)
                except Exception:
                    pass
                self.progress_label.setText("ℹ️ Capping workers to 2 for large models to avoid memory pressure.")
            # Best-effort low-memory detection (psutil optional)
            try:
                import psutil  # type: ignore
                avail_gb = psutil.virtual_memory().available / (1024 ** 3)
                if avail_gb < 8 and self.cfg.num_workers > 1:
                    self.cfg.num_workers = 1
                    try:
                        self.num_workers_combo.setCurrentIndex(self.cfg.num_workers - 1)
                    except Exception:
                        pass
                    self.progress_label.setText("ℹ️ Low memory detected (<8 GB available). Using single worker to improve stability.")
            except Exception:
                # psutil not present or check failed; ignore
                pass
            # Update the Advanced tab hint label
            try:
                self._update_parallel_hint()
            except Exception:
                pass
        except Exception:
            pass

        self.btn_transcribe.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Reset counters
        self.total_files = len(files)
        self.completed_count = 0
        self.error_count = 0
        self._init_failed = False

        # Show ongoing processing state
        try:
            self.progress_label.setText(f"Processing… 0/{self.total_files}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(1)
        except Exception:
            pass

        self.worker = Worker(files, self.cfg, model_name)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.device_info.connect(self.on_device_info)
        self.worker.start()

    def stop_transcription(self):
        """Immediately stop the current transcription worker."""
        if self.worker:
            self.worker.cancelled = True
            self.progress_label.setText("⚠️ Stopping… please wait for current file to finish.")
            logging.info("User requested stop.")

    def open_output_folder(self):
        """Open the output folder where transcriptions are saved."""
        try:
            output_dir = self.output_dir_edit.text().strip()
            if not output_dir:
                # If no output dir configured, use the first file's directory or app root
                if self.file_list.count() > 0:
                    first_file = self.file_list.item(0).text()
                    output_dir = os.path.dirname(os.path.abspath(first_file))
                    logging.info("[FileSystem] Using first file's directory as output: %s", output_dir)
                else:
                    output_dir = app_root()
                    logging.info("[FileSystem] Using app_root as output: %s", output_dir)
            else:
                # Make absolute if relative
                if not os.path.isabs(output_dir):
                    output_dir = os.path.abspath(output_dir)
                logging.info("[FileSystem] Using configured output directory: %s", output_dir)

            # Create directory if it doesn't exist
            logging.info("[FileSystem] Output directory exists: %s", os.path.exists(output_dir))
            logging.info("[FileSystem] Output directory is dir: %s", os.path.isdir(output_dir))

            if not os.path.isdir(output_dir):
                logging.info("[FileSystem] Creating output directory: %s", output_dir)
                os.makedirs(output_dir, exist_ok=True)
                logging.info("[FileSystem] Output directory created successfully")

            logging.info("[FileSystem] Output directory is writable: %s", os.access(output_dir, os.W_OK))
            # Use Qt to open folder cross-platform
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
        except Exception as e:
            logging.error("[FileSystem] Failed to open output folder: %s", str(e), exc_info=True)
            self.progress_label.setText(f"⚠️ Could not open output folder: {str(e)}")

    def open_logs(self):
        try:
            log_dir = os.path.dirname(self.log_path) if getattr(self, 'log_path', None) else os.path.join(app_root(), 'logs')
            logging.info("[FileSystem] Opening log directory: %s", log_dir)
            logging.info("[FileSystem] Log directory exists: %s", os.path.exists(log_dir))

            if not os.path.isdir(log_dir):
                logging.info("[FileSystem] Creating log directory: %s", log_dir)
                os.makedirs(log_dir, exist_ok=True)
                logging.info("[FileSystem] Log directory created successfully")

            # Use Qt to open folder cross-platform
            QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir))
        except Exception as e:
            logging.error("[FileSystem] Failed to open log folder: %s", str(e), exc_info=True)

    def on_progress_update(self, msg: str, pct: int):
        self.progress_label.setText(msg)
        # Use indeterminate progress bar (busy indicator) when percentage is 0
        # This is useful during model download which has no progress tracking
        if pct == 0:
            self.progress_bar.setRange(0, 0)  # Indeterminate/busy mode
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            if pct > 0 and pct < 100:
                self.progress_bar.setVisible(True)
            elif pct >= 100:
                self.progress_bar.setVisible(False)

    def on_device_info(self, text: str):
        self.device_runtime_label.setText(f"Runtime device: {text}")

    def on_file_done(self, wav):
        logging.info("Done: %s", wav)
        # Count success
        try:
            self.completed_count += 1
        except Exception:
            pass
        # Show ongoing processing aggregate instead of per-file "Completed"
        try:
            self.progress_label.setText(f"Processing… {self.completed_count + self.error_count}/{self.total_files}")
        except Exception:
            pass
        # Update overall progress when running multiple files
        try:
            if self.total_files > 0:
                pct = int(100 * (self.completed_count + self.error_count) / self.total_files)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(min(max(pct, 1), 99))
                self.progress_bar.setVisible(pct < 100)
        except Exception:
            pass

    def on_file_error(self, wav, err):
        logging.error("Error on %s: %s", wav, err)
        # Count failure; also mark init failure specially
        try:
            self.error_count += 1
            if str(wav) == "<init>":
                self._init_failed = True
        except Exception:
            pass
        # Show ongoing processing aggregate with error
        try:
            self.progress_label.setText(f"Processing… {self.completed_count + self.error_count}/{self.total_files} — last error: {os.path.basename(wav)}")
        except Exception:
            pass
        # Update overall progress when running multiple files
        try:
            if self.total_files > 0:
                pct = int(100 * (self.completed_count + self.error_count) / self.total_files)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(min(max(pct, 1), 99))
                self.progress_bar.setVisible(pct < 100)
        except Exception:
            pass

    def on_all_done(self):
        self.btn_transcribe.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Summarize results with guidance to check logs when needed
        if self._init_failed:
            self.progress_label.setText("❌ Failed to initialize the model. No transcriptions were created. Please check the logs (use the 'Open Logs' button).")
        elif self.error_count == 0 and self.completed_count == self.total_files and self.total_files > 0:
            self.progress_label.setText("✅ All transcriptions completed successfully!")
        elif self.total_files > 0:
            self.progress_label.setText(
                f"⚠️ Completed {self.completed_count} of {self.total_files}. {self.error_count} failed. Please check the logs (use the 'Open Logs' button)."
            )
        else:
            # No files scenario shouldn't normally reach here, but keep a safe default
            self.progress_label.setText("ℹ️ No files were processed.")
        self.progress_bar.setVisible(False)


def main():
    """Entry point for the WhisperDesk application."""
    # Set HuggingFace environment variables BEFORE any imports that might use them
    # This must be done at app startup, not in worker threads
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

    # Route HF cache to a per-user, writable location (fixes Windows frozen-app hangs)
    try:
        _hf_cache = hf_cache_root()
        os.environ['HF_HOME'] = _hf_cache
        os.environ['HUGGINGFACE_HUB_CACHE'] = _hf_cache
        logging.info("[Startup] HF cache configured: HF_HOME=%s", _hf_cache)
    except Exception as e:
        logging.error("[Startup] Failed to configure HF cache: %s", str(e), exc_info=True)

    # CRITICAL FIX for Windows PyInstaller builds:
    # Force httpx to use HTTP/1.1 instead of HTTP/2 which causes hangs on Windows
    # Also disable connection pooling which conflicts with Qt event loop
    if sys.platform == 'win32':
        logging.info("[Startup] Detected Windows platform, applying Windows-specific settings")
        os.environ['HTTPX_DISABLE_HTTP2'] = '1'
        # Use requests library instead of httpx for downloads (more compatible with PyInstaller)
        os.environ['HF_HUB_ENABLE_REQUESTS'] = '1'
        # Disable symlink usage on Windows to avoid permission and AV issues
        os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
        # Timeouts and telemetry
        os.environ.setdefault('HF_HUB_HTTP_TIMEOUT', '60')
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_ENABLE_TQDM'] = '0'
        logging.info("[Startup] Windows HTTP/download settings: HTTP/2 disabled, symlinks disabled, timeout=60s")

        # Ensure SSL certificate bundle is available to requests/httpx
        try:
            import certifi  # type: ignore
            ca_file = certifi.where()
            os.environ['SSL_CERT_FILE'] = ca_file
            os.environ['REQUESTS_CA_BUNDLE'] = ca_file
            # Force httpx to use certifi bundle (critical for PyInstaller)
            os.environ['HTTPX_VERIFY'] = ca_file
            logging.info("[Startup] SSL certificates configured from certifi: %s", ca_file)
            logging.info("[FileSystem] SSL cert file exists: %s", os.path.exists(ca_file))
        except Exception as e:
            logging.error("[Startup] Failed to configure SSL certificates: %s", str(e), exc_info=True)

    # Log critical paths for debugging
    logging.info("[Startup] app_root: %s", app_root())
    logging.info("[Startup] models_root: %s", models_root())
    logging.info("[FileSystem] app_root exists: %s", os.path.exists(app_root()))
    logging.info("[FileSystem] models_root exists: %s", os.path.exists(models_root()))
    logging.info("[FileSystem] Current working directory: %s", os.getcwd())

    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        logging.info("[Startup] Running as PyInstaller bundle, _MEIPASS: %s", _meipass)
    else:
        logging.info("[Startup] Running as normal Python script")

    setup_logging()    # Enable HiDPI support for sharp rendering on high-resolution displays
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Use Fusion style for consistent cross-platform appearance
    app.setStyle("Fusion")

    # Set professional color palette
    palette = QPalette()
    # Light mode palette suitable for professional/academic environments
    palette.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(220, 53, 69))
    palette.setColor(QPalette.ColorRole.Link, QColor(13, 110, 253))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(13, 110, 253))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Set professional font - use system default
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    # Apply modern stylesheet - VS Code / Notion inspired
    stylesheet = """
        /* Global Styles */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
        }

        /* Main Window */
        QWidget {
            background-color: #f5f5f5;
            color: #1e1e1e;
        }

        /* Cards - Framed panels with subtle shadows */
        QFrame#card {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            /* Add subtle shadow for better separation */
        }

        /* Add subtle inner shadow effect for better visual separation */
        QFrame#card:focus {
            border: 1px solid #d0d0d0;
        }

        /* Typography */
        QLabel#cardTitle {
            font-size: 14px;
            font-weight: 600;
            color: #1e1e1e;
            padding: 0px;
        }

        QLabel#subtitle {
            font-size: 12px;
            font-weight: 500;
            color: #424242;
        }

        QLabel#body {
            font-size: 13px;
            /* color: #424242; */
        }

        QLabel#caption {
            font-size: 11px;
            color: #757575;
        }

        QLabel {
            color: #424242;
            padding: 2px;
            background-color: transparent;
        }

        /* Tab Widget */
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            background-color: #ffffff;
            padding: 8px;
            top: -1px;
        }

        QTabBar::tab {
            background-color: transparent;
            color: #757575;
            padding: 8px 16px;
            margin-right: 4px;
            border: none;
            border-bottom: 2px solid transparent;
            font-weight: 500;
        }

        QTabBar::tab:selected {
            color: #1e1e1e;
            border-bottom: 2px solid #0078d4;
        }

        QTabBar::tab:hover:!selected {
            color: #424242;
            background-color: rgba(0, 0, 0, 0.02);
        }

        /* Primary Button */
        QPushButton#btn_primary {
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
            min-height: 28px;
        }

        QPushButton#btn_primary:hover {
            background-color: #106ebe;
        }

        QPushButton#btn_primary:pressed {
            background-color: #005a9e;
        }

        QPushButton#btn_primary:disabled {
            background-color: #e0e0e0;
            color: #9e9e9e;
        }

        /* Danger Button */
        QPushButton#btn_danger {
            background-color: #d32f2f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
            min-height: 28px;
        }

        QPushButton#btn_danger:hover {
            background-color: #c62828;
        }

        QPushButton#btn_danger:pressed {
            background-color: #b71c1c;
        }

        QPushButton#btn_danger:disabled {
            background-color: #e0e0e0;
            color: #9e9e9e;
        }

        /* Secondary Button */
        QPushButton#btn_secondary, QPushButton#btn_icon {
            background-color: #ffffff;
            color: #424242;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: 500;
            min-height: 28px;
        }

        QPushButton#btn_secondary:hover, QPushButton#btn_icon:hover {
            background-color: #f5f5f5;
            border-color: #bdbdbd;
        }

        QPushButton#btn_secondary:pressed, QPushButton#btn_icon:pressed {
            background-color: #eeeeee;
        }

        QPushButton#btn_secondary:disabled, QPushButton#btn_icon:disabled {
            background-color: #fafafa;
            color: #9e9e9e;
            border-color: #e0e0e0;
        }

        /* Default Buttons */
        QPushButton {
            background-color: #ffffff;
            color: #424242;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 4px 6px;
            font-weight: 500;
            min-height: 28px;
        }

        QPushButton:hover {
            background-color: #f5f5f5;
            border-color: #bdbdbd;
        }

        QPushButton:pressed {
            background-color: #eeeeee;
        }

        QPushButton:disabled {
            background-color: #fafafa;
            color: #9e9e9e;
            border-color: #e0e0e0;
        }

        /* Line Edit */
        QLineEdit {
            padding: 2px 4px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background-color: #ffffff;
            selection-background-color: #cce4f7;
            color: #1e1e1e;
            min-height: 32px;
        }

        QLineEdit:focus {
            border: 1px solid #0078d4;
            background-color: #ffffff;
        }

        QLineEdit:hover {
            border-color: #bdbdbd;
        }

        /* Combo Box */
        QComboBox {
            padding: 2px 6px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background-color: #ffffff;
            color: #1e1e1e;
            min-height: 32px;
        }

        QComboBox:focus {
            border: 1px solid #0078d4;
        }

        QComboBox:hover {
            border-color: #bdbdbd;
        }

        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #757575;
            margin-right: 8px;
        }

        QComboBox QAbstractItemView {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background-color: #ffffff;
            selection-background-color: #e3f2fd;
            selection-color: #1e1e1e;
            padding: 4px;
            outline: none;
        }

        /* List Widget */
        QListWidget {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background-color: #ffffff;
            padding: 4px;
            outline: none;
        }

        QListWidget::item {
            padding: 10px 12px;
            border-radius: 4px;
            margin: 2px 0px;
            color: #1e1e1e;
            border: none;
        }

        QListWidget::item:selected {
            background-color: #e3f2fd;
            color: #0078d4;
        }

        QListWidget::item:hover:!selected {
            background-color: #f5f5f5;
        }

        /* Check Box */
        QCheckBox {
            spacing: 8px;
            padding: 4px;
            color: #1e1e1e;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #bdbdbd;
            border-radius: 4px;
            background-color: #ffffff;
        }

        QCheckBox::indicator:hover {
            border-color: #0078d4;
        }

        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border-color: #0078d4;
            image: none;
        }

        /* Progress Bar */
        QProgressBar {
            border: none;
            border-radius: 4px;
            text-align: center;
            background-color: #e0e0e0;
            height: 6px;
            color: transparent;
        }

        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 4px;
        }

        /* Scrollbar */
        QScrollBar:vertical {
            border: none;
            background-color: transparent;
            width: 10px;
            margin: 0px;
        }

        QScrollBar::handle:vertical {
            background-color: #bdbdbd;
            border-radius: 5px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #9e9e9e;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }

        QScrollBar:horizontal {
            border: none;
            background-color: transparent;
            height: 10px;
            margin: 0px;
        }

        QScrollBar::handle:horizontal {
            background-color: #bdbdbd;
            border-radius: 5px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #9e9e9e;
        }

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }

        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }

        /* Form Layout Spacing */
        QFormLayout {
            spacing: 8px;
        }
    """

    app.setStyleSheet(stylesheet)
    app.setQuitOnLastWindowClosed(True)

    w = MainWindow()
    w.setWindowTitle("WhisperDesk")
    w.setMinimumSize(1200, 800)
    w.show()

    exit_code = app.exec()
    logging.info("Application exiting with code %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
