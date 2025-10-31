import os
import sys
import logging
from typing import List

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QListWidget, QListWidgetItem, QComboBox, QCheckBox, QTabWidget,
    QFormLayout, QLineEdit, QAbstractItemView, QProgressBar
)

from utils import setup_logging, models_root, app_root
from device import device_banner
from settings import Settings
from transcription import Transcriber
from diarization import diarize, assign_speakers_to_asr, ensure_ecapa_model
from exporters import export_txt, export_docx


class Worker(QThread):
    file_done = Signal(str)
    file_error = Signal(str, str)
    all_done = Signal()
    progress_update = Signal(str, int)  # (message, percentage 0-100)

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
            on_progress("Initializing model...", 0)
            # Set environment variable to avoid OpenMP conflicts with Qt
            os.environ['OMP_NUM_THREADS'] = '1'

            tr = Transcriber(
                model_name=self.model_name,
                device_mode=self.cfg.device_mode,
                language_hint=self.cfg.language_hint,
                word_timestamps=self.cfg.word_timestamps,
                num_workers=self.cfg.num_workers,
                progress_callback=on_progress
            )
            on_progress("Model ready.", 100)
            # Pre-fetch diarization model if user requested multi-speaker diarization and engine is enabled
            if self.cfg.diarization_max_speakers > 1 and self.cfg.diarization_engine != 'none':
                ensure_ecapa_model(progress_callback=on_progress)
        except Exception as e:
            import traceback
            logging.error("Model initialization failed: %s\n%s", e, traceback.format_exc())
            self.file_error.emit("<init>", str(e))
            self.all_done.emit()
            return

        for wav in self.files:
            if self.cancelled:
                break
            try:
                on_progress(f"Transcribing {os.path.basename(wav)}...", 0)
                asr = tr.transcribe(wav)
                # If diarization is disabled or single-speaker is selected, assign all to Speaker 1
                if self.cfg.diarization_engine == 'none' or self.cfg.diarization_max_speakers <= 1:
                    duration = float(asr.get("duration", 0.0))
                    diar = [(0.0, duration, 0)]
                else:
                    diar = diarize(wav, max_speakers=self.cfg.diarization_max_speakers, engine=self.cfg.diarization_engine)
                segs = assign_speakers_to_asr(asr["segments"], diar)

                # default speaker map
                spk_ids = sorted(set([s.get("speaker", 0) for s in segs]))
                speaker_map = {i: f"Speaker {i+1}" for i in spk_ids}

                # Determine if we should include speaker labels
                include_speakers = (
                    self.cfg.diarization_engine != 'none'
                    and self.cfg.diarization_max_speakers > 1
                    and len(spk_ids) > 1
                )

                # Determine output base path (respect configured output directory if set)
                out_dir = (self.cfg.output_dir or "").strip()
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
                model_short = self.model_name.replace("whisper-", "").replace("-ct2", "")
                base_name = f"{timestamp}_{original_name}_{model_short}"

                out_txt = os.path.join(out_dir, base_name + ".txt")
                out_docx = os.path.join(out_dir, base_name + ".docx")
                if "txt" in self.cfg.output_formats:
                    export_txt(out_txt, segs, speaker_map, include_speakers=include_speakers)
                if "docx" in self.cfg.output_formats:
                    export_docx(out_docx, segs, speaker_map, include_speakers=include_speakers)

                self.file_done.emit(wav)
            except Exception as e:
                self.file_error.emit(wav, str(e))
        self.all_done.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperDesk (Offline)")
        self.resize(800, 520)

        self.cfg = Settings()
        setup_logging(self.cfg.log_level)

        v = QVBoxLayout(self)

        # Status banner (device, RAM/VRAM)
        self.status_label = QLabel(device_banner(is_gpu=(self.cfg.device_mode != 'cpu')))
        v.addWidget(self.status_label)

        # Progress area
        self.progress_label = QLabel("Ready.")
        v.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        v.addWidget(self.progress_bar)

        # File list + controls
        self.file_list = QListWidget()
        # Use QAbstractItemView's enum for selection modes
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.file_list, 1)

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add WAV files…")
        self.btn_add_folder = QPushButton("Add folder…")
        self.btn_clear = QPushButton("Clear")
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addWidget(self.btn_clear)
        v.addLayout(btn_row)

        # Model + outputs row
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        # Build grouped model list with favorites
        self._rebuild_model_combo()
        # choose preference if available
        pref = f"whisper-{self.cfg.model_preference}-ct2"
        texts = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        if pref in texts:
            self.model_combo.setCurrentText(pref)
        # Favorite toggle button
        self.btn_fav = QPushButton("☆")
        self.btn_fav.setToolTip("Toggle favorite for selected model")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_fav.clicked.connect(self._toggle_favorite)
        controls.addWidget(self.model_combo)
        controls.addWidget(self.btn_fav)
        # Initialize star state
        self._on_model_changed(self.model_combo.currentText())

        self.chk_txt = QCheckBox("TXT")
        self.chk_txt.setChecked("txt" in self.cfg.output_formats)
        self.chk_docx = QCheckBox("DOCX")
        self.chk_docx.setChecked("docx" in self.cfg.output_formats)
        controls.addWidget(self.chk_txt)
        controls.addWidget(self.chk_docx)

        self.btn_transcribe = QPushButton("Transcribe")
        controls.addWidget(self.btn_transcribe)

        self.btn_cancel = QPushButton("Cancel remaining")
        self.btn_cancel.setEnabled(False)
        controls.addWidget(self.btn_cancel)

        v.addLayout(controls)

        # Tabs (Advanced)
        tabs = QTabWidget()
        tabs.addTab(self._build_easy_tab(), "Easy")
        tabs.addTab(self._build_advanced_tab(), "Advanced")
        v.addWidget(tabs)

        # Signals
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_clear.clicked.connect(self.file_list.clear)
        self.btn_transcribe.clicked.connect(self.start_transcription)
        self.btn_cancel.clicked.connect(self.cancel_transcription)

        self.worker = None

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

    def _build_easy_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        # Language hint
        self.lang_edit = QLineEdit(self.cfg.language_hint)
        f.addRow("Language hint:", self.lang_edit)
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
        self.cfg.language_hint = self.lang_edit.text().strip() or "en"
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
            self.progress_label.setText(f"ℹ️ Model '{model_name}' will be downloaded automatically on first use.")

        self.btn_transcribe.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = Worker(files, self.cfg, model_name)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.start()

    def cancel_transcription(self):
        if self.worker:
            self.worker.cancelled = True

    def on_progress_update(self, msg: str, pct: int):
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)
        if pct > 0 and pct < 100:
            self.progress_bar.setVisible(True)
        elif pct >= 100:
            self.progress_bar.setVisible(False)

    def on_file_done(self, wav):
        logging.info("Done: %s", wav)
        self.progress_label.setText(f"✅ Completed: {os.path.basename(wav)}")

    def on_file_error(self, wav, err):
        logging.error("Error on %s: %s", wav, err)
        self.progress_label.setText(f"❌ Error on {os.path.basename(wav)}: {err}")

    def on_all_done(self):
        self.btn_transcribe.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_label.setText("✅ All transcriptions completed successfully!")
        self.progress_bar.setVisible(False)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
