import os
import sys
import logging
from typing import List, Dict

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QListWidget, QListWidgetItem, QComboBox, QCheckBox, QTabWidget,
    QGroupBox, QFormLayout, QMessageBox, QLineEdit, QAbstractItemView
)

from utils import setup_logging, models_root, app_root
from device import device_banner
from settings import Settings
from transcription import Transcriber
from diarization import diarize, assign_speakers_to_asr
from exporters import export_txt, export_docx


class Worker(QThread):
    file_done = Signal(str)
    file_error = Signal(str, str)
    all_done = Signal()

    def __init__(self, files: List[str], cfg: Settings, model_name: str, parent=None):
        super().__init__(parent)
        self.files = files
        self.cfg = cfg
        self.model_name = model_name
        self.cancelled = False

    def run(self):
        try:
            tr = Transcriber(
                model_name=self.model_name,
                device_mode=self.cfg.device_mode,
                language_hint=self.cfg.language_hint,
                word_timestamps=self.cfg.word_timestamps
            )
        except Exception as e:
            self.file_error.emit("<init>", str(e))
            self.all_done.emit()
            return

        for wav in self.files:
            if self.cancelled:
                break
            try:
                asr = tr.transcribe(wav)
                diar = diarize(wav, max_speakers=self.cfg.diarization_max_speakers)
                segs = assign_speakers_to_asr(asr["segments"], diar)

                # default speaker map
                spk_ids = sorted(set([s.get("speaker", 0) for s in segs]))
                speaker_map = {i: f"Speaker {i+1}" for i in spk_ids}

                base = os.path.splitext(wav)[0]
                out_txt = base + ".txt"
                out_docx = base + ".docx"
                if "txt" in self.cfg.output_formats:
                    export_txt(out_txt, segs, speaker_map)
                if "docx" in self.cfg.output_formats:
                    export_docx(out_docx, segs, speaker_map)

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
        log_path = setup_logging(self.cfg.log_level)

        v = QVBoxLayout(self)

        # Status banner (device, RAM/VRAM)
        self.status_label = QLabel(device_banner(is_gpu=(self.cfg.device_mode != 'cpu')))
        v.addWidget(self.status_label)

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
        self.model_combo.addItems(self._available_models())
        # choose preference if available
        pref = f"whisper-{self.cfg.model_preference}-ct2"
        if pref in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(pref)
        controls.addWidget(self.model_combo)

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
        if not os.path.isdir(root):
            return []
        return sorted([d for d in os.listdir(root) if d.startswith('whisper-')])

    def _build_easy_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        # Language hint
        self.lang_edit = QLineEdit(self.cfg.language_hint)
        f.addRow("Language hint:", self.lang_edit)
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
        # Device override
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "gpu", "cpu"])
        self.device_combo.setCurrentText(self.cfg.device_mode)
        f.addRow("Device:", self.device_combo)
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

    def start_transcription(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.information(self, "WhisperDesk", "Add WAV files first.")
            return

        # Persist current settings
        self.cfg.language_hint = self.lang_edit.text().strip() or "en"
        self.cfg.word_timestamps = self.chk_word_ts.isChecked()
        self.cfg.diarization_max_speakers = int(self.max_spk.currentText())
        self.cfg.device_mode = self.device_combo.currentText()
        fmts = []
        if self.chk_txt.isChecked():
            fmts.append("txt")
        if self.chk_docx.isChecked():
            fmts.append("docx")
        self.cfg.output_formats = fmts or ["txt"]
        self.cfg.save()

        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "WhisperDesk", "No models found in the models/ folder.")
            return

        self.btn_transcribe.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = Worker(files, self.cfg, model_name)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.start()

    def cancel_transcription(self):
        if self.worker:
            self.worker.cancelled = True

    def on_file_done(self, wav):
        logging.info("Done: %s", wav)

    def on_file_error(self, wav, err):
        logging.error("Error on %s: %s", wav, err)
        QMessageBox.critical(self, "WhisperDesk", f"Error on {os.path.basename(wav)}:\n{err}")

    def on_all_done(self):
        self.btn_transcribe.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.information(self, "WhisperDesk", "All done.")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
