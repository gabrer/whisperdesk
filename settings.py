import os
from utils import app_root, load_json, save_json

_DEFAULTS = {
    "model_preference": "small",
    "language_hint": "en",
    "word_timestamps": False,
    "vad_sensitivity": "normal",
    "diarization_max_speakers": 3,
    "device_mode": "auto",
    "output_formats": ["txt", "docx"],
    "numbers_as_figures": True,
    "filter_profanity": False,
    "log_level": "INFO"
}


class Settings:
    def __init__(self):
        self.path = os.path.join(app_root(), 'presets.json')
        data = load_json(self.path, {"defaults": _DEFAULTS})
        defaults = data.get("defaults", {})
        self.model_preference = defaults.get("model_preference", _DEFAULTS["model_preference"])
        self.language_hint = defaults.get("language_hint", _DEFAULTS["language_hint"])
        self.word_timestamps = defaults.get("word_timestamps", _DEFAULTS["word_timestamps"])
        self.vad_sensitivity = defaults.get("vad_sensitivity", _DEFAULTS["vad_sensitivity"])
        self.diarization_max_speakers = int(defaults.get("diarization_max_speakers", _DEFAULTS["diarization_max_speakers"]))
        self.device_mode = defaults.get("device_mode", _DEFAULTS["device_mode"])  # auto|cpu|gpu
        self.output_formats = defaults.get("output_formats", _DEFAULTS["output_formats"])
        self.numbers_as_figures = defaults.get("numbers_as_figures", _DEFAULTS["numbers_as_figures"])
        self.filter_profanity = defaults.get("filter_profanity", _DEFAULTS["filter_profanity"])
        self.log_level = defaults.get("log_level", _DEFAULTS["log_level"])

    def to_dict(self):
        return {
            "defaults": {
                "model_preference": self.model_preference,
                "language_hint": self.language_hint,
                "word_timestamps": self.word_timestamps,
                "vad_sensitivity": self.vad_sensitivity,
                "diarization_max_speakers": self.diarization_max_speakers,
                "device_mode": self.device_mode,
                "output_formats": self.output_formats,
                "numbers_as_figures": self.numbers_as_figures,
                "filter_profanity": self.filter_profanity,
                "log_level": self.log_level
            }
        }

    def save(self):
        save_json(self.path, self.to_dict())
