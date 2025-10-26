from typing import List, Dict, Any
from utils import format_ts


def export_txt(path: str, segments: List[Dict[str, Any]], speaker_map: Dict[int, str], include_speakers: bool = True):
    lines = []
    for seg in segments:
        timestamp = f"[{format_ts(seg['start'])} – {format_ts(seg['end'])}]"
        if include_speakers:
            spk_name = speaker_map.get(seg.get("speaker", 0), f"Speaker {seg.get('speaker', 0)+1}")
            lines.append(f"{timestamp} {spk_name}: {seg['text']}")
        else:
            lines.append(f"{timestamp} {seg['text']}")
    content = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content + "\n")


def export_docx(path: str, segments: List[Dict[str, Any]], speaker_map: Dict[int, str], include_speakers: bool = True):
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn

    doc = Document()
    doc.add_heading('Transcript', level=1)

    for seg in segments:
        p = doc.add_paragraph()
        run1 = p.add_run(f"[{format_ts(seg['start'])} – {format_ts(seg['end'])}] ")
        if include_speakers:
            spk_name = speaker_map.get(seg.get("speaker", 0), f"Speaker {seg.get('speaker', 0)+1}")
            run2 = p.add_run(f"{spk_name}: ")
            run2.bold = True
        p.add_run(seg['text'])

    doc.save(path)
