from pathlib import Path
from docx import Document


def iter_chunks(docx_path: str, target_chars: int = 800):
    doc = Document(docx_path)
    buffer = []
    length = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        buffer.append(text)
        length += len(text)
        if length >= target_chars:
            yield " ".join(buffer)
            buffer, length = [], 0
    if buffer:
        yield " ".join(buffer)


def read_docx_chunks(docx_path: str, target_chars: int = 800) -> list[str]:
    return list(iter_chunks(docx_path, target_chars))
