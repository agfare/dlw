import json
from pathlib import Path

from .docx_reader import read_docx_chunks
from .classifier import Tagger


def tag_file(
    docx_path: str,
    output_path: str,
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
) -> list[dict]:
    chunks = read_docx_chunks(docx_path)
    tagger = Tagger(model_name=model_name, device=device)
    tagged = tagger.tag_chunks(chunks, source_file=docx_path)
    result = [t.to_dict() for t in tagged]
    Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def tag_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
    glob: str = "**/*.docx",
) -> dict[str, list[dict]]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = {}
    for docx_file in sorted(input_path.glob(glob)):
        out_file = output_path / (docx_file.stem + ".json")
        results[str(docx_file)] = tag_file(
            str(docx_file), str(out_file), model_name=model_name, device=device
        )
    return results
