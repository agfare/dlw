from pathlib import Path
from transformers import pipeline

from .tags import TAXONOMY, HYPOTHESIS_TEMPLATES, ChunkTags

_MULTI_LABEL_CATEGORIES = {"system", "location"}


class Tagger:
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )

    def tag_chunk(self, text: str, chunk_id: str, source_file: str) -> ChunkTags:
        tags = ChunkTags(
            chunk_id=chunk_id,
            source_file=source_file,
            text_preview=text[:120],
        )
        scores_map = {}
        for category, labels in TAXONOMY.items():
            template = HYPOTHESIS_TEMPLATES[category]
            multi = category in _MULTI_LABEL_CATEGORIES
            result = self._pipe(
                text,
                candidate_labels=labels,
                hypothesis_template=template,
                multi_label=multi,
            )
            label_scores = dict(zip(result["labels"], result["scores"]))
            scores_map[category] = label_scores
            if multi:
                # keep labels scoring above 0.5
                selected = [l for l, s in label_scores.items() if s >= 0.5]
                setattr(tags, category, selected)
            else:
                setattr(tags, category, result["labels"][0])
        tags.scores = scores_map
        return tags

    def tag_chunks(self, chunks: list[str], source_file: str) -> list[ChunkTags]:
        stem = Path(source_file).stem
        return [
            self.tag_chunk(text, f"{stem}_{i}", source_file)
            for i, text in enumerate(chunks)
        ]
