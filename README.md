# Document Tagger & Recommendation Engine

Zero-shot classification of medical and educational `.docx` files into a fixed taxonomy, paired with a rule-based recommendation engine with cold-start support.

**Stack:** Python 3.10+ · `transformers` · `facebook/bart-large-mnli` · no fine-tuning required

---

## How it works

1. **Chunk** — each `.docx` is split into ~800-character passages by merging paragraphs.
2. **Tag** — every chunk is classified across seven categories using `facebook/bart-large-mnli` zero-shot inference.
3. **Store** — tags are written to a `.json` file alongside confidence scores for every label.
4. **Recommend** — the engine scores chunks against a user's current state, surfaces the top-k, and advances the state after each view.

---

## Installation

```bash
pip install -r requirements.txt
```

The first run downloads `facebook/bart-large-mnli` (~1.6 GB). GPU is optional — pass `--device 0` to use it.

---

## Usage

### Tag a single file

```bash
python tag_documents.py --file path/to/doc.docx --output doc.json
```

### Tag a directory

```bash
python tag_documents.py --dir ./docs --output-dir ./tagged
```

### Recommend in Python

```python
from recommend.engine import RecommendationEngine

engine = RecommendationEngine("./tagged")

# New user — last model they opened was cardiovascular
state  = engine.cold_start(last_opened_system="cardiovascular")
chunks = engine.recommend(state, top_k=5)

# After the user views a chunk, advance the state
engine.advance_state(state, chunks[0]["chunk_id"])
```

---

## Taxonomy

Every chunk is assigned one value per single-label category and zero or more values for multi-label categories (`system`, `location`).

| Category | Values |
|---|---|
| `type` | `model`, `article` |
| `domain` | `pathology`, `part` |
| `system` ✱ | `cardiovascular`, `neurological`, `respiratory`, `digestive`, `musculoskeletal`, `endocrine`, `immune`, `reproductive`, `urinary`, `integumentary` |
| `location` ✱ | `brain`, `back`, `lungs`, `heart`, `liver`, `kidney`, `spine`, `chest`, `abdomen`, `neck`, `shoulder`, `knee`, `hip`, `pelvis`, `head` |
| `chunk_type` | `overview`\*, `diagnosis`, `treatment`, `faq` |
| `complexity` | `general`\*, `introductory`, `intermediate`, `advanced` |
| `audience` | `general`\*, `student`, `doctor`, `patient` |

✱ multi-label &nbsp;·&nbsp; \* cold-start default

---

## Scoring

Chunks are ranked by additive score. Already-seen chunks are excluded.

| Category | Bonus | Notes |
|---|---|---|
| `chunk_type` | +3.0 | primary recommendation driver |
| `system` | +2.0 | set from last-opened model on cold start |
| `complexity` | +1.5 | advances along sequence after each view |
| `audience` | +1.5 | static per session |

State progression: `overview → diagnosis → treatment → faq` · `general → introductory → intermediate → advanced`

---

## Project structure

```
dlw/
├── tag_documents.py       # CLI entry point
├── requirements.txt
├── tagger/
│   ├── tags.py            # taxonomy, ChunkTags dataclass
│   ├── docx_reader.py     # .docx → text chunks
│   ├── classifier.py      # zero-shot Tagger class
│   └── pipeline.py        # tag_file / tag_directory helpers
├── recommend/
│   └── engine.py          # RecommendationEngine, UserState
└── tests/
    └── test_recommend.py  # unit tests (no ML model needed)
```

---

## Tests

```bash
pip install pytest
pytest tests/
```

Tests use fake tagged JSON and do not require the ML model or any `.docx` files.
