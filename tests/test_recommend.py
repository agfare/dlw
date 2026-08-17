import json
import tempfile
from pathlib import Path

import pytest

from recommend.engine import RecommendationEngine, UserState


FAKE_CHUNKS = [
    {
        "chunk_id": "doc_0",
        "source_file": "doc.docx",
        "text_preview": "Overview of the cardiovascular system.",
        "type": "article",
        "domain": "part",
        "system": ["cardiovascular"],
        "location": ["heart"],
        "chunk_type": "overview",
        "complexity": "general",
        "audience": "general",
        "scores": {},
    },
    {
        "chunk_id": "doc_1",
        "source_file": "doc.docx",
        "text_preview": "Diagnosis of heart conditions.",
        "type": "article",
        "domain": "pathology",
        "system": ["cardiovascular"],
        "location": ["heart"],
        "chunk_type": "diagnosis",
        "complexity": "intermediate",
        "audience": "doctor",
        "scores": {},
    },
    {
        "chunk_id": "doc_2",
        "source_file": "doc.docx",
        "text_preview": "Treatment options for lung disease.",
        "type": "article",
        "domain": "pathology",
        "system": ["respiratory"],
        "location": ["lungs"],
        "chunk_type": "treatment",
        "complexity": "advanced",
        "audience": "doctor",
        "scores": {},
    },
]


@pytest.fixture()
def engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "doc.json").write_text(json.dumps(FAKE_CHUNKS))
        yield RecommendationEngine(tmpdir)


def test_cold_start_recommends_overview(engine):
    state = engine.cold_start()
    recs = engine.recommend(state)
    assert recs[0]["chunk_id"] == "doc_0"


def test_system_filter_ranks_matching_system_first(engine):
    state = engine.cold_start("cardiovascular")
    recs = engine.recommend(state)
    assert recs[0]["system"] == ["cardiovascular"]


def test_advance_state_progresses_chunk_type(engine):
    state = engine.cold_start()
    assert state.chunk_type == "overview"
    engine.advance_state(state, "doc_0")
    assert state.chunk_type == "diagnosis"


def test_seen_chunks_excluded(engine):
    state = engine.cold_start()
    state.history.append("doc_0")
    recs = engine.recommend(state)
    assert all(r["chunk_id"] != "doc_0" for r in recs)


def test_no_crash_on_empty_history(engine):
    state = UserState()
    recs = engine.recommend(state)
    assert isinstance(recs, list)
