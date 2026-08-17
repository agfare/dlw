import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tagger.tags import COLD_START_DEFAULTS

_CHUNK_TYPE_PROGRESSION = ["overview", "diagnosis", "treatment", "faq"]
_COMPLEXITY_PROGRESSION = ["general", "introductory", "intermediate", "advanced"]

_SCORE_WEIGHTS = {
    "chunk_type": 3.0,
    "system":     2.0,
    "complexity": 1.5,
    "audience":   1.5,
}


@dataclass
class UserState:
    system: Optional[str] = None
    chunk_type: str = COLD_START_DEFAULTS["chunk_type"]
    complexity: str = COLD_START_DEFAULTS["complexity"]
    audience: str = COLD_START_DEFAULTS["audience"]
    history: list = field(default_factory=list)


class RecommendationEngine:
    def __init__(self, tagged_dir: str):
        self._chunks: list[dict] = []
        self._load_tagged_dir(tagged_dir)

    def _load_tagged_dir(self, tagged_dir: str) -> None:
        for json_file in sorted(Path(tagged_dir).glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._chunks.extend(data)

    def _score_chunk(self, chunk: dict, state: UserState) -> float:
        if chunk["chunk_id"] in state.history:
            return -1.0

        score = 0.0

        if chunk.get("chunk_type") == state.chunk_type:
            score += _SCORE_WEIGHTS["chunk_type"]

        chunk_systems = chunk.get("system") or []
        if state.system and state.system in chunk_systems:
            score += _SCORE_WEIGHTS["system"]

        if chunk.get("complexity") == state.complexity:
            score += _SCORE_WEIGHTS["complexity"]

        if chunk.get("audience") == state.audience:
            score += _SCORE_WEIGHTS["audience"]

        return score

    def recommend(self, state: UserState, top_k: int = 5) -> list[dict]:
        scored = [
            (self._score_chunk(chunk, state), chunk)
            for chunk in self._chunks
        ]
        scored = [(s, c) for s, c in scored if s >= 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def advance_state(self, state: UserState, viewed_chunk_id: str) -> UserState:
        state.history.append(viewed_chunk_id)

        viewed = next((c for c in self._chunks if c["chunk_id"] == viewed_chunk_id), None)
        if viewed is None:
            return state

        current_ct = viewed.get("chunk_type")
        if current_ct in _CHUNK_TYPE_PROGRESSION:
            idx = _CHUNK_TYPE_PROGRESSION.index(current_ct)
            if idx + 1 < len(_CHUNK_TYPE_PROGRESSION):
                state.chunk_type = _CHUNK_TYPE_PROGRESSION[idx + 1]

        current_cx = viewed.get("complexity")
        if current_cx in _COMPLEXITY_PROGRESSION:
            idx = _COMPLEXITY_PROGRESSION.index(current_cx)
            if idx + 1 < len(_COMPLEXITY_PROGRESSION):
                state.complexity = _COMPLEXITY_PROGRESSION[idx + 1]

        return state

    def cold_start(self, last_opened_system: Optional[str] = None) -> UserState:
        state = UserState()
        if last_opened_system:
            state.system = last_opened_system
        return state
