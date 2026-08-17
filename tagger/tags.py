from dataclasses import dataclass, field
from typing import Optional

TAXONOMY = {
    "type": ["model", "article"],
    "domain": ["pathology", "part"],
    "system": [
        "cardiovascular", "neurological", "respiratory", "digestive",
        "musculoskeletal", "endocrine", "immune", "reproductive",
        "urinary", "integumentary",
    ],
    "location": [
        "brain", "back", "lungs", "heart", "liver", "kidney", "spine",
        "chest", "abdomen", "neck", "shoulder", "knee", "hip", "pelvis", "head",
    ],
    "chunk_type": ["overview", "diagnosis", "treatment", "faq"],
    "complexity": ["general", "introductory", "intermediate", "advanced"],
    "audience": ["student", "doctor", "patient", "general"],
}

HYPOTHESIS_TEMPLATES = {
    "type":       "This document is a {}.",
    "domain":     "This text covers {}.",
    "system":     "This text is about the {} system.",
    "location":   "This text focuses on the {}.",
    "chunk_type": "This passage is an {}.",
    "complexity": "This content is {} level.",
    "audience":   "This text is written for a {}.",
}

COLD_START_DEFAULTS = {
    "chunk_type": "overview",
    "complexity": "general",
    "audience":   "general",
}


@dataclass
class ChunkTags:
    chunk_id: str
    source_file: str
    text_preview: str
    type: Optional[str] = None
    domain: Optional[str] = None
    system: list = field(default_factory=list)
    location: list = field(default_factory=list)
    chunk_type: Optional[str] = None
    complexity: Optional[str] = None
    audience: Optional[str] = None
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "source_file": self.source_file,
            "text_preview": self.text_preview,
            "type":        self.type,
            "domain":      self.domain,
            "system":      self.system,
            "location":    self.location,
            "chunk_type":  self.chunk_type,
            "complexity":  self.complexity,
            "audience":    self.audience,
            "scores":      self.scores,
        }
