"""Data types shared by the NLP and inference halves of the pipeline.

`Symptom` is the contract between them: extraction produces a list of these,
the inference engine consumes them. Keep it free of spaCy objects so either
side can be tested or swapped independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Symptom:
    cui: str              # UMLS Concept Unique Identifier, e.g. "C0015967"
    text: str             # surface form found in the input ("burning up")
    start: int            # character offsets into the original text
    end: int
    source: str           # "lexicon" or "quickumls"
    is_negated: bool = False
    is_uncertain: bool = False
    is_historical: bool = False
    is_hypothetical: bool = False
    is_family: bool = False
    similarity: float = 1.0   # QuickUMLS string similarity; 1.0 for exact lexicon hits

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Diagnosis:
    cui: str
    name: str
    cf: float                                       # certainty factor in [-1, 1]
    fired_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
