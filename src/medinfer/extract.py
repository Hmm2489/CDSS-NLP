"""Public NLP entry point: free text -> list[Symptom]."""

from __future__ import annotations

from spacy.language import Language
from spacy.tokens import Span

from medinfer.nlp.pipeline import SYMPTOM_LABEL, build_nlp
from medinfer.nlp.preprocess import normalize
from medinfer.schema import Symptom


class SymptomExtractor:
    def __init__(self, nlp: Language | None = None):
        self.nlp = nlp or build_nlp()

    def __call__(self, text: str) -> list[Symptom]:
        doc = self.nlp(normalize(text))
        return [s for ent in doc.ents if (s := _to_symptom(ent)) is not None]


def _to_symptom(ent: Span) -> Symptom | None:
    if ent.label_ == SYMPTOM_LABEL:
        cui, source, similarity = ent._.target_rule.metadata["cui"], "lexicon", 1.0
    elif ent.label_.startswith("C") and ent.label_[1:].isdigit():
        # QuickUMLS sets the entity label to the CUI.
        cui, source, similarity = ent.label_, "quickumls", float(ent._.similarity)
    else:
        return None

    return Symptom(
        cui=cui,
        text=ent.text,
        start=ent.start_char,
        end=ent.end_char,
        source=source,
        is_negated=ent._.is_negated,
        is_uncertain=ent._.is_uncertain,
        is_historical=ent._.is_historical,
        is_hypothetical=ent._.is_hypothetical,
        is_family=ent._.is_family,
        similarity=similarity,
    )
