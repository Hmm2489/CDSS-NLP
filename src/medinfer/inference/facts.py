"""Turns extracted symptoms into initial working memory: {cui: cf}."""

from __future__ import annotations

from collections import defaultdict

from medinfer.inference.scoring import combine_all
from medinfer.schema import Symptom


def symptoms_to_evidence(symptoms: list[Symptom], config: dict) -> dict[str, float]:
    cfs = config["evidence_cf"]
    by_cui: dict[str, list[float]] = defaultdict(list)

    for s in symptoms:
        if any(getattr(s, flag) for flag in config["drop_if"]):
            continue
        if s.is_negated:
            cf = cfs["negated"]
        elif s.is_uncertain:
            cf = cfs["uncertain"]
        else:
            cf = cfs["present"]
        # Scale by match quality so fuzzy QuickUMLS hits count for less.
        by_cui[s.cui].append(cf * s.similarity)

    evidence = {}
    for cui, values in by_cui.items():
        # Repeat mentions shouldn't add up; contradictory ones ("no fever ... fever") partly cancel.
        evidence[cui] = max(values, key=abs) if _same_sign(values) else combine_all(values)
    return evidence


def _same_sign(values: list[float]) -> bool:
    return all(v >= 0 for v in values) or all(v <= 0 for v in values)
