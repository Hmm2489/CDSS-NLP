"""Builds the medSpaCy pipeline.

Component order matters:
  1. medspacy_pyrush         sentence splitting (ConText scopes stop at sentence ends)
  2. medspacy_target_matcher lay-language lexicon -> entities with known CUIs
  3. medspacy_quickumls      fuzzy UMLS matching; skips spans the lexicon already claimed
  4. medspacy_context        negation / uncertainty / history / family / hypothetical
"""

from __future__ import annotations

import json
from pathlib import Path

import medspacy
from loguru import logger
from medspacy.context import ConTextRule
from medspacy.target_matcher import TargetRule
from spacy.language import Language

from medinfer.config import load_config, resolve

# PyRuSH logs every token at DEBUG level through loguru.
logger.disable("PyRuSH")

SYMPTOM_LABEL = "SYMPTOM"


def load_lexicon_rules(path: Path) -> list[TargetRule]:
    with open(path) as f:
        concepts = json.load(f)["concepts"]
    return [
        TargetRule(term, SYMPTOM_LABEL, metadata={"cui": c["cui"], "name": c["name"]})
        for c in concepts
        for term in c["terms"]
    ]


def build_nlp(config: dict | None = None) -> Language:
    config = config or load_config("pipeline")
    quickumls_path = resolve(config.get("quickumls_path"))

    components = ["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"]
    if quickumls_path is not None:
        components.insert(2, "medspacy_quickumls")

    # Add QuickUMLS ourselves below so we can pass the full config.
    nlp = medspacy.load(medspacy_enable=[c for c in components if c != "medspacy_quickumls"])

    nlp.get_pipe("medspacy_target_matcher").add(
        load_lexicon_rules(resolve(config["lexicon_path"]))
    )

    if quickumls_path is not None:
        q = config["quickumls"]
        nlp.add_pipe(
            "medspacy_quickumls",
            before="medspacy_context",
            config={
                "quickumls_fp": str(quickumls_path),
                "threshold": q["threshold"],
                "similarity_name": q["similarity_name"],
                "window": q["window"],
                "best_match": q["best_match"],
                "accepted_semtypes": q["accepted_semtypes"],
            },
        )

    context_rules_path = resolve(config.get("context_rules_path"))
    if context_rules_path is not None:
        nlp.get_pipe("medspacy_context").add(ConTextRule.from_json(context_rules_path))

    return nlp
