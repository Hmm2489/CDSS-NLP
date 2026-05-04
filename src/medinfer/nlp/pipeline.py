"""Builds the medSpaCy pipeline.

Component order matters:
  1. medspacy_pyrush         sentence splitting (ConText scopes stop at sentence ends)
  2. medspacy_target_matcher lay-language lexicon -> entities with known CUIs
  3. medspacy_quickumls      fuzzy UMLS matching; skips spans the lexicon already claimed
  4. medspacy_context        negation / uncertainty / history / family / hypothetical
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import medspacy
from loguru import logger
from medspacy.context import ConTextRule
from medspacy.target_matcher import TargetRule
from spacy.language import Language
from spacy.tokens import Span

from medinfer.config import load_config, resolve

# PyRuSH logs every token at DEBUG level through loguru.
logger.disable("PyRuSH")

SYMPTOM_LABEL = "SYMPTOM"

# Clinical triggers that introduce a list of negated findings ("denies fever, chills and
# cough"). These keep negating across commas; every other forward negation stops at one.
LIST_NEGATION_TRIGGERS = {"deny", "negative for", "free of", "absence of", "clear of",
                          "lack of", "unremarkable for"}


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
    # The env var lets Docker point at a mounted index without editing the YAML.
    quickumls_path = resolve(os.environ.get("MEDINFER_QUICKUMLS_PATH") or config.get("quickumls_path"))

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

    for rule in nlp.get_pipe("medspacy_context").rules:
        if (rule.category == "NEGATED_EXISTENCE" and rule.direction == "FORWARD"
                and rule.literal.lower() not in LIST_NEGATION_TRIGGERS):
            rule.on_modifies = negation_stops_at_comma

    return nlp


def negation_stops_at_comma(target: Span, modifier: Span, span_between: Span) -> bool:
    """Patient text: "no fever, runny nose and sneezing" negates only the fever.

    ConText would otherwise negate everything up to the end of the sentence. An
    "or"/"nor" list ("no fever, cough, or chills") is still negated as a whole.
    """
    if not any(t.text == "," for t in span_between):
        return True
    rest_of_sentence = target.doc[modifier.end:target.sent.end]
    return any(t.lower_ in ("or", "nor") for t in rest_of_sentence)
