"""Evaluate symptom extraction at the CUI level.

    python -m evaluation.eval_extraction data/processed/sample.jsonl

Detection: micro P/R/F1 over (document, CUI) pairs, so the same symptom twice in
one document counts once. Assertion: of the correctly detected CUIs, how often
the negated/affirmed status is right. Errors are written for error analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.datasets import load_jsonl, write_jsonl
from evaluation.metrics import prf
from medinfer.config import PROJECT_ROOT, load_config
from medinfer.extract import SymptomExtractor
from medinfer.inference.facts import symptoms_to_evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation/results/extraction_errors.jsonl")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.data) if "symptoms" in r]
    extract = SymptomExtractor()
    config = load_config("inference")

    tp = fp = fn = assert_correct = 0
    errors = []
    for row in rows:
        gold = {s["cui"]: s["negated"] for s in row["symptoms"]}
        # Same family/history filtering and mention merging the engine sees.
        pred = {cui: cf < 0 for cui, cf in symptoms_to_evidence(extract(row["text"]), config).items()}

        hits = gold.keys() & pred.keys()
        tp += len(hits)
        fp += len(pred.keys() - gold.keys())
        fn += len(gold.keys() - pred.keys())
        wrong_assertion = [c for c in hits if gold[c] != pred[c]]
        assert_correct += len(hits) - len(wrong_assertion)

        if pred != gold:
            errors.append({
                "id": row["id"], "text": row["text"],
                "missed": sorted(gold.keys() - pred.keys()),
                "spurious": sorted(pred.keys() - gold.keys()),
                "wrong_negation": wrong_assertion,
            })

    results = {"documents": len(rows), "detection": prf(tp, fp, fn),
               "assertion_accuracy": assert_correct / tp if tp else 0.0}
    print(json.dumps(results, indent=2))
    write_jsonl(errors, args.out)
    print(f"{len(errors)} documents with errors -> {args.out}")


if __name__ == "__main__":
    main()
