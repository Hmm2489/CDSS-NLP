"""Evaluate end-to-end diagnosis ranking.

    python -m evaluation.eval_diagnosis data/processed/sample.jsonl

Reports top-k accuracy and MRR. Also reports how many gold diagnoses the
knowledge base doesn't contain at all: those cases can never be right, so a
low score with low coverage is a KB problem, not an engine problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.datasets import load_jsonl, write_jsonl
from evaluation.metrics import rank_of, ranking_metrics
from medinfer.config import PROJECT_ROOT
from medinfer.diagnoser import Diagnoser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation/results/diagnosis_cases.jsonl")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.data) if "diagnosis" in r]
    diagnoser = Diagnoser()
    known = {c for c in diagnoser.kb.concepts if diagnoser.kb.is_disease(c)}

    ranks, cases = [], []
    for row in rows:
        out = diagnoser(row["text"], top_k=50)
        ranked = [d.cui for d in out.diagnoses]
        rank = rank_of(row["diagnosis"], ranked)
        ranks.append(rank)
        cases.append({
            "id": row["id"], "gold": row["diagnosis"], "rank": rank,
            "in_kb": row["diagnosis"] in known,
            "predicted": [(d.name, round(d.cf, 3)) for d in out.diagnoses[:5]],
            "evidence": out.evidence,
        })

    results = {
        "cases": len(rows),
        "kb_coverage": sum(c["in_kb"] for c in cases) / len(rows),
        **ranking_metrics(ranks),
    }
    print(json.dumps(results, indent=2))
    write_jsonl(cases, args.out)
    print(f"Per-case results -> {args.out}")


if __name__ == "__main__":
    main()
