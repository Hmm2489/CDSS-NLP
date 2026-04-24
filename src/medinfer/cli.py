"""Command line: medinfer "I've had a fever and body aches since Monday"."""

from __future__ import annotations

import argparse
import json

from medinfer.diagnoser import Diagnoser


def _flags(s) -> str:
    names = ["negated", "uncertain", "historical", "hypothetical", "family"]
    return ", ".join(n for n in names if getattr(s, f"is_{n}")) or "present"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract symptoms and rank possible diagnoses.")
    ap.add_argument("text", help="Free-text patient description")
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = ap.parse_args()

    out = Diagnoser()(args.text, top_k=args.top_k)

    if args.json:
        print(json.dumps({
            "symptoms": [s.to_dict() for s in out.symptoms],
            "evidence": out.evidence,
            "diagnoses": [d.to_dict() for d in out.diagnoses],
        }, indent=2))
        return

    print("Symptoms:")
    for s in out.symptoms:
        print(f"  {s.cui}  {s.text!r:28} {_flags(s):12} via {s.source}")
    print("\nPossible diagnoses (not medical advice):")
    if not out.diagnoses:
        print("  none above zero certainty")
    for i, d in enumerate(out.diagnoses, 1):
        print(f"  {i}. {d.name:24} CF={d.cf:.2f}  rules: {', '.join(d.fired_rules)}")


if __name__ == "__main__":
    main()
