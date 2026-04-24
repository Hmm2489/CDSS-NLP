"""Rules and the knowledge base they live in.

rules.yaml format:

    concepts:
      C0021400: {name: Influenza, type: disease}
      RESP_INF: {name: Respiratory infection, type: intermediate}
    rules:
      - id: flu_core
        if: [C0015967, C0231528]     # fever AND myalgia
        then: C0021400
        cf: 0.6
      - id: flu_no_fever
        if: ["!C0015967"]            # "!" = premise holds when the finding is *absent*
        then: C0021400
        cf: -0.4                     # negative CF = evidence against

Conclusions can be intermediate concepts that appear in other rules' `if`
lists. That's what makes it forward *chaining*.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Premise:
    cui: str
    absent: bool = False

    @classmethod
    def parse(cls, s: str) -> Premise:
        return cls(s[1:], absent=True) if s.startswith("!") else cls(s)

    def strength(self, facts: dict[str, float], threshold: float) -> float | None:
        """How strongly this premise holds, or None if it doesn't hold."""
        cf = facts.get(self.cui)
        if cf is None:
            return None
        value = -cf if self.absent else cf
        return value if value >= threshold else None


@dataclass(frozen=True)
class Rule:
    id: str
    premises: tuple[Premise, ...]
    conclusion: str
    cf: float

    def fire(self, facts: dict[str, float], threshold: float) -> float | None:
        """CF this rule contributes to its conclusion, or None if it doesn't fire.

        Conjunctions take the weakest premise (MYCIN's AND = min).
        """
        strengths = [p.strength(facts, threshold) for p in self.premises]
        if any(s is None for s in strengths):
            return None
        return self.cf * min(strengths)


@dataclass
class KnowledgeBase:
    concepts: dict[str, dict]
    rules: list[Rule]

    def name(self, cui: str) -> str:
        return self.concepts.get(cui, {}).get("name", cui)

    def is_disease(self, cui: str) -> bool:
        return self.concepts.get(cui, {}).get("type") == "disease"

    @classmethod
    def load(cls, rules_path: Path, edges_path: Path | None = None) -> KnowledgeBase:
        with open(rules_path) as f:
            data = yaml.safe_load(f)
        concepts = dict(data.get("concepts", {}))
        rules = [
            Rule(
                id=r["id"],
                premises=tuple(Premise.parse(p) for p in r["if"]),
                conclusion=r["then"],
                cf=float(r["cf"]),
            )
            for r in data.get("rules", [])
        ]
        _check_cf_range(rules)
        if edges_path is not None:
            rules += _rules_from_edges(edges_path, concepts)
        return cls(concepts, rules)


def _rules_from_edges(path: Path, concepts: dict[str, dict]) -> list[Rule]:
    """One-premise rules from the CSV that kb/build_umls_kb.py writes."""
    rules = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concepts.setdefault(row["disease_cui"], {"name": row["disease_name"], "type": "disease"})
            rules.append(
                Rule(
                    id=f"umls:{row['symptom_cui']}->{row['disease_cui']}",
                    premises=(Premise(row["symptom_cui"]),),
                    conclusion=row["disease_cui"],
                    cf=float(row["weight"]),
                )
            )
    return rules


def _check_cf_range(rules: list[Rule]) -> None:
    bad = [r.id for r in rules if not -1.0 <= r.cf <= 1.0]
    if bad:
        raise ValueError(f"Rule CFs must be in [-1, 1]: {bad}")
