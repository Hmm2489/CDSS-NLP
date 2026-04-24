"""Weighted forward-chaining inference.

Each round fires every rule whose premises hold given current facts, then
recomputes every derived concept's CF from scratch by combining the
contributions of the rules that concluded it. This repeats until nothing
changes. Recomputing from scratch (rather than "fire each rule once") means a
rule sees its premises' final CFs even when they were derived late in the chain.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from medinfer.inference.rules import KnowledgeBase
from medinfer.inference.scoring import combine_all
from medinfer.schema import Diagnosis

_EPS = 1e-9


@dataclass
class InferenceResult:
    facts: dict[str, float]               # final working memory (evidence + derived)
    contributions: dict[str, float]       # rule id -> CF it contributed
    iterations: int
    converged: bool


class ForwardChainingEngine:
    def __init__(self, kb: KnowledgeBase, premise_threshold: float = 0.2, max_iterations: int = 50):
        self.kb = kb
        self.threshold = premise_threshold
        self.max_iterations = max_iterations
        self._rules_by_id = {r.id: r for r in kb.rules}

    def run(self, evidence: dict[str, float]) -> InferenceResult:
        contributions: dict[str, float] = {}
        facts = dict(evidence)

        for iteration in range(1, self.max_iterations + 1):
            fired = {}
            for rule in self.kb.rules:
                cf = rule.fire(facts, self.threshold)
                if cf is not None:
                    fired[rule.id] = cf

            if _same(fired, contributions):
                return InferenceResult(facts, contributions, iteration, converged=True)

            contributions = fired
            facts = self._derive(evidence, contributions)

        return InferenceResult(facts, contributions, self.max_iterations, converged=False)

    def diagnose(self, evidence: dict[str, float], top_k: int = 5) -> list[Diagnosis]:
        result = self.run(evidence)
        rules_for: dict[str, list[str]] = defaultdict(list)
        for rule_id in result.contributions:
            rules_for[self._rules_by_id[rule_id].conclusion].append(rule_id)

        ranked = sorted(
            (
                Diagnosis(cui, self.kb.name(cui), cf, rules_for[cui])
                for cui, cf in result.facts.items()
                if self.kb.is_disease(cui) and cf > 0
            ),
            key=lambda d: d.cf,
            reverse=True,
        )
        return ranked[:top_k]

    def _derive(self, evidence: dict[str, float], contributions: dict[str, float]) -> dict[str, float]:
        by_conclusion: dict[str, list[float]] = defaultdict(list)
        for rule_id, cf in contributions.items():
            by_conclusion[self._rules_by_id[rule_id].conclusion].append(cf)

        facts = dict(evidence)
        for cui, cfs in by_conclusion.items():
            # Directly observed evidence for a concept combines with derived support.
            facts[cui] = combine_all([evidence.get(cui, 0.0), *cfs])
        return facts


def _same(a: dict[str, float], b: dict[str, float]) -> bool:
    return a.keys() == b.keys() and all(abs(a[k] - b[k]) < _EPS for k in a)
