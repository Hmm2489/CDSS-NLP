import pytest

from medinfer.inference.engine import ForwardChainingEngine
from medinfer.inference.facts import symptoms_to_evidence
from medinfer.inference.rules import KnowledgeBase, Premise, Rule
from medinfer.inference.scoring import combine, combine_all
from medinfer.schema import Symptom

FEVER, COUGH, SOB = "C0015967", "C0010200", "C0013404"


def kb(*rules, diseases=("FLU", "PNA")):
    return KnowledgeBase({d: {"name": d, "type": "disease"} for d in diseases}, list(rules))


def rule(id, premises, then, cf):
    return Rule(id, tuple(Premise.parse(p) for p in premises), then, cf)


class TestCombine:
    def test_positive(self):
        assert combine(0.6, 0.5) == pytest.approx(0.8)

    def test_negative(self):
        assert combine(-0.6, -0.5) == pytest.approx(-0.8)

    def test_mixed(self):
        assert combine(0.6, -0.4) == pytest.approx(0.2 / 0.6)

    def test_full_contradiction_is_unknown(self):
        assert combine(1.0, -1.0) == 0.0

    def test_order_independent(self):
        assert combine_all([0.3, -0.2, 0.7]) == pytest.approx(combine_all([0.7, 0.3, -0.2]))


class TestEngine:
    def test_conjunction_uses_weakest_premise(self):
        engine = ForwardChainingEngine(kb(rule("r", [FEVER, COUGH], "FLU", 0.8)))
        [d] = engine.diagnose({FEVER: 1.0, COUGH: 0.5})
        assert d.cf == pytest.approx(0.4)

    def test_rule_needs_all_premises(self):
        engine = ForwardChainingEngine(kb(rule("r", [FEVER, COUGH], "FLU", 0.8)))
        assert engine.diagnose({FEVER: 1.0}) == []

    def test_chains_through_intermediate(self):
        engine = ForwardChainingEngine(kb(
            rule("to_lri", [COUGH, SOB], "LRI", 0.5),
            rule("to_pna", ["LRI"], "PNA", 0.8),
        ))
        [d] = engine.diagnose({COUGH: 1.0, SOB: 1.0})
        assert d.cui == "PNA"
        assert d.cf == pytest.approx(0.4)

    def test_chain_order_in_rule_list_does_not_matter(self):
        engine = ForwardChainingEngine(kb(
            rule("to_pna", ["LRI"], "PNA", 0.8),   # listed before the rule that derives LRI
            rule("to_lri", [COUGH, SOB], "LRI", 0.5),
        ))
        [d] = engine.diagnose({COUGH: 1.0, SOB: 1.0})
        assert d.cf == pytest.approx(0.4)

    def test_absent_premise_and_negative_evidence(self):
        engine = ForwardChainingEngine(kb(
            rule("flu_cough", [COUGH], "FLU", 0.6),
            rule("flu_no_fever", ["!" + FEVER], "FLU", -0.5),
        ))
        [d] = engine.diagnose({COUGH: 1.0, FEVER: -1.0})
        assert d.cf == pytest.approx(combine(0.6, -0.5))

    def test_absent_premise_needs_explicit_negation(self):
        # Not mentioning fever is not the same as denying it.
        engine = ForwardChainingEngine(kb(rule("r", ["!" + FEVER], "FLU", 0.5)))
        assert engine.diagnose({}) == []

    def test_reports_fired_rules(self):
        engine = ForwardChainingEngine(kb(rule("a", [COUGH], "FLU", 0.3), rule("b", [FEVER], "FLU", 0.3)))
        [d] = engine.diagnose({COUGH: 1.0, FEVER: 1.0})
        assert sorted(d.fired_rules) == ["a", "b"]


class TestEvidence:
    CONFIG = {
        "evidence_cf": {"present": 1.0, "uncertain": 0.4, "negated": -1.0},
        "drop_if": ["is_family", "is_hypothetical", "is_historical"],
    }

    def sym(self, cui, **flags):
        return Symptom(cui=cui, text="x", start=0, end=1, source="lexicon", **flags)

    def test_assertion_mapping(self):
        ev = symptoms_to_evidence(
            [self.sym(FEVER, is_negated=True), self.sym(COUGH, is_uncertain=True), self.sym(SOB, is_family=True)],
            self.CONFIG,
        )
        assert ev == {FEVER: -1.0, COUGH: 0.4}

    def test_repeat_mentions_do_not_stack(self):
        ev = symptoms_to_evidence([self.sym(COUGH, is_uncertain=True), self.sym(COUGH, is_uncertain=True)], self.CONFIG)
        assert ev == {COUGH: 0.4}
