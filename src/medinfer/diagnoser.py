"""End-to-end: text -> symptoms -> evidence -> ranked diagnoses."""

from __future__ import annotations

from dataclasses import dataclass

from medinfer.config import load_config, resolve
from medinfer.extract import SymptomExtractor
from medinfer.inference.engine import ForwardChainingEngine
from medinfer.inference.facts import symptoms_to_evidence
from medinfer.inference.rules import KnowledgeBase
from medinfer.schema import Diagnosis, Symptom


@dataclass
class DiagnoserOutput:
    symptoms: list[Symptom]
    evidence: dict[str, float]
    diagnoses: list[Diagnosis]


class Diagnoser:
    def __init__(self, extractor: SymptomExtractor | None = None, inference_config: dict | None = None):
        self.config = inference_config or load_config("inference")
        self.extractor = extractor or SymptomExtractor()
        self.kb = KnowledgeBase.load(resolve(self.config["rules_path"]), resolve(self.config.get("edges_path")))
        self.engine = ForwardChainingEngine(
            self.kb,
            premise_threshold=self.config["premise_threshold"],
            max_iterations=self.config["max_iterations"],
        )

    def __call__(self, text: str, top_k: int | None = None) -> DiagnoserOutput:
        symptoms = self.extractor(text)
        evidence = symptoms_to_evidence(symptoms, self.config)
        diagnoses = self.engine.diagnose(evidence, top_k or self.config["top_k"])
        return DiagnoserOutput(symptoms, evidence, diagnoses)
