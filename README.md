# Clinical Decision Support System

This project is a clinical decision support system that processes
unstructured patient symptom descriptions and generates a ranked list of potential
diagnoses. By leveraging natural language processing and rule-based reasoning, the
system helps transform free-text symptom descriptions into clinically meaningful
information and provide decision support to healthcare professionals.

Given a description such as *"I've been burning up for three days, my whole body aches and I
have a dry cough. No sore throat."*, the system:

1. **Extracts symptoms** from the text and maps each one to a standard UMLS concept identifier
   (CUI), so that "burning up", "feverish" and "pyrexia" all become the same concept: *Fever*.
2. **Determines each symptom's status**: present, negated ("no sore throat"), uncertain,
   historical, or experienced by a family member rather than the patient.
3. **Reasons over the findings** with a weighted forward-chaining rule engine that moves from
   symptoms to intermediate syndromes to diseases, and outputs a ranked differential with
   the rules that support each candidate.

```
$ medinfer "I've been burning up for 3 days with body aches and a dry cough. No sore throat."
Symptoms:
  C0015967  'burning up'     present
  C0231528  'body aches'     present
  C0010200  'dry cough'      present
  C0242429  'sore throat'    negated

Possible diagnoses (not medical advice):
  1. Influenza   CF=0.51  rules: flu_febrile_aches, flu_febrile_cough
  2. COVID-19    CF=0.18  rules: covid_febrile_cough
```

## Status

**In progress — core system complete, final refinements underway.**

The full pipeline is implemented end to end: symptom extraction with UMLS normalization,
contextual status detection, and forward-chaining diagnosis over a knowledge base built
from the UMLS Metathesaurus. Remaining work focuses on refinements:

- tighter negation scope for symptom lists (see Limitations)
- capturing symptom severity, duration, and onset as evidence

The system is intended for research and education.
**Its output is not medical advice and must not be used for clinical decisions.**

## Motivation

There are many challenges associated with the diagnostic process. Diagnostic error is
common: the US National Academies concluded that most people will experience at least one
diagnostic error in their lifetime [1], and roughly 1 in 20 US adults is affected by one in
outpatient care each year [2].

One of these challenges is that patients' descriptions of their symptoms are very
unstructured and subjective. Patients describe the same complaint in many different ways
("burning up", "running a temp", "feverish"), mix symptoms they have with ones they don't
("no fever, but…"), and mention things that belong to someone else ("my mom had a sore
throat"). Before any reasoning can happen, that free text has to be turned into structured,
standardized findings. This project explores how far established clinical NLP tools,
combined with an explainable rule-based reasoner, can go in doing that.

## Architecture

The system has two halves joined by a single data contract, `Symptom` (`schema.py`). The
NLP half only produces `Symptom` records; the inference half only consumes them. Each half
can therefore be tested, benchmarked, and replaced independently.

```
free text
   │
   ▼
┌─────────────────────────── NLP (medSpaCy pipeline) ───────────────────────────┐
│  PyRuSH          sentence splitting                                           │
│  TargetMatcher   lay-language lexicon  →  CUI     ("burning up" → Fever)      │
│  QuickUMLS       approximate string match against UMLS  →  CUI                │
│  ConText         negated / uncertain / historical / hypothetical / family     │
└───────────────────────────────────────────────────────────────────────────────┘
   │  list[Symptom]
   ▼
evidence  {CUI: certainty factor}      present = +1.0, uncertain = +0.4, negated = −1.0
   │
   ▼
┌────────────────────── Inference (forward chaining) ───────────────────────────┐
│  rules:   IF fever AND myalgia THEN influenza  (CF 0.6)                       │
│  chains:  symptoms → intermediate syndromes → diseases                        │
│  combines evidence with MYCIN certainty factors                               │
│  negative evidence: "no fever" lowers diseases that expect fever              │
└───────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
ranked diagnoses, each with the rules that fired for it
```

The NLP half is built on medSpaCy, QuickUMLS and the ConText algorithm, and normalizes
symptoms to concepts in the UMLS Metathesaurus. The knowledge base is generated from
symptom–disease relations in the same UMLS release by `build_umls_kb.py`.

Project layout:

```
.
├── configs/
│   ├── pipeline.yaml            NLP settings: QuickUMLS index path, threshold, semantic types
│   └── inference.yaml           evidence weights, premise threshold, top-k
├── src/medinfer/
│   ├── schema.py                Symptom / Diagnosis — the contract between the two halves
│   ├── extract.py               text → list[Symptom]
│   ├── diagnoser.py             end-to-end wrapper: text → symptoms → ranked diagnoses
│   ├── cli.py                   `medinfer` command-line entry point
│   ├── nlp/
│   │   ├── pipeline.py          builds the medSpaCy pipeline
│   │   ├── preprocess.py        offset-preserving text normalization
│   │   └── resources/           lay-language symptom lexicon, patient-phrasing ConText rules
│   ├── inference/
│   │   ├── scoring.py           certainty-factor arithmetic
│   │   ├── rules.py             rule and knowledge-base loading
│   │   ├── facts.py             symptoms → initial working memory
│   │   └── engine.py            weighted forward-chaining engine
│   └── kb/
│       └── build_umls_kb.py     extracts symptom–disease relations from the UMLS Metathesaurus
├── data/
│   ├── umls/                    UMLS Metathesaurus release (RRF files)
│   ├── quickumls_index/         QuickUMLS index built from the UMLS release
│   ├── kb/rules.yaml            knowledge base generated from UMLS relations
│   └── processed/               evaluation data in a common JSONL format
├── evaluation/                  extraction and diagnosis metrics (P/R/F1, top-k, MRR)
├── tests/
├── Dockerfile
└── compose.yaml
```

## Setup & Docker

### Local

Requires Python 3.11 (pinned in `mise.toml`; QuickUMLS dependencies do not build on 3.13+).

```bash
mise install                         # or install Python 3.11 another way
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                     # run the test suite
```

Usage:

```bash
.venv/bin/medinfer "I've been burning up for 3 days with body aches and a dry cough."
.venv/bin/medinfer --json "..."      # machine-readable output

.venv/bin/python -m evaluation.eval_extraction data/processed/sample.jsonl
.venv/bin/python -m evaluation.eval_diagnosis  data/processed/sample.jsonl
```

### Docker

```bash
docker compose build
docker compose run --rm medinfer "I've had a fever and body aches since Monday"
docker compose run --rm --entrypoint python medinfer -m evaluation.eval_diagnosis data/processed/sample.jsonl
docker build --target test .         # run the test suite inside the container
```

The UMLS data and the prebuilt QuickUMLS index ship in `data/`. Compose mounts
`data/quickumls_index/` into the container read-only, and the pipeline picks it up through
`MEDINFER_QUICKUMLS_PATH`. To rebuild the index from the UMLS release:

```bash
python -m quickumls.install data/umls/<release>/META data/quickumls_index
```

## Limitations and constraints

- **Not clinically validated.** The system is a research prototype and has not been
  evaluated in a clinical setting.
- **Paraphrase coverage.** QuickUMLS matches strings approximately, so it can miss
  descriptive paraphrases ("feel like I can't get enough air").
- **Negation scope errors.** ConText can over-extend negation across a list: in "no fever,
  runny nose and sneezing", all three symptoms are marked negated, although the patient
  most likely has the runny nose and sneezing.
- **No severity, duration, or onset.** "Mild cough for two days" and "severe cough for three
  weeks" currently produce the same evidence.
- **Certainty factors are not probabilities.** MYCIN-style certainty factors assume pieces of
  evidence are independent and do not have a rigorous probabilistic interpretation;
  scores rank candidates but are not calibrated likelihoods.
- **Relations without strengths.** UMLS records *which* symptoms relate to which diseases,
  but not *how strongly*, so rule weights are an approximation of real symptom–disease
  association strength.
- **English only**, and tuned for consumer phrasing rather than clinical notes.

## Future considerations

- **Transformer-based concept extraction.** Symptom extraction currently relies on lexicon
  and approximate string matching. A transformer model fine-tuned for clinical concept
  recognition could understand context and catch descriptive paraphrases ("feel like I
  can't get enough air") that string matching misses, while still feeding the same
  `Symptom` records into the rule-based reasoner.
- **Clinical notes and other languages.** Adapting the pipeline to clinician-written notes
  and to non-English patient descriptions, both of which use different vocabulary and
  phrasing.
- **Integration with health records.** Reading structured data such as age, sex, and medical
  history from an electronic health record (e.g. through the FHIR standard) to make the
  differential more specific to the patient.

## Citations

1. National Academies of Sciences, Engineering, and Medicine. *Improving Diagnosis in Health
   Care.* Washington, DC: The National Academies Press; 2015.
2. Singh H, Meyer AND, Thomas EJ. The frequency of diagnostic errors in outpatient care:
   estimations from three large observational studies involving US adult populations.
   *BMJ Quality & Safety.* 2014;23(9):727–731.
