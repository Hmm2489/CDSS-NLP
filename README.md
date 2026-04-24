# medinfer

Extracts symptoms from free-text patient descriptions and ranks possible
diagnoses with a weighted forward-chaining rule engine.

```
text ──► medSpaCy ──────────────────────────────► list[Symptom] ──► evidence {CUI: CF} ──► forward chaining ──► ranked diagnoses
         ├ PyRuSH        sentence splitting                                                 symptoms → syndromes → diseases
         ├ TargetMatcher lay-language lexicon → CUI                                         MYCIN certainty factors
         ├ QuickUMLS     fuzzy UMLS match → CUI
         └ ConText       negated / uncertain / historical / family / hypothetical
```

**Research/learning project. Not medical advice.**

## Setup

```bash
mise install                      # Python 3.11 (pinned in mise.toml; QuickUMLS won't build on 3.13+)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Usage

```bash
.venv/bin/medinfer "I've been burning up for 3 days with body aches and a dry cough. No sore throat."
.venv/bin/medinfer --json "..."

.venv/bin/python -m evaluation.eval_extraction data/processed/sample.jsonl
.venv/bin/python -m evaluation.eval_diagnosis  data/processed/sample.jsonl
```

## Layout

| Path | What |
|---|---|
| `configs/` | Pipeline and inference settings |
| `src/medinfer/schema.py` | `Symptom` / `Diagnosis`: the contract between NLP and inference |
| `src/medinfer/nlp/` | medSpaCy pipeline, lexicon, custom ConText rules |
| `src/medinfer/extract.py` | text → `list[Symptom]` |
| `src/medinfer/inference/` | CF arithmetic, rules/KB, forward-chaining engine |
| `src/medinfer/kb/build_umls_kb.py` | UMLS `MRREL` → symptom–disease edge CSV |
| `src/medinfer/diagnoser.py` | end-to-end wrapper |
| `data/kb/rules.yaml` | starter rules (illustrative weights) |
| `data/processed/` | benchmarks converted to the common JSONL format |
| `evaluation/` | metrics and eval scripts; results land in `evaluation/results/` |

## Roadmap

1. **UMLS license**: request at <https://uts.nlm.nih.gov/uts/signup-login>. Install with MetamorphoSys
   into `data/umls/` (include SNOMEDCT_US, MSH, MDR, CHV).
2. **QuickUMLS index**: `python -m quickumls.install data/umls/<release>/META data/quickumls_index`,
   then set `quickumls_path` in `configs/pipeline.yaml`.
3. **UMLS edges**: `python -m medinfer.kb.build_umls_kb data/umls/<release>/META data/kb/symptom_disease_edges.csv`,
   then set `edges_path` in `configs/inference.yaml`.
4. **Real weights**: replace the default edge weight with frequencies from HPO annotations
   (`phenotype.hpoa`), the Zhou et al. 2014 symptom–disease network, or DDXPlus.
5. **Benchmarks**: write converters into `evaluation/datasets.py` format for Symptom2Disease and
   DDXPlus first (public), then ShARe/CLEF 2013 and i2b2 2010 assertions (credentialed).
6. **Baselines** in `evaluation/baselines/`: scispaCy linker vs QuickUMLS; TF-IDF + logistic
   regression text → disease vs the rule engine.
