"""Extract symptom<->disease edges from a UMLS Metathesaurus release.

    python -m medinfer.kb.build_umls_kb data/umls/2026AA/META data/kb/symptom_disease_edges.csv

Reads three RRF files (pipe-delimited, no header):
  MRCONSO.RRF  concept names        CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|...
  MRSTY.RRF    semantic types       CUI|TUI|STN|STY|ATUI|CVF
  MRREL.RRF    relations            CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|...

Important: UMLS gives you *which* edges exist, not how strong they are. Every
edge gets `--default-weight`; the real work is replacing that with weights
from a frequency source (HPO annotations, Zhou et al. 2014, DDXPlus).
Edges are oriented by semantic type rather than trusting RELA direction,
since direction conventions differ between source vocabularies.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SYMPTOM_TUIS = {"T184", "T033"}           # Sign or Symptom, Finding
DISEASE_TUIS = {"T047", "T191", "T046"}   # Disease or Syndrome, Neoplastic Process, Pathologic Function

# RELA values that link findings to diseases. Inspect your release with:
#   cut -d'|' -f8,11 MRREL.RRF | sort | uniq -c | sort -rn | less
RELAS = {
    "has_manifestation", "manifestation_of",
    "disease_has_finding", "is_finding_of_disease",
    "disease_may_have_finding", "may_be_finding_of_disease",
    "has_associated_finding", "associated_finding_of",
    "has_definitional_manifestation", "definitional_manifestation_of",
}


def read_rrf(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("|")


def load_semtypes(meta: Path) -> dict[str, set[str]]:
    tuis = defaultdict(set)
    for row in read_rrf(meta / "MRSTY.RRF"):
        tuis[row[0]].add(row[1])
    return tuis


def load_names(meta: Path, cuis: set[str]) -> dict[str, str]:
    """English preferred name per CUI (TS=P, STT=PF, ISPREF=Y)."""
    names = {}
    for row in read_rrf(meta / "MRCONSO.RRF"):
        cui, lat, ts, stt, ispref, name = row[0], row[1], row[2], row[4], row[6], row[14]
        if cui in cuis and lat == "ENG" and ts == "P" and stt == "PF" and ispref == "Y":
            names.setdefault(cui, name)
    return names


def build_edges(meta: Path) -> dict[tuple[str, str], set[str]]:
    tuis = load_semtypes(meta)

    def has_type(cui: str, wanted: set[str]) -> bool:
        return bool(tuis.get(cui, set()) & wanted)

    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in read_rrf(meta / "MRREL.RRF"):
        cui1, cui2, rela, sab, suppress = row[0], row[4], row[7], row[10], row[14]
        if rela not in RELAS or suppress in {"O", "E", "Y"}:
            continue
        if has_type(cui1, SYMPTOM_TUIS) and has_type(cui2, DISEASE_TUIS):
            edges[(cui1, cui2)].add(sab)
        elif has_type(cui2, SYMPTOM_TUIS) and has_type(cui1, DISEASE_TUIS):
            edges[(cui2, cui1)].add(sab)
    return edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_dir", type=Path, help="UMLS META directory containing the .RRF files")
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--default-weight", type=float, default=0.3)
    args = ap.parse_args()

    edges = build_edges(args.meta_dir)
    names = load_names(args.meta_dir, {c for pair in edges for c in pair})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symptom_cui", "symptom_name", "disease_cui", "disease_name", "weight", "sources"])
        for (sym, dis), sabs in sorted(edges.items()):
            w.writerow([sym, names.get(sym, ""), dis, names.get(dis, ""), args.default_weight, ";".join(sorted(sabs))])

    print(f"Wrote {len(edges)} edges ({len({d for _, d in edges})} diseases) to {args.out_csv}")


if __name__ == "__main__":
    main()
