"""Runs the UMLS edge builder on a tiny fake META directory."""

from medinfer.kb.build_umls_kb import build_edges, load_names


def write_rrf(path, rows):
    path.write_text("".join("|".join(r) + "|\n" for r in rows))


def mrrel(cui1, rela, cui2, sab="SNOMEDCT_US", suppress="N"):
    return [cui1, "A1", "SCUI", "RO", cui2, "A2", "SCUI", rela, "R1", "", sab, sab, "", "Y", suppress, ""]


def test_build_edges(tmp_path):
    write_rrf(tmp_path / "MRSTY.RRF", [
        ["C_FEVER", "T184", "", "Sign or Symptom", "", ""],
        ["C_FLU", "T047", "", "Disease or Syndrome", "", ""],
        ["C_DRUG", "T121", "", "Pharmacologic Substance", "", ""],
    ])
    write_rrf(tmp_path / "MRREL.RRF", [
        mrrel("C_FLU", "manifestation_of", "C_FEVER"),          # symptom in CUI2
        mrrel("C_FEVER", "has_manifestation", "C_FLU", "MEDCIN"),  # reversed: same edge, second source
        mrrel("C_FEVER", "may_treat", "C_DRUG"),                 # irrelevant RELA
        mrrel("C_FEVER", "manifestation_of", "C_FLU", suppress="O"),  # obsolete row
    ])
    write_rrf(tmp_path / "MRCONSO.RRF", [
        ["C_FLU", "ENG", "P", "L1", "PF", "S1", "Y", "A1", "", "", "", "MSH", "PT", "D1", "Influenza", "0", "N", ""],
        ["C_FLU", "ENG", "S", "L2", "VO", "S2", "N", "A2", "", "", "", "MSH", "ET", "D1", "Flu", "0", "N", ""],
    ])

    edges = build_edges(tmp_path)
    assert edges == {("C_FEVER", "C_FLU"): {"SNOMEDCT_US", "MEDCIN"}}
    assert load_names(tmp_path, {"C_FLU"}) == {"C_FLU": "Influenza"}
