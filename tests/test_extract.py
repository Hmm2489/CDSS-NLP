import pytest

from medinfer.extract import SymptomExtractor

FEVER, COUGH, CHEST_PAIN, SORE_THROAT, HEADACHE = "C0015967", "C0010200", "C0008031", "C0242429", "C0018681"


@pytest.fixture(scope="module")
def extract():
    return SymptomExtractor()


def by_cui(symptoms):
    return {s.cui: s for s in symptoms}


def test_lay_terms_map_to_cuis(extract):
    found = by_cui(extract("I've been burning up and coughing all night"))
    assert {FEVER, COUGH} <= found.keys()


def test_negation(extract):
    found = by_cui(extract("No chest pain, but I have a cough."))
    assert found[CHEST_PAIN].is_negated
    assert not found[COUGH].is_negated  # "but" ends the negation scope


def test_patient_phrased_negation(extract):
    assert by_cui(extract("I don't have a fever."))[FEVER].is_negated


def test_family_history(extract):
    assert by_cui(extract("My mom had a sore throat."))[SORE_THROAT].is_family


def test_uncertainty(extract):
    assert by_cui(extract("Maybe a headache."))[HEADACHE].is_uncertain


def test_curly_apostrophe(extract):
    assert by_cui(extract("I don’t have a fever."))[FEVER].is_negated


def test_offsets_point_into_original_text(extract):
    text = "Woke up with a headache."
    [s] = extract(text)
    assert text[s.start:s.end] == s.text
