import json
import sys

import pytest

from medinfer import cli

FEVER = "C0015967"


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["medinfer", *args])
    cli.main()
    return capsys.readouterr().out


def test_json_output_is_valid(monkeypatch, capsys):
    out = json.loads(run(monkeypatch, capsys, "--json", "I've been burning up with body aches"))
    assert set(out) == {"symptoms", "evidence", "diagnoses"}
    assert FEVER in {s["cui"] for s in out["symptoms"]}
    assert all(-1.0 <= d["cf"] <= 1.0 for d in out["diagnoses"])


def test_top_k_limits_diagnoses(monkeypatch, capsys):
    out = json.loads(run(monkeypatch, capsys, "--json", "-k", "1", "fever, body aches and a dry cough"))
    assert len(out["diagnoses"]) <= 1


def test_text_output_has_disclaimer(monkeypatch, capsys):
    out = run(monkeypatch, capsys, "I have a fever")
    assert "Symptoms:" in out
    assert "not medical advice" in out


def test_missing_text_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["medinfer"])
    with pytest.raises(SystemExit):
        cli.main()
