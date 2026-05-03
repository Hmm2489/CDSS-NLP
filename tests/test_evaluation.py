import pytest

from evaluation.datasets import load_jsonl, write_jsonl
from evaluation.metrics import prf, rank_of, ranking_metrics


def test_prf_by_hand():
    m = prf(tp=6, fp=2, fn=4)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_prf_no_predictions_is_zero_not_error():
    assert prf(tp=0, fp=0, fn=3) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 3}


def test_rank_of():
    ranked = ["flu", "covid", "cold"]
    assert rank_of("flu", ranked) == 1
    assert rank_of("cold", ranked) == 3
    assert rank_of("migraine", ranked) is None


def test_ranking_metrics_by_hand():
    # Gold ranked 1st, 2nd, 4th and missing.
    m = ranking_metrics([1, 2, 4, None], ks=(1, 3, 5))
    assert m["top1"] == pytest.approx(0.25)
    assert m["top3"] == pytest.approx(0.5)
    assert m["top5"] == pytest.approx(0.75)
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 4)


def test_jsonl_round_trip(tmp_path):
    rows = [{"id": "a", "text": "fever", "diagnosis": "C0021400"}, {"id": "b", "text": "cough"}]
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(rows, path)
    assert load_jsonl(path) == rows
