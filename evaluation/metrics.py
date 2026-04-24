"""Metric helpers shared by the evaluation scripts."""

from __future__ import annotations


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def rank_of(gold: str, ranked: list[str]) -> int | None:
    """1-based rank of the gold label, or None if absent."""
    return ranked.index(gold) + 1 if gold in ranked else None


def ranking_metrics(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict[str, float]:
    n = len(ranks)
    out = {f"top{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out
