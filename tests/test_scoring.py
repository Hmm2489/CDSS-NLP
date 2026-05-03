import itertools

import pytest

from medinfer.inference.scoring import combine, combine_all

VALUES = [-1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.9, 1.0]


@pytest.mark.parametrize("x,y", itertools.product(VALUES, repeat=2))
def test_result_stays_in_range(x, y):
    assert -1.0 <= combine(x, y) <= 1.0


@pytest.mark.parametrize("x,y", itertools.product(VALUES, repeat=2))
def test_commutative(x, y):
    assert combine(x, y) == pytest.approx(combine(y, x))


@pytest.mark.parametrize("x", VALUES)
def test_zero_is_no_information(x):
    assert combine(x, 0.0) == pytest.approx(x)


def test_certainty_is_absorbing():
    assert combine(1.0, 0.4) == 1.0
    assert combine(-1.0, -0.4) == -1.0


def test_agreeing_evidence_strengthens():
    assert combine(0.5, 0.5) > 0.5
    assert combine(-0.5, -0.5) < -0.5


def test_equal_and_opposite_cancel():
    assert combine(0.6, -0.6) == pytest.approx(0.0)


def test_combine_all_empty_is_zero():
    assert combine_all([]) == 0.0


def test_combine_all_single_value():
    assert combine_all([0.42]) == pytest.approx(0.42)
