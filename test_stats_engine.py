"""
Unit tests for stats_engine.py.

Run with:  pytest tests/
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from stats_engine import interpret, recommend_test, run_anova, run_chi_square, run_ttest


def test_recommend_test_ttest():
    assert recommend_test(is_continuous=True, n_groups=2) == "t-test"


def test_recommend_test_anova():
    assert recommend_test(is_continuous=True, n_groups=3) == "anova"


def test_recommend_test_chi_square():
    assert recommend_test(is_continuous=False, n_groups=2) == "chi-square"


def test_ttest_detects_clear_difference():
    rng = np.random.default_rng(0)
    group_a = pd.Series(rng.normal(50, 5, 200))
    group_b = pd.Series(rng.normal(60, 5, 200))  # clearly different mean
    result = run_ttest(group_a, group_b)
    assert result["p_value"] < 0.05
    assert result["ci"][0] < result["ci"][1]


def test_ttest_no_real_difference():
    rng = np.random.default_rng(1)
    group_a = pd.Series(rng.normal(50, 5, 300))
    group_b = pd.Series(rng.normal(50, 5, 300))  # same distribution
    result = run_ttest(group_a, group_b)
    assert result["p_value"] > 0.05


def test_anova_detects_group_difference():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(10, 2, 100))
    b = pd.Series(rng.normal(10, 2, 100))
    c = pd.Series(rng.normal(15, 2, 100))  # clearly different
    result = run_anova([a, b, c])
    assert result["p_value"] < 0.05


def test_chi_square_runs_and_returns_valid_p():
    table = pd.DataFrame(
        {"converted": [50, 90], "not_converted": [450, 410]},
        index=["control", "treatment"],
    )
    result = run_chi_square(table)
    assert 0 <= result["p_value"] <= 1
    assert "low_expected_count_warning" in result


def test_interpret_significant():
    is_sig, msg = interpret(0.01, alpha=0.05)
    assert is_sig is True
    assert "Statistically significant" in msg


def test_interpret_not_significant():
    is_sig, msg = interpret(0.5, alpha=0.05)
    assert is_sig is False
    assert "Not statistically significant" in msg
