"""
Core statistical testing functions for the A/B Testing Toolkit and.

Kept separate from the Streamlit UI layer(app.py) on purpose cz these functions
take plain pandas/numpy inputs and return plain dicts, so they can be unit
tested and reused without spinning up the app.
"""
import numpy as np
import pandas as pd
from scipy import stats


def recommend_test(is_continuous: bool, n_groups: int):
    """Suggest which statistical test fits the shape of the data."""
    if is_continuous and n_groups == 2:
        return "t-test"
    elif is_continuous and n_groups > 2:
        return "anova"
    else:
        return "chi-square"


def run_ttest(group_a: pd.Series, group_b: pd.Series, confidence= 0.95) -> dict:
    
    #Welch's t-test(independent two sample test)
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    n1 = len(group_a)
    n2 = len(group_b)
    v1 = group_a.var(ddof=1) #because of sample data ddof (n-1) for vairnace
    v2 = group_b.var(ddof=1)
    diff = group_a.mean() - group_b.mean()
    se = np.sqrt(v1 / n1 + v2 / n2)

    # Welch-Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    return {
        "test": "Welch's t-test",
        "statistic": stat,
        "p_value": p_value,
        "mean_diff": diff,
        "ci": (ci_low, ci_high),
        "confidence": confidence,
        "df": df,
    }


def run_anova(samples: list) -> dict:
    """One-way ANOVA for comparing 3 or more group means."""
    samples = [s.dropna() for s in samples]
    stat, p_value = stats.f_oneway(*samples)
    return {
        "test": "One-way ANOVA",
        "statistic": stat,
        "p_value": p_value,
    }


def run_chi_square(contingency_table: pd.DataFrame) -> dict:
    """Chi-square test of independence for categorical (or binary) outcomes."""
    stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    low_expected_count = bool((expected < 5).any())
    return {
        "test": "Chi-square test",
        "statistic": stat,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
        "low_expected_count_warning": low_expected_count,
    }


def interpret(p_value: float, alpha: float = 0.05) -> tuple:
    """Translate a p-value into a plain-English verdict. Returns (is_significant, message)."""
    if p_value < alpha:
        return True, (
            f"Statistically significant (p = {p_value:.4f} < \u03b1 = {alpha}). "
            "The difference between groups is unlikely to be due to random chance(Null hypothesis rejected)."
        )
    else:
        return False, (
            f"Not statistically significant (p = {p_value:.4f} \u2265 \u03b1 = {alpha}). "
            "We can't confidently rule out random chance as the explanation for the observed difference."
        )
