"""
Generating a realistic sample A/B test dataset, so the app
has something meaningful to show before the user uploads their own CSV.
"""
import numpy as np
import pandas as pd


def generate_sample_data(seed: int = 42, n_per_group: int = 600) -> pd.DataFrame:
    """
    Simulates a website A/B test: a control landing page vs. a treatment
    (redesigned) landing page, measuring two metrics:
      - time_spent_sec: continuous  -> analyzed with a t-test
      - converted: binary (0/1)     -> analyzed with a chi-square test

    The treatment is given a modest, deliberate real lift in both metrics
    so the toolkit has a genuine (not purely random) effect to detect.
    """
    rng = np.random.default_rng(seed)

    control_time = rng.normal(120, 30, size=n_per_group)
    treatment_time = rng.normal(128, 30, size=n_per_group)

    control_conversion = rng.binomial(1, 0.10, size=n_per_group)
    treatment_conversion = rng.binomial(1, 0.15, size=n_per_group)

    df = pd.DataFrame({
        "user_id": range(1, 2 * n_per_group + 1),
        "variant": ["control"] * n_per_group + ["treatment"] * n_per_group,
        "time_spent_sec": np.concatenate([control_time, treatment_time]).round(1),
        "converted": np.concatenate([control_conversion, treatment_conversion]),
    })

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
