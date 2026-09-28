# 🧪 A/B Testing & Hypothesis Testing Toolkit

An interactive Streamlit app that takes any two (or more) groups of data and tells you — with real statistical rigor and a plain-English explanation — whether the difference between them is likely real or just noise.

Built to apply core statistics (hypothesis testing, confidence intervals, p-values) to a genuinely useful tool, rather than another one-off EDA notebook.

## What it does

1. Upload your own CSV, or use the built-in simulated website A/B test dataset (control vs. treatment landing page).
2. Pick a **group column** (e.g. control/treatment) and a **metric column** (e.g. conversion, time spent).
3. The app automatically picks the right statistical test based on the shape of your data:
   - **Welch's t-test** — 2 groups, continuous numeric metric
   - **One-way ANOVA** — 3+ groups, continuous numeric metric
   - **Chi-square test of independence** — categorical or binary (0/1) metric
4. You get the test statistic, p-value, a confidence interval, and a plain-language verdict — not just raw numbers.

## Tech stack

- **Python** — core logic
- **Pandas / NumPy** — data handling
- **SciPy** — statistical tests
- **Seaborn / Matplotlib** — visualization
- **Streamlit** — interactive web app
- **Pytest** — unit tests for the statistics engine

## Project structure

```
ab-testing-toolkit/
├── app.py                    # Streamlit UI
├── stats_engine.py           # Statistical test logic (pure functions, UI-independent)
├── data_utils.py             # Sample data generator
├── tests/
│   └── test_stats_engine.py  # Unit tests
├── requirements.txt
└── README.md
```

The statistics logic lives in `stats_engine.py`, completely separate from the Streamlit UI in `app.py`. That's a deliberate design choice: it means the core logic can be unit tested directly (see `tests/`) without needing to run the whole app, and it could be reused elsewhere (a CLI tool, a notebook, a different UI) with no changes.

## Running locally

```bash
git clone <your-repo-url>
cd ab-testing-toolkit
pip install -r requirements.txt
streamlit run app.py
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

## Deploying

This app is ready to deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub.
2. Go to share.streamlit.io, connect your GitHub account, and point it at `app.py`.
3. You'll get a live public link — add it to your resume/LinkedIn so recruiters can actually try it, not just read about it.

## The statistics, briefly

- **p-value** — the probability of seeing a difference this large (or larger) if there were actually no real difference between the groups. A small p-value (conventionally < 0.05) suggests the observed difference is unlikely to be random chance.
- **Confidence interval** — a range of plausible values for the true difference between groups. If it doesn't contain 0, that supports a real effect.
- **Why Welch's t-test, not Student's?** Welch's doesn't assume the two groups have equal variance, which is a safer default for real-world data where variances often differ between groups.
- **Why chi-square for binary metrics (like conversion)?** A 0/1 "converted" column is really a proportion, not a continuous quantity — chi-square (or a two-proportion test) is the more standard tool than a t-test for comparing conversion rates.

## Known limitations / possible extensions

Being upfront about these is part of the project, not a weakness:

- No effect-size reporting (e.g., Cohen's d) alongside the p-value yet — statistical significance isn't the same as practical significance.
- No sample size / power analysis calculator ("how many users would this test actually need?").
- Chi-square assumes reasonably large expected cell counts; the app warns when that assumption looks shaky, but doesn't yet fall back to Fisher's exact test automatically.
- Frequentist only — a Bayesian A/B testing view (probability that treatment beats control) would be a natural next addition.


