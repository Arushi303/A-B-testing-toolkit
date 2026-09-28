import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from data_utils import generate_sample_data
from stats_engine import interpret, recommend_test, run_anova, run_chi_square, run_ttest

st.set_page_config(page_title="A/B Testing & Hypothesis Testing Toolkit", layout="wide")

st.title(" A/B Testing & Hypothesis Testing Toolkit")
st.caption(
    "Upload two or more groups, pick a metric, and get a statistically sound,"
    "verdict on whether the difference is real/significant."
)

# Load data block
st.sidebar.header("1. Load data")
source = st.sidebar.radio("Data source", ["Use sample data", "Upload CSV"])

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info(" Upload a CSV in the sidebar, or switch to sample data to try it out.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = generate_sample_data()
    st.sidebar.success("Using simulated website A/B test data (control vs treatment)")

#  Choose columns block
st.sidebar.header("2. Choose columns")
columns = list(df.columns)
default_group_idx = columns.index("variant") if "variant" in columns else 0
group_col = st.sidebar.selectbox("Group column", columns, index=default_group_idx)

metric_options = [c for c in columns if c != group_col]
default_metric = "converted" if "converted" in metric_options else metric_options[0]
metric_col = st.sidebar.selectbox(
    "Metric to test", metric_options, index=metric_options.index(default_metric)
)

alpha = st.sidebar.slider("Significance level (\u03b1)", 0.01, 0.10, 0.05, 0.01)

groups = sorted(df[group_col].dropna().unique().tolist())
n_groups = len(groups)

# A numeric column with only 2 unique values (e.g. a 0/1 conversion flag) behaves
# like a categorical outcome for testing purposes, not a continuous metric.
is_numeric_dtype = pd.api.types.is_numeric_dtype(df[metric_col])
is_binary_numeric = is_numeric_dtype and df[metric_col].dropna().nunique() == 2
use_continuous_test = is_numeric_dtype and not is_binary_numeric

st.subheader("Data preview")
st.dataframe(df.head(), width='stretch')

if n_groups < 2:
    st.error(f"'{group_col}' needs at least 2 groups to compare \u2014 found {n_groups}.")
    st.stop()

# ---------- 3. Descriptive stats + visual ----------
st.subheader(f"Comparing `{metric_col}` across `{group_col}` ({n_groups} groups)")

left, right = st.columns(2)

with left:
    if use_continuous_test:
        desc = df.groupby(group_col)[metric_col].agg(["count", "mean", "std"]).round(4)
    else:
        desc = df.groupby(group_col)[metric_col].agg(["count"])
        if is_numeric_dtype:
            desc["rate"] = df.groupby(group_col)[metric_col].mean().round(4)
    st.dataframe(desc, width='stretch')

with right:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if use_continuous_test:
        sns.boxplot(data=df, x=group_col, y=metric_col, ax=ax)
    elif is_numeric_dtype:
        rate_df = df.groupby(group_col)[metric_col].mean().reset_index()
        sns.barplot(data=rate_df, x=group_col, y=metric_col, ax=ax)
        ax.set_ylabel(f"Mean {metric_col}")
    else:
        sns.countplot(data=df, x=group_col, hue=metric_col, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

# ---------- 4. Run the right test ----------
st.subheader("Test results")

test_choice = recommend_test(use_continuous_test, n_groups)
st.write(f"Recommended test based on your data shape: **{test_choice}**")

if test_choice == "t-test":
    g_a = df[df[group_col] == groups[0]][metric_col]
    g_b = df[df[group_col] == groups[1]][metric_col]
    result = run_ttest(g_a, g_b)

    c1, c2, c3 = st.columns(3)
    c1.metric("t-statistic", f"{result['statistic']:.3f}")
    c2.metric("p-value", f"{result['p_value']:.4f}")
    c3.metric(f"Mean diff ({groups[0]} \u2212 {groups[1]})", f"{result['mean_diff']:.3f}")
    st.write(
        f"95% CI for the difference: [{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]"
    )
    st.caption(f"Welch-Satterthwaite degrees of freedom: {result['df']:.1f}")
    p_value = result["p_value"]

elif test_choice == "anova":
    samples = [df[df[group_col] == g][metric_col] for g in groups]
    result = run_anova(samples)

    c1, c2 = st.columns(2)
    c1.metric("F-statistic", f"{result['statistic']:.3f}")
    c2.metric("p-value", f"{result['p_value']:.4f}")
    p_value = result["p_value"]

else:  # chi-square
    contingency = pd.crosstab(df[group_col], df[metric_col])
    st.write("Contingency table:")
    st.dataframe(contingency, width='stretch')

    result = run_chi_square(contingency)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-square statistic", f"{result['statistic']:.3f}")
    c2.metric("p-value", f"{result['p_value']:.4f}")
    c3.metric("Degrees of freedom", result["dof"])
    if result["low_expected_count_warning"]:
        st.caption(
            "\u26a0\ufe0f Some expected cell counts are below 5 \u2014 the chi-square "
            "approximation may be less reliable here. Consider Fisher's exact test "
            "for small samples."
        )
    p_value = result["p_value"]

# ---------- 5. Plain-language verdict ----------
st.subheader("What does this mean?")
is_significant, message = interpret(p_value, alpha)
if is_significant:
    st.success(f"\u2705 {message}")
else:
    st.warning(f"\u26a0\ufe0f {message}")

st.caption(
    "Reminder: statistical significance isn't the same as practical significance. "
    "A tiny, real difference can still be 'significant' with enough data \u2014 always "
    "sanity check the effect size (the mean difference or rate difference above) "
    "alongside the p-value."
)
