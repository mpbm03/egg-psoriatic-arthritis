"""
Age-adjusted group comparison (ANCOVA-style), comparing the group effect
on key features with and without age as a covariate.

For each feature, two models are fitted at the subject level:

  Model A (unadjusted):  feature ~ group
  Model B (age-adjusted): feature ~ group + age

If the group coefficient remains significant (and similar in magnitude)
in Model B, this suggests the group difference is not simply explained
by the age difference between groups. If the group coefficient becomes
non-significant, or drops substantially, once age is included, this
would suggest age may be a confounding factor.

Usage
-----
Run from the project root after main.py has produced
saved_models/df_features.joblib, and with age_map.csv available:

    python age_analysis.py

Output
------
- Console: p-values and group coefficients, with and without age, per feature
- age_covariate_ancova_results.csv — full results table

Available Functions
-------------------
[Public]
run
    Load the feature table and age map, fit the unadjusted and
    age-adjusted models for each feature in FEATURES_TO_TEST, and save
    the results to age_covariate_ancova_results.csv.

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import joblib
import statsmodels.formula.api as smf

# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

# The 11 features from the Statistical Analysis table (ordered by lowest
# raw p-value) — test all of them, not just an arbitrary subset
FEATURES_TO_TEST = [
    "CH2_Power Instability Coefficient",   # rank 1
    "CH3_morlet_sim",                       # rank 2
    "CH4_Percent Power [2-4 CPM]",          # rank 3
    "CH2_morlet_sim",                       # rank 4
    "CH2_Percent Power [2-4 CPM]",          # rank 5
    "CH4_0_Slope",                          # rank 6
    "CH1_Power Density Ratio",              # rank 7
    "CH1_0_Centroid",                       # rank 8
    "CH2_0_Centroid",                       # rank 9
    "CH2_0_Zero crossing rate",             # rank 10
    "CH1_0_Autocorrelation",                # rank 11
]

# Features that, in the original statistical analysis, used the
# Mann-Whitney U test (for failing the Shapiro-Wilk normality test) — for
# these, the group/age comparison is performed on ranks rather than raw
# values, to stay consistent with the non-parametric methodology already
# used originally.
NONPARAMETRIC_FEATURES = {
    "CH3_morlet_sim",
    "CH4_Percent Power [2-4 CPM]",
    "CH2_morlet_sim",
    "CH4_0_Slope",
    "CH1_Power Density Ratio",
    "CH2_0_Zero crossing rate",
    "CH1_0_Autocorrelation",
}


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    Load the feature table and age map, fit the unadjusted (feature ~
    group) and age-adjusted (feature ~ group + age) models for each
    feature in FEATURES_TO_TEST, and save the results to
    age_covariate_ancova_results.csv.
    """
    df_features = joblib.load("saved_models/df_features.joblib")

    # ── load the age map built from the questionnaires ────────────────────
    age_map = pd.read_csv("age_map.csv").set_index("subject_id")["age"]

    df_subj = (
        df_features.groupby("subject_id")
        .agg(
            label=("label", "first"),
            **{f: (f, "median") for f in FEATURES_TO_TEST},   # median, as in the original script
        )
        .reset_index()
    )
    df_subj["age"] = df_subj["subject_id"].map(age_map)

    # confirm that no subject is left without an age
    n_missing_age = df_subj["age"].isna().sum()
    if n_missing_age > 0:
        print(f"[WARNING] {n_missing_age} subject(s) with no mapped age — check before continuing")
        print(df_subj[df_subj["age"].isna()]["subject_id"].tolist())
        return
    df_subj["group"] = df_subj["label"].map({0: "Control", 1: "PsA"})

    print(f"{'=' * 80}")
    print(f"  Group effect comparison, with and without age adjustment")
    print(f"{'=' * 80}\n")
    print(f"{'Feature':<40} {'p (without age)':>15} {'p (with age, ANCOVA)':>25}")
    print("-" * 80)

    results = []

    for feat in FEATURES_TO_TEST:
        # ── convert to ranks when the original test was Mann-Whitney ──────
        # (OLS regression assumes normality; using ranks approximates the
        # Mann-Whitney logic while still allowing age to be included)
        if feat in NONPARAMETRIC_FEATURES:
            df_subj[f"{feat}_rank"] = df_subj[feat].rank()
            formula_var = f"Q('{feat}_rank')"
        else:
            formula_var = f"Q('{feat}')"

        # ── Model A: without age adjustment ────────────────────────────
        # cov_type="HC3" approximates Welch's t-test behaviour (does not
        # assume equal variances between groups), matching the original
        # script, which uses ttest_ind(..., equal_var=False)
        model_a = smf.ols(f"{formula_var} ~ C(group)", data=df_subj).fit(cov_type="HC3")
        p_without_age = model_a.pvalues["C(group)[T.PsA]"]

        # ── Model B: with age as a covariate ───────────────────────────
        model_b = smf.ols(f"{formula_var} ~ C(group) + age", data=df_subj).fit(cov_type="HC3")
        p_with_age = model_b.pvalues["C(group)[T.PsA]"]
        p_age = model_b.pvalues["age"]

        test_type = "rank (non-parametric)" if feat in NONPARAMETRIC_FEATURES else "OLS (parametric)"
        print(f"{feat:<40} [{test_type:<22}] {p_without_age:>10.4f} {p_with_age:>15.4f}")

        results.append({
            "feature": feat,
            "p_without_age": round(p_without_age, 4),
            "p_with_age_ancova": round(p_with_age, 4),
            "p_age_effect": round(p_age, 4),
            "coef_group_without_age": round(model_a.params["C(group)[T.PsA]"], 4),
            "coef_group_with_age": round(model_b.params["C(group)[T.PsA]"], 4),
        })

    df_result = pd.DataFrame(results)
    print(f"\n{'=' * 80}")
    print(df_result.to_string(index=False))
    df_result.to_csv("age_covariate_ancova_results.csv", index=False)
    print(f"\n[save] age_covariate_ancova_results.csv")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()