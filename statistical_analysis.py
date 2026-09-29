"""
Statistical significance analysis of top ANOVA-ranked features.

For each feature in the ANOVA ranking, tests whether the feature
distribution differs significantly between the control and pathological
groups.

Normality is assessed with the Shapiro-Wilk test for each group
independently. If both groups are normal (p >= 0.05), Welch's t-test
is applied; otherwise, the Mann-Whitney U test is used.

Results are saved to a CSV file and printed to the console.

Usage
-----
Run from the project root after the main pipeline has produced
saved_models/df_features.joblib and saved_models/f_score_ranking.joblib:

    python statistical_analysis.py

Output
------
- Console: table of results sorted by F-score (most discriminative first)
- statistical_analysis.csv — full results table

Available Functions
-------------------
[Public]
run
    Load the feature table and the feature ranking (ANOVA or manual),
    test each feature for a significant difference between the control
    and pathological groups (Welch's t-test or Mann-Whitney U, chosen per
    feature based on normality), apply FDR correction across all tests,
    and save the results to statistical_analysis.csv.

------------------
[Private]

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import joblib
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

DF_FEATURES_PATH     = "saved_models_out_NaN/df_features.joblib"
F_SCORE_RANKING_PATH = "saved_models_out_NaN/f_score_ranking.joblib"
OUTPUT_PATH          = "saved_models_out_NaN/statistical_analysis.csv"

# Number of top features to analyse (None = all) — only used in "auto" mode
N_TOP_FEATURES       = None

# Significance level
ALPHA                = 0.05

# ── Feature selection mode ────────────────────────────────────────────
# "manual" → use MANUAL_FEATURES list below (useful for LOSO analysis
#             where f_score_ranking.joblib is not available)
FEATURE_SOURCE = "manual"

MANUAL_FEATURES = ['CH1_0_Autocorrelation', 'CH1_0_Centroid',
       'CH1_0_ECDF Percentile Count_0', 'CH1_0_ECDF Percentile Count_1',
       'CH1_0_Entropy', 'CH1_0_Kurtosis', 'CH1_0_Negative turning points',
       'CH1_0_Neighbourhood peaks', 'CH1_0_Positive turning points',
       'CH1_0_Skewness', 'CH1_0_Slope', 'CH1_0_Zero crossing rate',
       'CH2_0_Autocorrelation', 'CH2_0_Centroid',
       'CH2_0_ECDF Percentile Count_0', 'CH2_0_ECDF Percentile Count_1',
       'CH2_0_Entropy', 'CH2_0_Kurtosis', 'CH2_0_Negative turning points',
       'CH2_0_Neighbourhood peaks', 'CH2_0_Positive turning points',
       'CH2_0_Skewness', 'CH2_0_Slope', 'CH2_0_Zero crossing rate',
       'CH3_0_Autocorrelation', 'CH3_0_Centroid',
       'CH3_0_ECDF Percentile Count_0', 'CH3_0_ECDF Percentile Count_1',
       'CH3_0_Entropy', 'CH3_0_Kurtosis', 'CH3_0_Negative turning points',
       'CH3_0_Neighbourhood peaks', 'CH3_0_Positive turning points',
       'CH3_0_Skewness', 'CH3_0_Slope', 'CH3_0_Zero crossing rate',
       'CH4_0_Autocorrelation', 'CH4_0_Centroid',
       'CH4_0_ECDF Percentile Count_0', 'CH4_0_ECDF Percentile Count_1',
       'CH4_0_Entropy', 'CH4_0_Kurtosis', 'CH4_0_Negative turning points',
       'CH4_0_Neighbourhood peaks', 'CH4_0_Positive turning points',
       'CH4_0_Skewness', 'CH4_0_Slope', 'CH4_0_Zero crossing rate',
       'CH1_Dominant Frequency (CPM)', 'CH1_Frequency Instability Coefficient',
       'CH1_Power Instability Coefficient', 'CH1_Percent Power [2-4 CPM]',
       'CH1_Power Density Ratio', 'CH2_Dominant Frequency (CPM)',
       'CH2_Frequency Instability Coefficient',
       'CH2_Power Instability Coefficient', 'CH2_Percent Power [2-4 CPM]',
       'CH2_Power Density Ratio', 'CH3_Dominant Frequency (CPM)',
       'CH3_Frequency Instability Coefficient',
       'CH3_Power Instability Coefficient', 'CH3_Percent Power [2-4 CPM]',
       'CH3_Power Density Ratio', 'CH4_Dominant Frequency (CPM)',
       'CH4_Frequency Instability Coefficient',
       'CH4_Power Instability Coefficient', 'CH4_Percent Power [2-4 CPM]',
       'CH4_Power Density Ratio', 'CH1_morlet_sim',
       'CH2_morlet_sim', 'CH3_morlet_sim', 'CH4_morlet_sim',
       'start_time_sec_feature', 'is_long']

# Alternative, smaller manual feature list, keeping only features with significant raw p-value:
# MANUAL_FEATURES = ['CH1_0_Autocorrelation', 'CH1_0_Centroid', 'CH2_0_Centroid', 'CH2_0_Zero crossing rate',
#                    'CH4_0_Slope', 'CH1_Power Density Ratio', 'CH2_Power Instability Coefficient', 'CH2_Percent Power [2-4 CPM]',
#                    'CH4_Percent Power [2-4 CPM]', 'CH2_morlet_sim', 'CH3_morlet_sim']


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    Load the feature table and the feature ranking (ANOVA or manual),
    test each feature for a significant difference between the control
    and pathological groups (Welch's t-test or Mann-Whitney U, chosen per
    feature based on normality), apply FDR correction across all tests,
    and save the results to statistical_analysis.csv.
    """
    # ── 1. Load data ────────────────────────────────────────────────────
    df_features = joblib.load(DF_FEATURES_PATH)

    print(df_features.columns)

    # ── Select features to test ──────────────────────────────────────────
    if FEATURE_SOURCE == "manual":
        ranked_names = [f for f in MANUAL_FEATURES if f in df_features.columns]
        ranked_scores = [np.nan] * len(ranked_names)
        print(f"[load] Using {len(ranked_names)} manually defined features")
    else:
        f_score_ranking = joblib.load(F_SCORE_RANKING_PATH)
        ranked_names = [feat for feat, _ in f_score_ranking]
        ranked_scores = [score for _, score in f_score_ranking]

        if N_TOP_FEATURES is not None:
            ranked_names = ranked_names[:N_TOP_FEATURES]
            ranked_scores = ranked_scores[:N_TOP_FEATURES]

        ranked_names = [f for f in ranked_names if f in df_features.columns]
        ranked_scores = [s for f, s in zip(ranked_names, ranked_scores)
                         if f in df_features.columns]
        print(f"[load] Loaded f_score_ranking — testing {len(ranked_names)} features")

    # ── 2. Aggregate by subject ───────────────────────────────────────────
    # Each subject contributes multiple segments — treating segments as
    # independent observations violates the independence assumption of both
    # Welch's t-test and Mann-Whitney U. Aggregating by subject (median
    # across segments) produces one observation per subject, ensuring
    # independence and making the test a proper subject-level comparison.
    subject_labels = df_features.groupby("subject_id")["label"].first()
    df_subject = df_features.groupby("subject_id")[ranked_names].median()
    df_subject["label"] = subject_labels

    ctrl = df_subject[df_subject["label"] == 0]
    patho = df_subject[df_subject["label"] == 1]

    print(f"Groups (subject level): Control n={len(ctrl)} subjects, "
          f"Pathological n={len(patho)} subjects")
    print(f"Features to test: {len(ranked_names)}\n")

    # ── 3. Run tests ─────────────────────────────────────────────────────
    rows = []

    for feat, f_score in zip(ranked_names, ranked_scores):
        x_ctrl = ctrl[feat].dropna().values.astype(float)
        x_patho = patho[feat].dropna().values.astype(float)

        if len(x_ctrl) < 3 or len(x_patho) < 3:
            continue

        # ── Normality — Shapiro-Wilk independently for each group ────────
        _, p_normal_ctrl = shapiro(x_ctrl)
        _, p_normal_patho = shapiro(x_patho)
        both_normal = (p_normal_ctrl >= ALPHA) and (p_normal_patho >= ALPHA)

        # ── Statistical test ──────────────────────────────────────────────
        if both_normal:
            test_name = "Welch t-test"
            stat, p_value = ttest_ind(x_ctrl, x_patho, equal_var=False)
        else:
            test_name = "Mann-Whitney U"
            stat, p_value = mannwhitneyu(x_ctrl, x_patho, alternative="two-sided")

        rows.append({
            "feature": feat,
            "f_score": round(f_score, 4),
            "p_normal_ctrl": round(p_normal_ctrl, 4),
            "p_normal_patho": round(p_normal_patho, 4),
            "both_normal": both_normal,
            "test": test_name,
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < ALPHA,
            "ctrl_mean": x_ctrl.mean(),
            "ctrl_std": round(x_ctrl.std(), 6),
            "patho_mean": round(x_patho.mean(), 6),
            "patho_std": round(x_patho.std(), 6),
        })

    df_results = pd.DataFrame(rows)

    # ── 4. FDR correction (Benjamini-Hochberg) ────────────────────────────
    # When testing many features simultaneously, the probability of obtaining
    # at least one false positive by chance increases. FDR correction controls
    # the expected proportion of false positives among the significant results.
    reject, p_corrected, _, _ = multipletests(
        df_results["p_value"].values,
        alpha=ALPHA,
        method="fdr_bh",
    )
    df_results["p_corrected"] = p_corrected.round(6)
    df_results["significant_corrected"] = reject

    # ── 5. Print results ─────────────────────────────────────────────────
    n_sig_raw = df_results["significant"].sum()
    n_sig_fdr = df_results["significant_corrected"].sum()
    print(f"{'=' * 145}")
    print(f"  Statistical significance — α = {ALPHA}  |  "
          f"Significant (raw): {n_sig_raw}/{len(df_results)}  |  "
          f"Significant (FDR corrected): {n_sig_fdr}/{len(df_results)}")
    print(f"{'=' * 145}")
    print(f"  {'#':>3}  {'Feature':<45} {'F-score':>8}  {'Test':<16} "
          f"{'p_value':>10}  {'Sig(raw)':>9}  {'p_fdr':>10}  {'Sig(fdr)':>9}  "
          f"{'Ctrl mean':>10}  {'Patho mean':>10}  "
          f"{'p_norm_ctrl':>12}  {'p_norm_patho':>13}")
    print("  " + "-" * 160)

    for rank, (_, row) in enumerate(df_results.iterrows(), 1):
        sig_raw = "✓" if row["significant"] else " "
        sig_fdr = "✓" if row["significant_corrected"] else " "
        print(f"  {rank:>3}. {row['feature']:<45} "
              f"{row['f_score']:>8.2f}  "
              f"{row['test']:<16} "
              f"{row['p_value']:>10.6f}  "
              f"{sig_raw:>9}  "
              f"{row['p_corrected']:>10.6f}  "
              f"{sig_fdr:>9}  "
              f"{row['ctrl_mean']:>14.5e}  "
              f"{row['patho_mean']:>14.5e}  "
              f"{row['p_normal_ctrl']:>12.4f}  "
              f"{row['p_normal_patho']:>13.4f}")

    # ── 6. Save ──────────────────────────────────────────────────────────
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVE] {OUTPUT_PATH}")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()