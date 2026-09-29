"""
SHAP (SHapley Additive exPlanations) analysis for trained ML models.

Computes SHAP values for each model trained on the full feature set and
produces summary plots that show:
  - Which features have the highest overall importance (beeswarm plot)
  - How each feature influences the prediction (positive = towards pathological)
  - Per-subject waterfall plots showing why the model classified each
    test segment as it did

For each model, two KDE pairplots are produced to visualise class separation:
  - Top 4 features by mean absolute SHAP value — reflects the model's own
    assessment of feature importance after training, capturing non-linear
    effects and feature interactions.
  - Top 4 features by ANOVA F-score (SelectKBest) — reflects univariate
    discriminability, ordered by the true F-score ranking computed during
    the sweep (loaded from saved_models/f_score_ranking.joblib). Comparing
    these two plots shows whether the univariate ANOVA criterion and the
    model-based SHAP importance agree on which features best separate the
    two classes.

Unlike SelectKBest (which evaluates features independently before training),
SHAP analyses the model after training and captures feature interactions
and non-linear effects.

Usage
-----
1. Run main.py at least once to generate saved_models/.
2. Run:  python shap_analysis.py

Results are saved to shap_analysis/.

Available Functions
-------------------
[Public]
run
    Load saved models and test set, compute SHAP values, and save plots
    for every model in MODELS_TO_ANALYSE (or all saved models if None).

------------------
[Private]
_get_explainer
    Build the appropriate SHAP explainer for each model type (TreeExplainer
    for Random Forest, LinearExplainer for Logistic Regression, and
    KernelExplainer as a model-agnostic fallback) and compute SHAP values
    on the test set.
_plot_summary
    Beeswarm summary plot of the top_n most important features by SHAP
    value, coloured by feature value.
_plot_scatter_top
    4x4 KDE pairplot of the four features with the highest mean absolute
    SHAP value, showing marginal and joint distributions per class.
_plot_scatter_selectk
    4x4 KDE pairplot of the top four features selected by SelectKBest
    (ANOVA F-score), for comparison against the SHAP-based ranking.

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import seaborn as sns
from sklearn.feature_selection import f_classif
# internal imports


# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

_HERE      = os.path.dirname(os.path.abspath(__file__))

SEED = 44   # which top_splits/seed_<SEED>/ folder to analyse

SAVED_MODELS_DIR = os.path.join(_HERE, "saved_models")
MODELS_DIR        = os.path.join(_HERE, "top_splits", f"seed_{SEED}")
SAVE_DIR          = os.path.join(_HERE, "shap_analysis_best")

# Feature set to analyse — 'full' or 'best_k'
FEATURE_SET = "best_k"

# Number of top features to show in the summary plot
TOP_N = 20

# Models to analyse — set to None to analyse all saved models
MODELS_TO_ANALYSE = None   # e.g. ["Logistic Regression", "Random Forest"]

# Number of samples for KernelExplainer approximation (used for SVM and KNN).
# Higher = more accurate SHAP values but slower computation.
# None = adaptive formula: 2 * n_features + 2048 (recommended)
# For full feature set (~221 features): 2*221 + 2048 = 2490 samples
KERNEL_EXPLAINER_NSAMPLES = None


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    Load saved models and test set, compute SHAP values, and save plots.
    """
    print(f"\n{'='*60}")
    print(f"  SHAP analysis  ({FEATURE_SET} features)")
    print(f"{'='*60}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load artefacts ────────────────────────────────────────────────────
    print("[load] Reading saved artefacts …")
    best_test               = joblib.load(os.path.join(MODELS_DIR, "best_test.joblib"))
    best_features_per_model = joblib.load(os.path.join(MODELS_DIR, "best_features_per_model.joblib"))
    best_k_per_model        = joblib.load(os.path.join(MODELS_DIR, "best_k_per_model.joblib"))

    # Load F-score ranking computed during sweep — used to order SelectKBest
    # KDE plot by true importance rather than DataFrame column order
    f_score_ranking_path = os.path.join(MODELS_DIR, "f_score_ranking.joblib")
    f_score_ranking = joblib.load(f_score_ranking_path) if os.path.exists(f_score_ranking_path) else None

    # Load train set to fit the explainer background
    df_features = joblib.load(os.path.join(SAVED_MODELS_DIR, "df_features.joblib"))
    best_test_ids = joblib.load(os.path.join(MODELS_DIR, "best_test_ids.joblib"))
    df_train = df_features[~df_features["subject_id"].isin(best_test_ids)]

    # ── Determine which models to analyse ────────────────────────────────
    models_available = list(best_k_per_model.keys())
    models = MODELS_TO_ANALYSE or models_available

    for model_name in models:
        print(f"\n{'─'*55}")
        print(f"  {model_name}")
        print(f"{'─'*55}")

        # Load model
        if FEATURE_SET == "full":
            path = os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_')}_full.joblib")
        else:
            k = best_k_per_model[model_name]
            path = os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_')}_bestk{k}.joblib")

        # In both cases, the pipeline expects the FULL set of raw features —
        # the "selector" (SelectKBest) reduces internally to the k best ones.
        feat_cols = [c for c in best_test.columns
                     if c not in {"label", "subject_id", "_is_clean",
                                  "n_segments_total", "duration_sec"}]

        pipeline = joblib.load(path)
        print(f"[load] {path}")

        X_test_raw = best_test[feat_cols].values.astype(float)
        X_train_raw = df_train[feat_cols].values.astype(float)

        preprocessor = pipeline[:-1]
        X_test_scaled = preprocessor.transform(X_test_raw)
        X_train_scaled = preprocessor.transform(X_train_raw)


        cols_after_corr = feat_cols
        if "corr" in pipeline.named_steps:
            corr_filter = pipeline.named_steps["corr"]
            cols_after_corr = [feat_cols[i] for i in corr_filter.cols_to_keep_]

        if "selector" in pipeline.named_steps:
            support_mask = pipeline.named_steps["selector"].get_support()
            used_cols = [c for c, keep in zip(cols_after_corr, support_mask) if keep]
        else:
            used_cols = cols_after_corr

        X_test_df = pd.DataFrame(X_test_scaled, columns=used_cols)
        X_train_df = pd.DataFrame(X_train_scaled, columns=used_cols)

        y_train     = df_train["label"].values.astype(int)

        # ── Build SHAP explainer ──────────────────────────────────────────
        explainer, shap_values = _get_explainer(
            model_name, pipeline, X_train_df, X_test_df
        )
        # Normalise to 2D (n_samples, n_features) — class 1 values
        # TreeExplainer may return 3D (n_samples, n_features, n_classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # ── Summary plot (beeswarm) — computed on test set ────────────────
        _plot_summary(shap_values, X_test_df, model_name, SAVE_DIR, TOP_N)

        # ── KDE pairplot — top 4 SHAP features, plotted on TRAIN set ─────
        # Training data is used here (508 segments vs 155 test) to obtain
        # more stable density estimates and a more representative picture
        # of the feature distributions for each class.
        _plot_scatter_top(shap_values, X_train_df, y_train, model_name, SAVE_DIR)

        # ── KDE pairplot — top 4 SelectKBest features, on TRAIN set ──────
        _plot_scatter_selectk(
            X_test_df       = X_train_df,
            y_test          = y_train,
            model_name      = model_name,
            k_features      = best_features_per_model.get(model_name, []),
            save_dir        = SAVE_DIR,
            f_score_ranking = f_score_ranking,
        )

    print(f"\n[done] All plots saved to {SAVE_DIR}/")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_explainer(
    model_name:   str,
    pipeline:     object,
    X_train_df:   pd.DataFrame,
    X_test_df:    pd.DataFrame,
):
    """
    Build the appropriate SHAP explainer for each model type and compute
    SHAP values on the test set.

    - Random Forest : TreeExplainer  (exact, fast)
    - All others    : LinearExplainer for linear models,
                      KernelExplainer as fallback (slower but universal)

    :param model_name: model name, used to pick the explainer type
    :param pipeline:   fitted sklearn Pipeline (scaler [+corr] [+selector] + clf)
    :param X_train_df: scaled training features, used as explainer background
    :param X_test_df:  scaled test features to compute SHAP values for
    :return: (explainer, shap_values) where shap_values has shape
             (n_samples, n_features) for binary classification (values for class 1)
    """
    clf = pipeline.named_steps["clf"]

    if "RandomForest" in type(clf).__name__:
        print(f"  [shap] Using TreeExplainer")
        explainer   = shap.TreeExplainer(clf)
        shap_vals   = explainer.shap_values(X_test_df)
        # TreeExplainer returns list [class0, class1] — take class 1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    elif "LogisticRegression" in type(clf).__name__:
        print(f"  [shap] Using LinearExplainer")
        explainer = shap.LinearExplainer(clf, X_train_df)
        shap_vals = explainer.shap_values(X_test_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    else:
        # KernelExplainer — model-agnostic but slower
        # Use a summary of the training data as background
        print(f"  [shap] Using KernelExplainer (may be slow)")
        #background = shap.kmeans(X_train_df, 50)
        background = X_train_df
        predict_fn = lambda x: clf.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        n_samples = KERNEL_EXPLAINER_NSAMPLES or (2 * X_test_df.shape[1] + 2048)
        shap_vals = explainer.shap_values(X_test_df, nsamples=n_samples)
    return explainer, shap_vals


def _plot_summary(
    shap_values: np.ndarray,
    X_test_df:   pd.DataFrame,
    model_name:  str,
    save_dir:    str,
    top_n:       int = 20,
) -> None:
    """
    Beeswarm summary plot — shows the top_n most important features,
    with each dot representing one test segment coloured by feature value
    (red = high, blue = low). The x-axis shows the SHAP value: positive
    values push the prediction towards pathological, negative towards control.

    :param shap_values: (n_samples, n_features) SHAP values for class 1
    :param X_test_df:   scaled test features as DataFrame
    :param model_name:  used in title and filename
    :param save_dir:    output directory
    :param top_n:       number of top features to show
    """
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    shap.summary_plot(
        shap_values,
        X_test_df,
        max_display = top_n,
        show        = False,
        plot_type   = "dot",
    )
    plt.title(f"{model_name} — SHAP summary (top {top_n} features)\n"
              f"Positive SHAP → pathological  |  Negative SHAP → control",
              fontsize=10)
    plt.tight_layout()
    fname = os.path.join(save_dir,
                         f"shap_summary_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] Summary plot saved → {fname}")


def _plot_scatter_top(
    shap_values: np.ndarray,
    X_test_df:   pd.DataFrame,
    y_test:      np.ndarray,
    model_name:  str,
    save_dir:    str,
) -> None:
    """
    4×4 KDE pairplot of the four features with the highest mean absolute
    SHAP value.

    Grid layout:
      diagonal    → KDE of each feature individually per class
      off-diagonal → joint KDE contours + scatter of each feature pair

    This shows both the marginal distributions (how each feature alone
    separates the classes) and the joint distributions (whether pairs of
    features together provide better separation).

    :param shap_values: (n_samples, n_features) SHAP values for class 1
    :param X_test_df:   scaled test features as DataFrame
    :param y_test:      true labels (0=control, 1=pathological)
    :param model_name:  used in title and filename
    :param save_dir:    output directory
    """

    # Identify top 4 features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top4_idx      = np.argsort(mean_abs_shap)[::-1][:4]
    top4_feats    = [X_test_df.columns[i] for i in top4_idx]

    df_plot = pd.DataFrame(
        {f: X_test_df[f].values for f in top4_feats} |
        {"Class": ["Control" if l == 0 else "Pathological" for l in y_test]}
    )

    palette = {"Control": "#1D9E75", "Pathological": "#D85A30"}

    g = sns.PairGrid(df_plot, vars=top4_feats, hue="Class",
                     palette=palette, diag_sharey=False)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidths=1.5)
    try:
        g.map_offdiag(sns.kdeplot, levels=4, alpha=0.6, linewidths=1.2,
                      warn_singular=False)
    except (IndexError, ValueError) as e:
        print(f"  [WARN] KDE offdiag failed ({e}) — skipping contours")
    g.map_offdiag(sns.scatterplot, alpha=0.4, s=25,
                  edgecolors="white", linewidths=0.3)
    g.add_legend(title="Class", fontsize=9)

    g.figure.suptitle(
        f"{model_name} — top 4 SHAP features",
        y=1.02, fontsize=11,
    )

    fname = os.path.join(save_dir,
                         f"shap_kde_{model_name.replace(' ', '_')}.png")
    g.figure.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] KDE pairplot saved → {fname}")
    print(f"         Top 4 features: {top4_feats}")


def _plot_scatter_selectk(
    X_test_df:       pd.DataFrame,
    y_test:          np.ndarray,
    model_name:      str,
    k_features:      list,
    save_dir:        str,
    f_score_ranking: list = None,
) -> None:
    """
    4×4 KDE pairplot of the top 4 features selected by SelectKBest
    (ANOVA F-score), ordered by F-score so position 1 is genuinely
    the most discriminative feature.

    Same layout as _plot_scatter_top but uses the features selected by
    the sweep instead of the SHAP-based features. Comparing the two plots
    shows whether the univariate ANOVA criterion and the model-based SHAP
    importance agree on which features carry discriminative information.

    :param X_test_df:       scaled test features as DataFrame (full feature set)
    :param y_test:          true labels (0=control, 1=pathological)
    :param model_name:      used in title and filename
    :param k_features:      list of features selected by SelectKBest sweep
                            (in DataFrame column order — re-sorted by F-score)
    :param save_dir:        output directory
    :param f_score_ranking: list of (feature, f_score) tuples sorted by F-score
                            descending, as returned by sweep_feature_selection
                            and saved to saved_models/f_score_ranking.joblib.
                            If None, F-score is recalculated locally (fallback).
    """


    if len(k_features) < 4:
        print(f"  [shap] Not enough SelectKBest features to plot — skipping")
        return

    missing = [f for f in k_features if f not in X_test_df.columns]
    if missing:
        print(f"  [shap] SelectKBest features not in X_test_df — skipping")
        return

    # Order by the F-score ranking computed during the sweep (same ranking
    # used to train the model). If not available, fall back to recalculating.
    if f_score_ranking is not None:
        ranked = [(feat, score) for feat, score in f_score_ranking
                  if feat in k_features]
    else:
        X_k = X_test_df[k_features]
        f_scores, _ = f_classif(X_k, y_test)
        ranked = sorted(
            [(feat, score) for feat, score in zip(k_features, f_scores)
             if not np.isnan(score)],
            key=lambda x: x[1], reverse=True
        )
    top4_feats = [feat for feat, _ in ranked[:4]]

    # Print all k features ordered by F-score
    print(f"\n  SelectKBest features ranked by F-score ({model_name}):")
    for rank, (feat, score) in enumerate(ranked, 1):
        marker = " ← top 4" if rank <= 4 else ""
        print(f"    {rank:>3}. F={score:>8.2f}  {feat}{marker}")

    # Warn about zero-variance features in any class — KDE will be skipped
    # for those features but the plot will still be saved
    class_labels = {0: "Control", 1: "Pathological"}
    for f in top4_feats:
        for cls, cls_name in class_labels.items():
            mask = y_test == cls
            if X_test_df[f][mask].std() == 0:
                print(f"  [WARN] '{f}' has zero variance in {cls_name} "
                      f"— KDE cannot be estimated for this group in the plot")

    df_plot = pd.DataFrame(
        {f: X_test_df[f].values for f in top4_feats} |
        {"Class": ["Control" if l == 0 else "Pathological" for l in y_test]}
    )
    for f in top4_feats:
        for cls, cls_name in class_labels.items():
            mask = y_test == cls
            vals = X_test_df[f][mask]
            print(f"  [DEBUG] {f} | {cls_name}: n={mask.sum()}, "
                  f"std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"unique={vals.nunique()}")

    palette = {"Control": "#1D9E75", "Pathological": "#D85A30"}

    g = sns.PairGrid(df_plot, vars=top4_feats, hue="Class",
                     palette=palette, diag_sharey=False)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidths=1.5,
               warn_singular=False)
    try:
        g.map_offdiag(sns.kdeplot, levels=4, alpha=0.6, linewidths=1.2,
                      warn_singular=False)
    except (IndexError, ValueError) as e:
        print(f"  [WARN] KDE offdiag failed ({e}) — skipping contours, scatter only")
    g.map_offdiag(sns.scatterplot, alpha=0.4, s=25,
                  edgecolors="white", linewidths=0.3)
    g.add_legend(title="Class", fontsize=9)

    g.figure.suptitle(
        f"{model_name} — top 4 SelectKBest features (ANOVA F-score)",
        y=1.02, fontsize=11,
    )

    fname = os.path.join(save_dir,
                         f"selectk_kde_{model_name.replace(' ', '_')}.png")
    g.figure.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] SelectKBest KDE pairplot saved → {fname}")
    print(f"         Top 4 features (by F-score): {top4_feats}")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()