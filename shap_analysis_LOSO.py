"""
SHAP analysis aggregated across all LOSO folds.

For each classifier, computes SHAP values independently on each of the 25
LOSO folds' held-out test segments, using that fold's own fitted pipeline,
and concatenates the results into a single aggregated analysis — as if all
25 held-out subjects were evaluated together.

This is only valid if every fold's pipeline was trained on the SAME set of
feature columns. This holds automatically in 'full' mode (all 74 features,
no selection), and empirically also holds for KNN in 'bestk_sweep' mode
(the ANOVA-based selector converges on the same two features — Morlet
similarity of EGG2 and EGG3 — in all 25 folds). For any other
model/mode combination where the selected columns differ across folds,
the script detects this and skips aggregation for that model, printing a
warning instead of silently producing an incorrect result.

Usage
-----
Run from the project root after loso_evaluation.py has produced
loso_results/<mode>/saved_models/:

    python shap_analysis_LOSO.py

Output
------
- Console: per-model progress and warnings (skipped models due to
  inconsistent feature selection across folds)
- shap_summary_loso_<model>.png — SHAP summary plot, aggregated across folds
- shap_kde_loso_<model>.png     — KDE pairplot of the top 4 SHAP features

Available Functions
-------------------
[Public]
run
    For each available model, compute SHAP values on every LOSO fold's
    held-out test segments, aggregate them across all 25 folds (when the
    selected feature columns are consistent across folds), and save the
    summary and KDE pairplot figures.

------------------
[Private]
_get_explainer
    Pick the appropriate SHAP explainer for the classifier type
    (TreeExplainer, LinearExplainer, or KernelExplainer) and compute the
    SHAP values for the given test set.
_plot_summary
    Plot and save the aggregated SHAP summary (beeswarm) plot for one model.
_plot_scatter_top
    Plot and save a KDE pairplot of the top 4 SHAP features for one model,
    colored by class.

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

# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

_HERE          = os.path.dirname(os.path.abspath(__file__))
LOSO_SAVE_DIR  = os.path.join(_HERE, "loso_results")   # SAVE_DIR from loso_evaluation.py
SAVE_DIR       = os.path.join(_HERE, "shap_analysis_loso_KNN_best")

# 'full' or 'bestk_sweep' — must match the mode_slug used in the LOSO run
MODE_SLUG = "bestk_sweep"

TOP_N = 20
MODELS_TO_ANALYSE = None   # None = all available models

KERNEL_EXPLAINER_NSAMPLES = None   # None = adaptive formula

N_SUBJECTS = 25


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    For each available model, compute SHAP values on every LOSO fold's
    held-out test segments, aggregate them across all 25 folds (when the
    selected feature columns are consistent across folds), and save the
    summary and KDE pairplot figures.
    """
    print(f"\n{'=' * 60}")
    print(f"  Aggregated SHAP analysis — LOSO [{MODE_SLUG}]")
    print(f"{'=' * 60}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    models_root = os.path.join(LOSO_SAVE_DIR, MODE_SLUG, "saved_models")
    shared_root = os.path.join(models_root, "_shared")

    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"Not found: {models_root}")

    # discover model names from the existing subfolders
    available_models = sorted(
        d for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d)) and d != "_shared"
    )
    models = MODELS_TO_ANALYSE or available_models

    # discover folds from the folders in _shared
    fold_dirs = sorted(
        d for d in os.listdir(shared_root)
        if os.path.isdir(os.path.join(shared_root, d))
    )
    print(f"[info] {len(fold_dirs)} folds found in {shared_root}")

    for model_name in models:
        print(f"\n{'─' * 55}")
        print(f"  {model_name}")
        print(f"{'─' * 55}")

        model_dir = os.path.join(models_root, model_name)

        all_shap_values = []
        all_X_test       = []
        all_y_test        = []
        reference_columns = None
        columns_consistent = True

        for fold_dir in fold_dirs:
            fold_tag = fold_dir  # e.g. "fold_ID3"

            pipeline_path = os.path.join(model_dir, f"{fold_tag}_pipeline.joblib")
            if not os.path.exists(pipeline_path):
                print(f"  [SKIP] {fold_tag}: pipeline not found")
                continue

            pipeline = joblib.load(pipeline_path)
            columns = joblib.load(os.path.join(shared_root, fold_dir, "columns.joblib"))
            testdata = joblib.load(os.path.join(shared_root, fold_dir, "testdata.joblib"))
            X_test_fold, y_test_fold = testdata["X_test"], testdata["y_test"]

            # ── raw features, in the same order the pipeline was trained on ──
            X_test_raw = X_test_fold[columns].values.astype(float)

            # ── apply preprocessing, excluding the classifier ──────────────
            # (includes the "selector", if any — so the result may already
            #  have fewer columns than `columns`)
            preprocessor = pipeline[:-1]
            X_test_transformed = preprocessor.transform(X_test_raw)

            # ── determine the column names ACTUALLY used by the classifier ──
            # The pipeline is scaler -> corr -> selector -> clf: the "corr"
            # step (CorrelationFilter) already drops highly-correlated
            # columns BEFORE the selector ever sees them, so `columns` (the
            # original 74 raw feature names) and the selector's
            # get_support() mask (defined over the ~56 columns that survived
            # "corr") are NOT the same length and must not be zipped
            # directly — doing so silently mislabels the selected features.
            # cols_to_keep_ on the CorrelationFilter is what lets us recover
            # the intermediate (post-corr) column names, exactly as done in
            # loso_evaluation.py when it records `selected_feats` per fold.
            if "selector" in pipeline.named_steps:
                selector = pipeline.named_steps["selector"]
                if "corr" in pipeline.named_steps:
                    corr_filter = pipeline.named_steps["corr"]
                    cols_after_corr = list(np.array(columns)[corr_filter.cols_to_keep_])
                else:
                    cols_after_corr = columns
                support_mask = selector.get_support()
                selected_columns = [c for c, keep in zip(cols_after_corr, support_mask) if keep]
            else:
                selected_columns = columns

            # ── check consistency of the SELECTED features across folds ─────
            if reference_columns is None:
                reference_columns = selected_columns
            elif selected_columns != reference_columns:
                columns_consistent = False
                break

            X_test_df = pd.DataFrame(X_test_transformed, columns=reference_columns)

            # ── rebuild this fold's X_train for the SHAP background ─────────
            # (only needed for KernelExplainer — SVM/KNN)
            clf = pipeline.named_steps["clf"]
            X_train_df = None
            if "RandomForest" not in type(clf).__name__ and \
                    "LogisticRegression" not in type(clf).__name__:
                df_features = joblib.load(os.path.join(_HERE, "saved_models", "df_features.joblib"))
                test_subject = testdata.get("test_subject") or fold_tag.replace("fold_", "")
                df_train = df_features[df_features["subject_id"] != test_subject]
                X_train_raw = df_train[columns].values.astype(float)
                X_train_transformed = preprocessor.transform(X_train_raw)
                X_train_df = pd.DataFrame(X_train_transformed, columns=reference_columns)

            explainer, shap_vals = _get_explainer(clf, X_train_df, X_test_df)
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]

            all_shap_values.append(shap_vals)
            all_X_test.append(X_test_df)
            all_y_test.append(y_test_fold)

        if not columns_consistent:
            print(f"  [WARN] The selected columns vary across folds for "
                  f"{model_name} in '{MODE_SLUG}' mode — direct aggregation "
                  f"is not valid. Skipping this model.")
            continue

        if not all_shap_values:
            print(f"  [WARN] No valid fold found for {model_name}")
            continue

        # ── concatenate all folds into a single set ───────────────────────
        shap_values_agg = np.concatenate(all_shap_values, axis=0)
        X_test_agg       = pd.concat(all_X_test, axis=0, ignore_index=True)
        y_test_agg        = np.concatenate(all_y_test, axis=0)

        print(f"  [info] Aggregated: {shap_values_agg.shape[0]} segments "
              f"from {len(all_shap_values)} folds, {shap_values_agg.shape[1]} features")

        _plot_summary(shap_values_agg, X_test_agg, model_name, SAVE_DIR, TOP_N)
        _plot_scatter_top(shap_values_agg, X_test_agg, y_test_agg, model_name, SAVE_DIR)

    print(f"\n[done] Aggregated plots saved to {SAVE_DIR}/")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions (adapted from the original script)
# ------------------------------------------------------------------------------------------------------------------- #

def _get_explainer(clf, X_train_df, X_test_df):
    """
    Pick the appropriate SHAP explainer for the classifier type and
    compute the SHAP values for the given test set.

    :param clf: fitted classifier (last step of the pipeline)
    :param X_train_df: background data for KernelExplainer, or None when
                        not needed (TreeExplainer/LinearExplainer)
    :param X_test_df: test data to explain
    :return: (explainer, shap_values)
    """
    if "RandomForest" in type(clf).__name__:
        print(f"    [shap] TreeExplainer")
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_test_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    elif "LogisticRegression" in type(clf).__name__:
        print(f"    [shap] LinearExplainer")
        # LinearExplainer needs a background — uses X_test itself as a
        # lightweight approximation when X_train_df was not built
        explainer = shap.LinearExplainer(clf, X_test_df)
        shap_vals = explainer.shap_values(X_test_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    else:
        print(f"    [shap] KernelExplainer (can be slow)")
        background = X_train_df
        predict_fn = lambda x: clf.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        n_samples = KERNEL_EXPLAINER_NSAMPLES or (2 * X_test_df.shape[1] + 2048)
        shap_vals = explainer.shap_values(X_test_df, nsamples=n_samples)

    return explainer, shap_vals


def _plot_summary(shap_values, X_test_df, model_name, save_dir, top_n=20):
    """
    Plot and save the aggregated SHAP summary (beeswarm) plot for one model.

    :param shap_values: aggregated SHAP values array
    :param X_test_df: aggregated test features DataFrame
    :param model_name: model name (used in the title and filename)
    :param save_dir: directory where the figure is saved
    :param top_n: number of top features to display
    """
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    shap.summary_plot(
        shap_values, X_test_df,
        max_display=top_n, show=False, plot_type="dot",
    )
    plt.title(f"{model_name} — SHAP summary, aggregated across 25 LOSO folds "
              f"\nPositive SHAP → pathological  |  Negative SHAP → control",
              fontsize=10)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"shap_summary_loso_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] Summary plot saved → {fname}")


def _plot_scatter_top(shap_values, X_test_df, y_test, model_name, save_dir):
    """
    Plot and save a KDE pairplot of the top 4 SHAP features for one model,
    colored by class.

    :param shap_values: aggregated SHAP values array
    :param X_test_df: aggregated test features DataFrame
    :param y_test: aggregated true labels array
    :param model_name: model name (used in the title and filename)
    :param save_dir: directory where the figure is saved
    """
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
        print(f"    [WARN] KDE offdiag failed ({e}) — scatter only")
    g.map_offdiag(sns.scatterplot, alpha=0.4, s=25,
                  edgecolors="white", linewidths=0.3)
    g.add_legend(title="Class", fontsize=9)
    g.figure.suptitle(f"{model_name} — top 4 SHAP features (LOSO, aggregated)",
                      y=1.02, fontsize=11)

    fname = os.path.join(save_dir, f"shap_kde_loso_{model_name.replace(' ', '_')}.png")
    g.figure.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] KDE pairplot saved → {fname}")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()