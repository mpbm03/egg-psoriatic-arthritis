"""
Plotting utilities for model evaluation: learning curves, confusion
matrices, and feature selection sweeps.

Available Functions
-------------------
[Public]
get_display_labels               - Return the class display labels ("control"/"pathologic" or "preprandial"/"postprandial") for a given LABEL_MODE
plot_learning_curves             - Plot learning curves (train vs validation F1) for all fitted models, using GroupKFold
plot_confusion_matrices          - Plot confusion matrices for all fitted models, sharing axis labels on the first panel only
plot_confusion_matrices_pipeline - Plot confusion matrices for models whose pipeline applies scaler/corr/selector internally, from the original pre-pipeline features
plot_feature_selection           - Plot test F1 vs number of features (k) per model, marking the best k chosen by CV F1
plot_learning_curves_bestk       - Plot learning curves for each model using its own best-k feature set
plot_confusion_matrices_bestk    - Plot confusion matrices for each model using its own best-k feature set
plot_learning_curves_pipeline    - Plot learning curves for models whose pipeline applies all transformations internally, from the original pre-pipeline features


------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline        import Pipeline
from sklearn.metrics         import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from typing import Dict
from sklearn.model_selection import GroupKFold


# internal imports
from constants import LABEL_MODE

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_display_labels(label_mode: str):
    """
        Return the class display labels for a given LABEL_MODE.

        :param label_mode: labeling mode, "group"
        :return: list of two display label strings, [negative_class, positive_class]
    """
    if label_mode == "group":
        return ["control", "pathologic"]
    else:
        raise ValueError(f"Unknown LABEL_MODE: {label_mode}")


def plot_learning_curves(
    models:            Dict[str, Pipeline],
    X_train:           np.ndarray,
    y_train:           np.ndarray,
    subject_ids_train: np.ndarray,
    cv:                int = 5,
    save_path:         str = None,
) -> None:
    """
    Plot learning curves for all models to assess overfitting.

    Uses GroupKFold to ensure segments from the same subject never appear
    in both train and validation folds — consistent with the main pipeline.

    :param models:            dict of fitted sklearn Pipelines
    :param X_train:           np.ndarray
    :param y_train:           np.ndarray
    :param subject_ids_train: np.ndarray  subject IDs for GroupKFold
    :param cv:                int  number of folds. Default 5.
    :param save_path:         if provided, saves the figure to this path.
    """

    group_kfold = GroupKFold(n_splits=cv)
    cv_splits   = list(group_kfold.split(X_train, y_train, subject_ids_train))

    n_models  = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv           = cv_splits,
            scoring      = "f1_macro",
            n_jobs       = -1,
            train_sizes  = np.linspace(0.1, 1.0, 10),
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        ax.plot(train_sizes, train_mean, "o-", color="steelblue",  label="Training")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="steelblue")
        ax.plot(train_sizes, val_mean,   "o-", color="darkorange", label="Validation")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="darkorange")

        ax.set_title(name, fontsize=20)
        ax.set_xlabel("Training set size", fontsize=20)
        ax.set_ylabel("F1 macro", fontsize=20)
        ax.legend(loc="lower right", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Learning Curves", fontsize=24, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()


def plot_confusion_matrices(
    models: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Plot confusion matrices for all fitted models.

    Only the first (leftmost) panel keeps the axis labels and tick labels
    (control/pathological on both axes), since all panels share the same
    classes. This frees up horizontal space for the remaining panels,
    which can then be drawn larger.

    :param models: dict of fitted sklearn Pipelines
    :param X_test: np.ndarray
    :param y_test: np.ndarray
    :param save_path: if provided, saves the figure to this path.
    """
    n_models  = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels = get_display_labels(LABEL_MODE),
            ax             = ax,
            colorbar       = False,
        )
        ax.set_title(name, fontsize=20)

        if i == 0:
            # First panel: keeps everything
            ax.set_xlabel(ax.get_xlabel(), fontsize=20)
            ax.set_ylabel(ax.get_ylabel(), fontsize=20)
            ax.tick_params(axis="both", labelsize=16)
        else:
            # Remaining panels: remove redundant axis labels
            ax.set_xlabel(ax.get_xlabel(), fontsize=20)
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelleft=False)

        for text in ax.texts:
            text.set_fontsize(20)

    plt.suptitle("Confusion Matrices", fontsize=22, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()


def plot_confusion_matrices_pipeline(
        fitted_models: Dict[str, Pipeline],
        X_test_per_model: Dict[str, np.ndarray],
        y_test: np.ndarray,
        best_k_per_model: Dict[str, int] = None,
        save_path: str = None,
) -> None:
    """
    Plot confusion matrices for each model where the pipeline handles
    all transformations internally (scaler → corr → selector → clf).

    Use this when X_test_per_model contains the original (pre-pipeline)
    features — the pipeline's predict() applies all steps internally.
    This is the correct function to use when CorrelationFilter and
    SelectKBest are inside the pipeline.

    Only the first (leftmost) panel keeps the y-axis labels and tick
    labels (control/pathological), since all panels share the same
    classes. This frees up horizontal space for the remaining panels,
    which can then be drawn larger.

    :param fitted_models:    dict  model_name → fitted Pipeline
    :param X_test_per_model: dict  model_name → X_test with original features
                             (before any pipeline transformation)
    :param y_test:           np.ndarray  test labels
    :param best_k_per_model: dict  model_name → best k value (for title only)
    :param save_path:        str   if provided, saves the figure to this path
    """
    n_models = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, (ax, (name, model)) in enumerate(zip(axes, fitted_models.items())):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test_per_model[name], y_test,
            display_labels=get_display_labels(LABEL_MODE),
            ax=ax,
            colorbar=False,
        )
        title = name
        if best_k_per_model and name in best_k_per_model:
            title += f"\n(top {best_k_per_model[name]} features)"
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)

        if i == 0:
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelleft=False)

        for text in ax.texts:
            text.set_fontsize(18)

    plt.suptitle("Confusion Matrices — best k per model", fontsize=20, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()


def plot_feature_selection(
    df_results: pd.DataFrame,     # test F1
    df_cv_results: pd.DataFrame,  # CV F1
    save_path: str = None,
) -> None:
    """
    Plot Test F1 vs number of features for all models, marking the best k based on CV F1.

    :param df_results: pd.DataFrame  Test F1 scores (n_features x models)
    :param df_cv_results: pd.DataFrame CV F1 scores (n_features x models)
    :param save_path: str             If provided, saves the figure to this path
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue", "darkorange", "green", "red"]
    markers = ["o", "s", "^", "D"]

    for (name, color, marker) in zip(df_results.columns, colors, markers):
        # Plot test F1
        ax.plot(
            df_results.index,
            df_results[name],
            label=f"{name} (Test F1)",
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )

        # Best k based on CV F1
        best_k = df_cv_results[name].idxmax()
        best_f1 = df_results[name][best_k]  # show test F1 at best CV k

        # Dashed vertical line for best k
        ax.axvline(best_k, color=color, linestyle="--", alpha=0.3)

        # Annotate best k
        ax.text(best_k, best_f1 + 0.02, f"k={best_k}", color=color, fontsize=10, ha='center')

    ax.set_xlabel("Number of features (k)", fontsize=12)
    ax.set_ylabel("Test F1 (macro)", fontsize=12)
    ax.set_title("SelectKBest Feature Selection Sweep", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xticks(df_results.index)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature selection plot saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()

def plot_learning_curves_bestk(
    fitted_models:     Dict[str, Pipeline],
    X_train_per_model: Dict[str, np.ndarray],
    y_train:           np.ndarray,
    subject_ids_train: np.ndarray,
    best_k_per_model:  Dict[str, int],
    save_path:         str = None,
) -> None:
    """
    Plot learning curves for each model using its own best-k feature set.

    Uses GroupKFold to ensure segments from the same subject never appear
    in both train and validation folds — consistent with the main pipeline.

    :param fitted_models:     dict  model_name → fitted Pipeline
    :param X_train_per_model: dict  model_name → X_train with best-k features
    :param y_train:           np.ndarray  training labels
    :param subject_ids_train: np.ndarray  subject IDs for GroupKFold
    :param best_k_per_model:  dict  model_name → best k value
    :param save_path:         str   if provided, saves the figure to this path
    """
    from sklearn.model_selection import GroupKFold

    n_models  = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, fitted_models.items()):
        X_tr = X_train_per_model[name]
        k    = best_k_per_model[name]

        group_kfold = GroupKFold(n_splits=5)
        cv_splits   = list(group_kfold.split(X_tr, y_train, subject_ids_train))

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_tr, y_train,
            cv          = cv_splits,
            scoring     = "f1_macro",
            n_jobs      = -1,
            train_sizes = np.linspace(0.1, 1.0, 10),
        )
        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        v_mean  = val_scores.mean(axis=1)
        v_std   = val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, "o-", color="steelblue",  label="Training")
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")
        ax.plot(train_sizes, v_mean,  "o-", color="darkorange", label="Validation")
        ax.fill_between(train_sizes, v_mean - v_std,  v_mean + v_std,  alpha=0.15, color="darkorange")
        ax.set_title(f"{name}\n(top {k} features)")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("F1 macro")
        ax.legend(loc="lower right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Learning Curves — best k per model", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()

def plot_confusion_matrices_bestk(
    fitted_models:    Dict[str, Pipeline],
    X_test_per_model: Dict[str, np.ndarray],
    y_test:           np.ndarray,
    best_k_per_model: Dict[str, int],
    save_path:        str = None,
) -> None:
    """
    Plot confusion matrices for each model using its own best-k feature set.

    :param fitted_models:    dict  model_name → fitted Pipeline
    :param X_test_per_model: dict  model_name → X_test with best-k features
    :param y_test:           np.ndarray  test labels
    :param best_k_per_model: dict  model_name → best k value
    :param save_path:        str   if provided, saves the figure to this path
    """
    n_models  = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, fitted_models.items()):
        k = best_k_per_model[name]
        ConfusionMatrixDisplay.from_estimator(
            model, X_test_per_model[name], y_test,
            display_labels = get_display_labels(LABEL_MODE),
            ax             = ax,
            colorbar       = False,
        )
        ax.set_title(f"{name}\n(top {k} features)")

    plt.suptitle("Confusion Matrices — best k per model", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()

def plot_learning_curves_pipeline(
    fitted_models:     Dict[str, Pipeline],
    X_train:           np.ndarray,
    y_train:           np.ndarray,
    subject_ids_train: np.ndarray,
    best_k_per_model:  Dict[str, int] = None,
    save_path:         str = None,
) -> None:
    """
    Plot learning curves for each model where the pipeline handles all
    transformations internally (scaler → corr → selector → clf).

    Use this when the CorrelationFilter and SelectKBest are inside the
    pipeline — X_train contains the original features (after variance
    filter only) and the pipeline applies all other steps internally.

    :param fitted_models:     dict  model_name → fitted Pipeline
    :param X_train:           np.ndarray  original features (pre-pipeline)
    :param y_train:           np.ndarray  training labels
    :param subject_ids_train: np.ndarray  subject IDs for GroupKFold
    :param best_k_per_model:  dict  model_name → best k value (for title only)
    :param save_path:         str   if provided, saves the figure to this path
    """
    n_models  = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    group_kfold = GroupKFold(n_splits=5)
    cv_splits   = list(group_kfold.split(X_train, y_train, subject_ids_train))

    for ax, (name, model) in zip(axes, fitted_models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv          = cv_splits,
            scoring     = "f1_macro",
            n_jobs      = -1,
            train_sizes = np.linspace(0.1, 1.0, 10),
        )
        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        v_mean  = val_scores.mean(axis=1)
        v_std   = val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, "o-", color="steelblue",  label="Training")
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color="steelblue")
        ax.plot(train_sizes, v_mean,  "o-", color="darkorange", label="Validation")
        ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std,
                        alpha=0.15, color="darkorange")

        title = name
        if best_k_per_model and name in best_k_per_model:
            title += f"\n(top {best_k_per_model[name]} features)"
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Training set size", fontsize=20)
        ax.set_ylabel("F1 macro", fontsize=20)
        ax.legend(loc="lower right", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Learning Curves — best k per model (pipeline)", fontsize=24, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved → {save_path}")

    if plt.get_backend().lower() != "agg":
        plt.show()
