"""
Utility functions for the ML pipeline.

Available Functions
-------------------
[Public]
prepare_Xy   - Split a feature DataFrame into X (features) and y (labels), excluding label/metadata columns

------------------
[Private]


------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from typing import Tuple, List

# internal imports



# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def prepare_Xy(
    df: pd.DataFrame,
    drop_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a feature DataFrame into X (features) and y (labels).

    Automatically excludes 'label' and 'subject_id' from features.

    :param df: feature DataFrame with 'label' column
    :param drop_cols: extra columns to drop besides 'label' and 'subject_id'

    :return: (X, y) as np.ndarray
    """
    # _is_clean is a deployment metadata column — never a feature
    exclude = {"label", "subject_id", "_is_clean", "start_index", "duration_sec", "n_segments_total"}
    if drop_cols:
        exclude.update(drop_cols)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y