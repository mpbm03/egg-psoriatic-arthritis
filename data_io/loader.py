"""
Load raw EGG recordings from disk into per-recording DataFrames.

Expected directory structure:
    data_dir/
        CONTROL GROUP/
            <subject_id>/
                baseline.txt
                long.txt
        Pathological/
            <subject_id>/
                baseline.txt
                long.txt

File format: tab/whitespace-separated .txt, columns defined in ALL_COLUMNS
(or ALL_COLUMNS_NO_CH5 for subjects recorded without CH8, e.g. ID7).


Available Functions
-------------------
[Public]
load_recording   - Load a single .txt recording file into a DataFrame
load_all_data    - Load every baseline/long recording for every subject in every group folder

------------------
[Private]
_read_txt_file        - Read one .txt recording, adapting columns to the file's actual column count
_get_group_dirs        - Find known group folders (Control/Pathological) inside data_dir
_get_subject_dirs      - List subject directories inside a group folder
_get_recording_paths   - List valid baseline/long .txt files inside a subject directory

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
from pathlib import Path

# internal imports
from constants import DATA_DIR, FS, TIME_COL, ALL_COLUMNS, ALL_COLUMNS_NO_CH5

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GROUP_LABELS = {
    "Control": 0,
    "Pathological": 1,
}

VALID_RECORDINGS = {"baseline", "long"}

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_recording(filepath: Path) -> pd.DataFrame:
    """
    Load a single .txt recording file.

    :param filepath: Path to the .txt file.
    :return: DataFrame with columns defined in ALL_COLUMNS plus a 'time' column.
    """
    return _read_txt_file(filepath)


def load_all_data(
        data_dir: Path = DATA_DIR,
) -> list[dict]:
    """
    Load all recordings for every subject in every group found in data_dir.

    Expected structure::

        data_dir/
            CONTROL GROUP/
                ID1/
                    baseline.txt
                    long.txt
            Pathological/
                ID2/
                    baseline.txt

    :param data_dir: Path  root directory containing the group folders.

    :return: list of dicts, one per recording::

        [
            {
                'subject_id':    'ID1',
                'group':         'CONTROL GROUP',
                'label':         0,
                'recording':     'baseline',   # 'baseline' or 'long'
                'df':            pd.DataFrame,
            },
            ...
        ]
    """
    group_dirs = _get_group_dirs(data_dir)
    if not group_dirs:
        raise FileNotFoundError(
            f"No known group folders found in: {data_dir}\n"
            f"Expected one of: {list(GROUP_LABELS)}"
        )

    records = []

    for group_dir, label in group_dirs:
        group_name = group_dir.name
        subject_dirs = _get_subject_dirs(group_dir)

        print(f"\nGroup: {group_name}  (label={label})  —  "
              f"{len(subject_dirs)} subject(s)")

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            fps = _get_recording_paths(subject_dir)

            if not fps:
                print(f"  [WARNING] No baseline/long .txt found for {subject_id}")
                continue

            for fp in fps:
                try:
                    df = _read_txt_file(fp, subject_id)
                    records.append({
                        "subject_id": subject_id,
                        "subject_dir": subject_dir,  # Path — used to locate manual artifact files
                        "group": group_name,
                        "label": label,
                        "recording": fp.stem,  # 'baseline' or 'long'
                        "df": df,
                    })
                    print(f"  [OK] {subject_id}/{fp.stem}  —  {len(df)} rows")
                except PermissionError:
                    print(f"  [SKIP] {subject_id}/{fp.stem}  —  PermissionError "
                          f"(file may be open or locked)")
                except Exception as e:
                    print(f"  [SKIP] {subject_id}/{fp.stem}  —  {e}")

    print(f"\nDone — {len(records)} recording(s) loaded in total.")
    return records

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _read_txt_file(filepath: Path, subject_id: str = None) -> pd.DataFrame:
    """
    Read a single tab-separated .txt recording into a DataFrame.

    The column list is adapted to the actual number of columns in the file.
    Some subjects (e.g. ID7) were recorded without CH8, producing 9 columns
    instead of the standard 10. In that case the last column of ALL_COLUMNS
    is simply omitted so the file loads correctly without losing any data.

    :param filepath:   path to the .txt recording file
    :param subject_id: subject ID, used to detect the ID7 special case (optional)
    :return: DataFrame with the resolved columns plus a 'time' column
    """
    df = pd.read_csv(filepath, sep=r"\s+", comment="#", header=None, engine="c")
    n_cols = df.shape[1]

    if n_cols == len(ALL_COLUMNS):
        df.columns = ALL_COLUMNS
    elif subject_id is not None and subject_id.upper() == "ID7":
        df.columns = ALL_COLUMNS_NO_CH5
        print(f"  [INFO] {filepath.name}: ID7 — using NO_CH5 column layout")
    else:
        raise ValueError(f"{filepath.name}: unexpected column count {n_cols}")

    df[TIME_COL] = df["nSeq"] / FS
    return df


def _get_group_dirs(data_dir: Path) -> list[tuple[Path, int]]:
    """
    Return (group_path, label) pairs for each known group folder found in data_dir.
    Warns if a folder is found that is not in GROUP_LABELS.

    :param data_dir: root directory containing the group folders
    :return: list of (group_path, label) tuples
    """
    found = []
    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if entry.name in GROUP_LABELS:
            found.append((Path(entry.path), GROUP_LABELS[entry.name]))
        else:
            print(f"  [WARNING] Unknown folder '{entry.name}' — skipping "
                  f"(expected one of: {list(GROUP_LABELS)})")
    return found


def _get_subject_dirs(group_dir: Path) -> list[Path]:
    """
        Return sorted list of subject directories inside a group folder.

        :param group_dir: path to a group folder
        :return: sorted list of subject directory paths
        """
    return sorted(
        Path(e.path) for e in os.scandir(group_dir) if e.is_dir()
    )

def _get_recording_paths(subject_dir: Path) -> list[Path]:
    """
    Return sorted list of valid .txt recording files (baseline / long)
    found inside a subject directory.

    :param subject_dir: path to a subject folder
    :return: sorted list of valid recording file paths
    """
    return sorted(
        fp for fp in subject_dir.glob("*.txt")
        if fp.stem in VALID_RECORDINGS
    )