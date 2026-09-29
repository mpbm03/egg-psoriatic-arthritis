"""
Load manually annotated artifact intervals from per-subject annotation files.

Expected file location and name:
    <subject_dir>/<subject_id>_noisy_intervals.txt

File format (Python literal assignments, one variable per channel):
    noisy_intervals1 = [(start, end), ...]
    noisy_intervals2 = [(start, end), ...]
    ...
    questionable_intervals1 = [...]


Available Functions
-------------------
[Public]
manual_artifacts_available                 - Check whether a manual annotation file exists for a subject/recording
load_manual_artifact_indices_per_channel    - Load and merge noisy intervals from the annotation file, grouped by channel
load_manual_questionable_indices_per_channel - Load and merge questionable intervals from the annotation file, grouped by channel

------------------
[Private]
_annotation_path        - Build the expected file path for a subject's annotation file
_parse_annotation_file   - Parse a Python-literal annotation file into a dict of variable assignments

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import ast
import re
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def manual_artifacts_available(subject_dir: Path, subject_id: str, recording_name: str) -> bool:
    """
    Return True if a manual annotation file exists for this subject and recording.

    :param subject_dir:    Path  directory that contains the subject recordings
    :param subject_id:     str   e.g. 'ID3'
    :param recording_name: str   'baseline' or 'long'
    :return: bool
    """
    return _annotation_path(subject_dir, subject_id, recording_name).exists()



def load_manual_artifact_indices_per_channel(
        subject_dir: Path,
        subject_id: str,
        recording_name: str,
        channel_map: dict | None = None,
) -> dict:
    """
    Load manual artifact intervals from the annotation file, grouped by EGG channel.

    Each noisy_intervals<N> variable in the file maps to a specific EGG channel
    via channel_map.  Intervals for each channel are sorted and merged
    independently — a noisy interval on CH2 does NOT affect CH1.

    :param subject_dir:    Path  directory containing the subject recordings
    :param subject_id:     str   e.g. 'ID3'
    :param recording_name: str   'baseline' or 'long'
    :param channel_map:    dict  maps suffix number (str) to channel name.
                                 Defaults to {"1": "CH1", "2": "CH2",
                                              "3": "CH3", "4": "CH4"}.

    :return: dict  {channel_name: List[Tuple[int, int]]}
             e.g. {"CH1": [(0, 500), ...], "CH2": [(200, 800), ...], ...}
             Channels with no annotated intervals are included with an empty list.
    :raises FileNotFoundError: if the annotation file does not exist
    """
    if channel_map is None:
        channel_map = {"1": "CH1", "2": "CH2", "3": "CH3", "4": "CH4"}

    path = _annotation_path(subject_dir, subject_id, recording_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Manual artifact file not found: {path}\nExpected: {path.name} inside {subject_dir}"
        )

    variables = _parse_annotation_file(path)

    # Initialise empty lists for every mapped channel
    per_channel: dict = {ch: [] for ch in channel_map.values()}

    for var_name, value in variables.items():
        if not var_name.startswith("noisy_intervals"):
            continue

        suffix = var_name[len("noisy_intervals"):]  # e.g. "1", "2", ...
        if suffix not in channel_map:
            continue  # unmapped suffix — skip

        channel = channel_map[suffix]
        if not isinstance(value, list):
            raise ValueError(
                f"Expected a list for '{var_name}' in {path}, got {type(value)}"
            )
        per_channel[channel].extend(value)

    # Sort and merge each channel independently
    for ch in per_channel:
        intervals = sorted(per_channel[ch], key=lambda x: x[0])
        if not intervals:
            continue
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        per_channel[ch] = merged

    return per_channel


def load_manual_questionable_indices_per_channel(
        subject_dir: Path,
        subject_id: str,
        recording_name: str,
        channel_map: dict | None = None,
) -> dict:
    """
    Load questionable artifact intervals from the annotation file, grouped
    by EGG channel.

    Identical to load_manual_artifact_indices_per_channel but reads
    questionable_intervals<N> variables instead of noisy_intervals<N>.
    Returns an empty list per channel if no questionable intervals exist.

    :param subject_dir:    Path  directory containing the subject recordings
    :param subject_id:     str   e.g. 'ID3'
    :param recording_name: str   'baseline' or 'long'
    :param channel_map:    dict  maps suffix number to channel name.
                                 Defaults to {"1": "CH1", "2": "CH2",
                                              "3": "CH3", "4": "CH4"}.

    :return: dict  {channel_name: List[Tuple[int, int]]}
    """
    if channel_map is None:
        channel_map = {"1": "CH1", "2": "CH2", "3": "CH3", "4": "CH4"}

    path = _annotation_path(subject_dir, subject_id, recording_name)
    if not path.exists():
        return {ch: [] for ch in channel_map.values()}

    variables = _parse_annotation_file(path)

    per_channel: dict = {ch: [] for ch in channel_map.values()}

    for var_name, value in variables.items():
        if not var_name.startswith("questionable_intervals"):
            continue

        suffix = var_name[len("questionable_intervals"):]
        if suffix not in channel_map:
            continue

        channel = channel_map[suffix]
        if not isinstance(value, list):
            continue
        per_channel[channel].extend(value)

    # sort and merge each channel independently
    for ch in per_channel:
        intervals = sorted(per_channel[ch], key=lambda x: x[0])
        if not intervals:
            continue
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        per_channel[ch] = merged

    return per_channel


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _annotation_path(subject_dir: Path, subject_id: str, recording_name: str) -> Path:
    """
    Return the expected path of the annotation file for a given recording.

    Naming convention:
        baseline -> <subject_id>_baseline_noisy_intervals.txt
        long     -> <subject_id>_noisy_intervals.txt
        other    -> <subject_id>_<recording_name>_noisy_intervals.txt  (fallback)

    :param subject_dir:    directory containing the subject recordings
    :param subject_id:     e.g. 'ID3'
    :param recording_name: 'baseline' or 'long'
    :return: expected Path of the annotation file
    """
    if recording_name == "long":
        filename = f"{subject_id}_noisy_intervals.txt"
    else:
        filename = f"{subject_id}_{recording_name}_noisy_intervals.txt"
    return subject_dir / filename


def _parse_annotation_file(path: Path) -> dict:
    """
    Parse a Python-literal annotation file into a dict of {variable_name: value}.

    Each line of the form:
        variable_name = <python_literal>
    is evaluated safely with ast.literal_eval.  Lines that are comments (#),
    blank, or that cannot be parsed are silently skipped.

    :param path: Path to the annotation .txt file
    :return: dict  {variable_name: parsed_value}
    :raises ValueError: if a parseable line contains a non-literal expression
    """
    variables = {}
    pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$")

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = pattern.match(line)
            if not match:
                continue  # continuation line or unrecognised syntax — skip

            var_name = match.group(1)
            expression = match.group(2)

            try:
                variables[var_name] = ast.literal_eval(expression)
            except (ValueError, SyntaxError):
                # Multi-line literals (list spread over several lines) are not
                # handled here — the file format uses single-line lists so this
                # should not occur in practice.
                print(f"  [artifact_loader] Could not parse line {lineno} "
                      f"in {path.name}: {line[:60]}...")

    return variables