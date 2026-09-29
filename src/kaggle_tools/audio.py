"""Window helpers for soundscape competitions (BirdCLEF-style ``row_id`` format).

``row_id`` = ``<file_stem>_<end_second>``: one row per fixed-length window. Return a
separate prediction per window — averaging over the file and repeating it for every
``row_id`` raises no error but costs ~0.1 AUC.
"""

from __future__ import annotations

import numpy as np


def parse_row_id(row_id: str) -> tuple[str, int]:
    """``"BC2026_Test_0001_S05_20250227_010002_5"`` -> ``("BC2026_..._010002", 5)``."""
    stem, end = row_id.rsplit("_", 1)
    return stem, int(end)


def window_index(end_second: int, window_seconds: int = 5) -> int:
    """0-based window index for a window ending at ``end_second``."""
    return end_second // window_seconds - 1


def window_row_ids(stem: str, n_windows: int, window_seconds: int = 5) -> list[str]:
    return [f"{stem}_{(i + 1) * window_seconds}" for i in range(n_windows)]


def frame_windows(
    waveform: np.ndarray, sample_rate: int, window_seconds: float = 5.0, pad: bool = True
) -> np.ndarray:
    """Split a 1-D waveform into ``(n_windows, samples)`` non-overlapping windows.

    Batch all windows of a file through the model at once (on GPU) rather than
    looping per window on CPU — the latter is a classic inference timeout.
    """
    size = round(window_seconds * sample_rate)
    n = len(waveform) // size
    if pad and len(waveform) % size:
        n += 1
        waveform = np.pad(waveform, (0, n * size - len(waveform)))
    return np.asarray(waveform[: n * size]).reshape(n, size)
