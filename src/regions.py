"""Region selection utilities for attribution signals."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


Region = Tuple[int, int, float]


def find_top_k_important_windows(
    sequence: str,
    attribution: np.ndarray,
    k: int = 3,
    window_size: int = 500,
) -> List[Region]:
    if len(attribution) != len(sequence):
        raise ValueError("Sequence and attribution must have same length")
    if window_size > len(sequence):
        raise ValueError("window_size cannot exceed sequence length")

    conv = np.convolve(attribution, np.ones(window_size, dtype=np.float32), mode="valid")
    means = conv / window_size
    sorted_indices = np.argsort(means)[::-1]

    selected: List[Region] = []
    for idx in sorted_indices:
        start = int(idx)
        end = int(idx + window_size)
        if any(not (end <= s or start >= e) for s, e, _ in selected):
            continue
        selected.append((start, end, float(means[idx])))
        if len(selected) == k:
            break

    selected.sort(key=lambda x: x[0])
    return selected


def regions_to_frame(regions: Sequence[Region], branch: str, sample_name: str) -> pd.DataFrame:
    data = [
        {
            "sample": sample_name,
            "branch": branch,
            "rank": idx + 1,
            "start": start,
            "end": end,
            "mean_importance": score,
        }
        for idx, (start, end, score) in enumerate(regions)
    ]
    return pd.DataFrame(data)


def save_regions_table(regions: Sequence[Region], path: Path, branch: str, sample_name: str) -> None:
    frame = regions_to_frame(regions, branch, sample_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
