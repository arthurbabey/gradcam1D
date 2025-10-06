"""Data loading, encoding, caching, and validation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

BACTERIA_REQUIRED_COLS = {"bacterium_id", "bacterium_sequence", "sequence_length"}
PHAGE_REQUIRED_COLS = {"phage_id", "phage_sequence", "sequence_length"}
COUPLES_REQUIRED_COLS = {"bacterium_id", "phage_id", "interaction_type"}

DEFAULT_THRESHOLDS = {"phage": 200000, "bacteria": 7000000}

NUCLEOTIDE_ONEHOT = {
    "A": np.array([1, 0, 0, 0], dtype=np.uint8),
    "C": np.array([0, 1, 0, 0], dtype=np.uint8),
    "G": np.array([0, 0, 1, 0], dtype=np.uint8),
    "T": np.array([0, 0, 0, 1], dtype=np.uint8),
    "R": np.array([1, 0, 1, 0], dtype=np.uint8),
    "Y": np.array([0, 1, 0, 1], dtype=np.uint8),
    "K": np.array([0, 0, 1, 1], dtype=np.uint8),
    "M": np.array([1, 1, 0, 0], dtype=np.uint8),
    "S": np.array([0, 1, 1, 0], dtype=np.uint8),
    "W": np.array([1, 0, 0, 1], dtype=np.uint8),
    "B": np.array([0, 1, 1, 1], dtype=np.uint8),
    "D": np.array([1, 0, 1, 1], dtype=np.uint8),
    "H": np.array([1, 1, 0, 1], dtype=np.uint8),
    "V": np.array([1, 1, 1, 0], dtype=np.uint8),
    "N": np.array([1, 1, 1, 1], dtype=np.uint8),
    "Z": np.array([0, 0, 0, 0], dtype=np.uint8),
}

ASCII_TO_ONEHOT = np.zeros((256, 4), dtype=np.uint8)
for base, vec in NUCLEOTIDE_ONEHOT.items():
    ASCII_TO_ONEHOT[ord(base)] = vec


class InteractionDataset(torch.utils.data.Dataset):
    """Dataset pulling cached tensors for bacterium/phage interactions."""

    def __init__(
        self,
        couples_df: pd.DataFrame,
        phage_id_to_path: Dict[str, Path],
        bacterium_id_to_path: Dict[str, Path],
    ) -> None:
        self.couples = couples_df.reset_index(drop=True)
        self.phage_path = phage_id_to_path
        self.bact_path = bacterium_id_to_path

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return len(self.couples)

    def __getitem__(self, idx: int):
        row = self.couples.iloc[idx]
        phage_id = str(row["phage_id"])
        bacterium_id = str(row["bacterium_id"])
        phage_arr = np.load(self.phage_path[phage_id])
        bact_arr = np.load(self.bact_path[bacterium_id])
        phage_tensor = torch.from_numpy(phage_arr.astype(np.float32))
        bacterium_tensor = torch.from_numpy(bact_arr.astype(np.float32))
        label = torch.tensor([row["interaction_type"]], dtype=torch.float32)
        return bacterium_tensor, phage_tensor, label


def validate_dataframe(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {name}: {sorted(missing)}")


def load_datasets(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bacteria_df = pd.read_csv(paths["bacteria"])
    phages_df = pd.read_csv(paths["phages"])
    couples_df = pd.read_csv(paths["couples"])

    validate_dataframe(bacteria_df, BACTERIA_REQUIRED_COLS, "bacteria_df")
    validate_dataframe(phages_df, PHAGE_REQUIRED_COLS, "phages_df")
    validate_dataframe(couples_df, COUPLES_REQUIRED_COLS, "couples_df")

    return bacteria_df, phages_df, couples_df


def encode_sequence_onehot(seq: str, target_length: int) -> np.ndarray:
    seq_bytes = seq.upper().encode("ascii", "ignore")
    seq_codes = np.frombuffer(seq_bytes, dtype=np.uint8)
    if seq_codes.size == 0:
        onehot_seq = np.zeros((0, 4), dtype=np.uint8)
    else:
        onehot_seq = ASCII_TO_ONEHOT[seq_codes]

    if onehot_seq.shape[0] >= target_length:
        padded = onehot_seq[:target_length, :]
    else:
        padded = np.zeros((target_length, 4), dtype=np.uint8)
        padded[: onehot_seq.shape[0], :] = onehot_seq
    return padded.T


def tensor_to_sequence(onehot_tensor: np.ndarray) -> str:
    arr = onehot_tensor
    if arr.ndim == 2 and arr.shape[0] == 4:
        arr = np.expand_dims(arr, axis=0)

    n, c, _ = arr.shape
    if n != 1 or c != 4:
        raise ValueError("Expected shape (1, 4, L) or (4, L)")

    decoded = []
    for row in arr[0].T:
        decoded.append(_decode_row(tuple(row.tolist())))
    return "".join(decoded)


def _decode_row(row: Tuple[int, ...]) -> str:
    for base, vec in NUCLEOTIDE_ONEHOT.items():
        if tuple(vec.tolist()) == row:
            return base
    return "N"


def ensure_cache_dirs(cache_dir: Path) -> Tuple[Path, Path]:
    phage_cache = cache_dir / "phage"
    bacteria_cache = cache_dir / "bacteria"
    phage_cache.mkdir(parents=True, exist_ok=True)
    bacteria_cache.mkdir(parents=True, exist_ok=True)
    return phage_cache, bacteria_cache


def precompute_and_cache_sequences(
    phages_df: pd.DataFrame,
    bacteria_df: pd.DataFrame,
    cache_dir: Path,
    thresholds: Dict[str, int],
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    phage_cache, bacteria_cache = ensure_cache_dirs(cache_dir)
    phage_map: Dict[str, Path] = {}
    bacteria_map: Dict[str, Path] = {}

    phage_threshold = thresholds.get("phage", DEFAULT_THRESHOLDS["phage"])
    bacteria_threshold = thresholds.get("bacteria", DEFAULT_THRESHOLDS["bacteria"])

    for _, row in phages_df.iterrows():
        pid = str(row["phage_id"])
        out_path = phage_cache / f"{pid}.npy"
        phage_map[pid] = out_path
        if not out_path.exists():
            arr = encode_sequence_onehot(row["phage_sequence"], phage_threshold)
            np.save(out_path, arr)

    for _, row in bacteria_df.iterrows():
        bid = str(row["bacterium_id"])
        out_path = bacteria_cache / f"{bid}.npy"
        bacteria_map[bid] = out_path
        if not out_path.exists():
            arr = encode_sequence_onehot(row["bacterium_sequence"], bacteria_threshold)
            np.save(out_path, arr)

    return phage_map, bacteria_map


def check_cache_integrity(
    phage_map: Dict[str, Path],
    bacteria_map: Dict[str, Path],
    thresholds: Dict[str, int],
    sample_size: int = 5,
) -> None:
    issues: List[str] = []

    for branch, mapping in (("phage", phage_map), ("bacteria", bacteria_map)):
        expected_shape = (4, thresholds.get(branch, DEFAULT_THRESHOLDS[branch]))
        for idx, (seq_id, path) in enumerate(mapping.items()):
            if not path.exists():
                issues.append(f"Missing cache file for {branch} {seq_id}: {path}")
                continue
            if idx >= sample_size:
                continue
            arr = np.load(path, mmap_mode="r")
            if arr.shape != expected_shape:
                issues.append(
                    f"Cache shape mismatch for {branch} {seq_id}: {arr.shape} != {expected_shape}"
                )

    if issues:
        raise ValueError("\n".join(issues))


def dump_manifest(
    manifest_path: Path,
    phage_map: Dict[str, Path],
    bacteria_map: Dict[str, Path],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phage": {k: str(v) for k, v in phage_map.items()},
        "bacteria": {k: str(v) for k, v in bacteria_map.items()},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    phage_map = {k: Path(v) for k, v in payload.get("phage", {}).items()}
    bacteria_map = {k: Path(v) for k, v in payload.get("bacteria", {}).items()}
    return phage_map, bacteria_map


def write_fasta(sequence: str, regions: List[Tuple[int, int, float]], fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as fh:
        for idx, (start, end, score) in enumerate(regions, start=1):
            fh.write(f">region_{idx}_{start}_{end}_score_{score:.4f}\n")
            fh.write(sequence[start:end] + "\n")
