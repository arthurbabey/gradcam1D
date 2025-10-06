"""Model architecture and prediction helpers."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

warnings.filterwarnings(
    "ignore",
    message="h5py is running against HDF5",
    category=UserWarning,
)

import h5py
import torch
import torch.nn as nn
import json

from torch.utils.data import DataLoader


class BacteriaBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=5)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=10, stride=5, bias=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):  # pragma: no cover - tested indirectly
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        return x


class PhageBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):  # pragma: no cover - tested indirectly
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        return x


class PerphectInteractionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bacteria_branch = BacteriaBranch()
        self.phage_branch = PhageBranch()
        self.fc1 = nn.Linear(15296, 100, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 1, bias=True)

    def forward(self, bacteria_input, phage_input):  # pragma: no cover
        bact_features = self.bacteria_branch(bacteria_input)
        phage_features = self.phage_branch(phage_input)
        combined = torch.cat([bact_features, phage_features], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


def resolve_device(preferred: str | torch.device = "auto") -> torch.device:
    if isinstance(preferred, torch.device):
        return preferred
    if preferred is None or str(preferred).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    key = str(preferred).lower()
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if key == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def build_model(checkpoint: Path, device: str | torch.device = "cpu") -> nn.Module:
    resolved_device = resolve_device(device)
    model = PerphectInteractionModel()
    state_dict = torch.load(checkpoint, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model


def predict_dataset(
    model: nn.Module,
    dataset,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> List[Dict[str, float]]:
    resolved_device = resolve_device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results: List[Dict[str, float]] = []
    for batch_idx, (bact_x, phage_x, label) in enumerate(loader):
        bact_x = bact_x.to(resolved_device)
        phage_x = phage_x.to(resolved_device)
        with torch.no_grad():
            preds = model(bact_x, phage_x).cpu().numpy().flatten()
        for i, pred in enumerate(preds):
            results.append(
                {
                    "dataset_index": batch_idx * batch_size + i,
                    "prediction": float(pred),
                    "label": float(label[i].item()),
                }
            )
    return results


def run_parity_check(
    model: nn.Module,
    dataset,
    sample_indices: Sequence[int],
    reference_path: Path,
    tolerance: float = 1e-5,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    resolved_device = resolve_device(device)
    reference: Dict[str, float] = {}
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))

    current: Dict[str, float] = {}
    for idx in sample_indices:
        bact_x, phage_x, _ = dataset[idx]
        bact_x = bact_x.to(resolved_device)
        phage_x = phage_x.to(resolved_device)
        with torch.no_grad():
            pred = model(bact_x.unsqueeze(0), phage_x.unsqueeze(0)).item()
        key = str(idx)
        current[key] = float(pred)
        if reference:
            expected = reference.get(key)
            if expected is None:
                raise ValueError(f"Reference missing sample {key}")
            if abs(expected - current[key]) > tolerance:
                raise AssertionError(
                    f"Parity mismatch for sample {key}: expected {expected}, got {current[key]}"
                )

    if not reference:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        logging.info("Stored new parity reference at %s", reference_path)
    else:
        logging.info("Parity check passed for samples: %s", ", ".join(map(str, sample_indices)))

    return current


def load_keras_weights(pytorch_model: nn.Module, keras_h5_path: Path) -> nn.Module:
    with h5py.File(keras_h5_path, "r") as handle:
        weights = handle["model_weights"]

        def copy_conv(layer_name: str, pytorch_w: str, pytorch_b: str) -> None:
            kernel = weights[layer_name][layer_name]["kernel:0"][()]
            bias = weights[layer_name][layer_name]["bias:0"][()]
            pytorch_model.state_dict()[pytorch_w].copy_(torch.tensor(kernel).permute(2, 1, 0))
            pytorch_model.state_dict()[pytorch_b].copy_(torch.tensor(bias))

        copy_conv("bacterial_conv_1", "bacteria_branch.conv1.weight", "bacteria_branch.conv1.bias")
        copy_conv("bacterial_conv_2", "bacteria_branch.conv2.weight", "bacteria_branch.conv2.bias")
        copy_conv("bacterial_conv_3", "bacteria_branch.conv3.weight", "bacteria_branch.conv3.bias")
        copy_conv("phage_conv_1", "phage_branch.conv1.weight", "phage_branch.conv1.bias")
        copy_conv("phage_conv_2", "phage_branch.conv2.weight", "phage_branch.conv2.bias")

        dense_kernel = weights["dense"]["dense"]["kernel:0"][()]
        dense_bias = weights["dense"]["dense"]["bias:0"][()]
        dense1_kernel = weights["dense_1"]["dense_1"]["kernel:0"][()]
        dense1_bias = weights["dense_1"]["dense_1"]["bias:0"][()]

        pytorch_model.state_dict()["fc1.weight"].copy_(torch.tensor(dense_kernel).T)
        pytorch_model.state_dict()["fc1.bias"].copy_(torch.tensor(dense_bias))
        pytorch_model.state_dict()["fc2.weight"].copy_(torch.tensor(dense1_kernel).T)
        pytorch_model.state_dict()["fc2.bias"].copy_(torch.tensor(dense1_bias))

    return pytorch_model
