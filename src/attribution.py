"""Attribution helpers wrapping Captum to produce base-wise scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import tensor_to_sequence


def prepare_sample(dataset, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    bacterium_tensor, phage_tensor, label = dataset[idx]
    return bacterium_tensor, phage_tensor, float(label.item())


def get_sequences(
    bacterium_tensor: torch.Tensor,
    phage_tensor: torch.Tensor,
) -> Dict[str, str]:
    bact_seq = tensor_to_sequence(bacterium_tensor.cpu().numpy())
    phage_seq = tensor_to_sequence(phage_tensor.cpu().numpy())
    return {"bacteria": bact_seq, "phage": phage_seq}


def combine_attributions(attr_tensor: torch.Tensor, input_length: int | None = None) -> np.ndarray:
    if attr_tensor.dim() != 3:
        raise ValueError("Attribution tensor must be 3D (batch, channels, length)")

    batch, channels, length = attr_tensor.shape

    if channels == 4:
        combined = attr_tensor.squeeze(0).sum(dim=0)
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
        return combined.cpu().numpy()

    if channels == 1:
        if input_length is None:
            raise ValueError("input_length required for single-channel attribution")
        upsampled = F.interpolate(attr_tensor, size=(input_length,), mode="nearest")
        combined = upsampled.squeeze(0).squeeze(0)
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
        return combined.cpu().numpy()

    raise ValueError("Unsupported channel dimension; expected 1 or 4")


def compute_attribution(
    model,
    branch: str,
    method: str,
    target_layer,
    bacterium_tensor: torch.Tensor,
    phage_tensor: torch.Tensor,
    thresholds: Dict[str, int],
    device: torch.device,
) -> np.ndarray:
    method = method.lower()
    bact_input = bacterium_tensor.to(device).unsqueeze(0).requires_grad_()
    phage_input = phage_tensor.to(device).unsqueeze(0).requires_grad_()
    inputs = (bact_input, phage_input)

    if method == "guided":
        from captum.attr import GuidedGradCam

        guided = GuidedGradCam(model, target_layer)
        attributions = guided.attribute(inputs)
        attr_tensor = attributions[0] if branch == "bacteria" else attributions[1]
        return combine_attributions(attr_tensor.detach())

    if method == "layercam":
        from captum.attr import LayerGradCam

        layercam = LayerGradCam(model, target_layer)
        attr = layercam.attribute(inputs)
        length = thresholds.get(branch)
        return combine_attributions(attr.detach(), input_length=length)

    raise ValueError(f"Unsupported attribution method: {method}")


def trim_to_sequence(sequence: str, attribution: np.ndarray) -> np.ndarray:
    return attribution[: len(sequence)]


def save_attribution(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def topk_summary(attribution: np.ndarray, k: int = 5) -> str:
    indices = np.argsort(attribution)[-k:][::-1]
    return ", ".join(f"{idx}:{attribution[idx]:.3f}" for idx in indices)
