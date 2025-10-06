"""Backward-compatible imports for legacy notebooks/scripts."""

from src.attribution import combine_attributions  # noqa: F401
from src.data import (  # noqa: F401
    DEFAULT_THRESHOLDS,
    InteractionDataset,
    encode_sequence_onehot,
    precompute_and_cache_sequences,
    tensor_to_sequence,
    write_fasta,
)
from src.modeling import load_keras_weights  # noqa: F401

BACTERIUM_THRESHOLD = DEFAULT_THRESHOLDS["bacteria"]
PHAGE_THRESHOLD = DEFAULT_THRESHOLDS["phage"]

__all__ = [
    "InteractionDataset",
    "encode_sequence_onehot",
    "precompute_and_cache_sequences",
    "tensor_to_sequence",
    "write_fasta",
    "combine_attributions",
    "load_keras_weights",
    "BACTERIUM_THRESHOLD",
    "PHAGE_THRESHOLD",
]
