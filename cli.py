"""Unified CLI for pipeline operations."""

from __future__ import annotations

import argparse
import logging
import re
from typing import Iterable, List

from src.config import PipelineConfig, load_config
from src.data import load_datasets
from src.pipeline import (
    annotate_blast_results,
    blast_to_bed,
    cache_sequences,
    compute_sample_attribution,
    generate_report,
    get_samples,
    parity_test,
    run_blast_stage,
    run_predictions,
    select_top_regions,
)


def _load_config(args) -> PipelineConfig:
    config = load_config(args.config)
    if getattr(args, "device", None):
        config.raw.setdefault("hardware", {})["device"] = args.device
    return config


def _add_inline_sample(config: PipelineConfig, token: str) -> bool:
    match = re.fullmatch(r"(phage|bacteria)_(\d+)", token)
    if not match:
        return False

    branch, idx_str = match.groups()
    idx = int(idx_str)

    datasets = config.raw.setdefault("datasets", {})
    if not datasets:
        raise ValueError("Datasets not configured; cannot interpret inline sample")

    _, _, couples_df = load_datasets(datasets)
    if idx < 0 or idx >= len(couples_df):
        raise ValueError(f"Sample index {idx} out of range (0..{len(couples_df) - 1})")

    samples_cfg = config.raw.setdefault("samples", [])
    if any(sample.get("name") == token for sample in samples_cfg):
        return True

    samples_cfg.append({"name": token, "idx": idx, "branch": branch})
    logging.info("Added inline sample '%s' (idx=%s, branch=%s)", token, idx, branch)
    return True


def _select_samples(config, requested: Iterable[str] | None) -> List[str]:
    available = [sample["name"] for sample in get_samples(config)]
    if not requested:
        return available

    resolved: List[str] = []
    for token in requested:
        if token in available:
            resolved.append(token)
            continue
        if _add_inline_sample(config, token):
            available = [sample["name"] for sample in get_samples(config)]
            resolved.append(token)
            continue
        raise ValueError(f"Unknown sample(s): {token}")
    return resolved


def cmd_cache(args) -> None:
    config = _load_config(args)
    cache_sequences(config)


def cmd_predict(args) -> None:
    config = _load_config(args)
    run_predictions(config)


def cmd_explain(args) -> None:
    config = _load_config(args)
    for sample in _select_samples(config, args.sample):
        compute_sample_attribution(config, sample)


def cmd_regions(args) -> None:
    config = _load_config(args)
    for sample in _select_samples(config, args.sample):
        select_top_regions(config, sample)


def cmd_blast(args) -> None:
    config = _load_config(args)
    for sample in _select_samples(config, args.sample):
        run_blast_stage(config, sample)


def cmd_annotate(args) -> None:
    config = _load_config(args)
    for sample in _select_samples(config, args.sample):
        annotate_blast_results(config, sample)


def cmd_bed(args) -> None:
    config = _load_config(args)
    for sample in _select_samples(config, args.sample):
        blast_to_bed(config, sample)


def cmd_report(args) -> None:
    config = _load_config(args)
    generate_report(config)


def cmd_test(args) -> None:
    config = _load_config(args)
    parity_test(config)


def cmd_run(args) -> None:
    config = _load_config(args)
    cache_sequences(config)
    run_predictions(config)
    for sample in _select_samples(config, args.sample):
        compute_sample_attribution(config, sample)
        select_top_regions(config, sample)
        run_blast_stage(config, sample)
        annotate_blast_results(config, sample)
        blast_to_bed(config, sample)
    if config.raw.get("report", {}).get("enabled", True):
        generate_report(config)


def main() -> None:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Preferred torch device (overrides config)",
    )

    parser = argparse.ArgumentParser(
        description="GradCAM 1D pipeline CLI",
        parents=[common_parser],
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "cache",
        help="Encode sequences and cache tensors",
        parents=[common_parser],
    ).set_defaults(func=cmd_cache)

    subparsers.add_parser(
        "predict",
        help="Run model predictions",
        parents=[common_parser],
    ).set_defaults(func=cmd_predict)

    explain_parser = subparsers.add_parser(
        "explain",
        help="Compute attributions for samples",
        parents=[common_parser],
    )
    explain_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    explain_parser.set_defaults(func=cmd_explain)

    regions_parser = subparsers.add_parser(
        "regions",
        help="Select top regions for samples",
        parents=[common_parser],
    )
    regions_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    regions_parser.set_defaults(func=cmd_regions)

    blast_parser = subparsers.add_parser(
        "blast",
        help="Run BLAST for samples",
        parents=[common_parser],
    )
    blast_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    blast_parser.set_defaults(func=cmd_blast)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Fetch GenBank annotations",
        parents=[common_parser],
    )
    annotate_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    annotate_parser.set_defaults(func=cmd_annotate)

    bed_parser = subparsers.add_parser(
        "bed",
        help="Generate BED files from BLAST hits",
        parents=[common_parser],
    )
    bed_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    bed_parser.set_defaults(func=cmd_bed)

    subparsers.add_parser(
        "report",
        help="Build summary report",
        parents=[common_parser],
    ).set_defaults(func=cmd_report)

    subparsers.add_parser(
        "test",
        help="Run parity test",
        parents=[common_parser],
    ).set_defaults(func=cmd_test)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute full pipeline",
        parents=[common_parser],
    )
    run_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI surface
        logging.getLogger().exception("Command failed: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
