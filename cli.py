"""Unified CLI for pipeline operations."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from src.config import load_config
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


def _select_samples(config, requested: Iterable[str] | None) -> List[str]:
    available = [sample["name"] for sample in get_samples(config)]
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown sample(s): {', '.join(missing)}")
    return list(requested)


def cmd_cache(args) -> None:
    config = load_config(args.config)
    cache_sequences(config)


def cmd_predict(args) -> None:
    config = load_config(args.config)
    run_predictions(config)


def cmd_explain(args) -> None:
    config = load_config(args.config)
    for sample in _select_samples(config, args.sample):
        compute_sample_attribution(config, sample)


def cmd_regions(args) -> None:
    config = load_config(args.config)
    for sample in _select_samples(config, args.sample):
        select_top_regions(config, sample)


def cmd_blast(args) -> None:
    config = load_config(args.config)
    for sample in _select_samples(config, args.sample):
        run_blast_stage(config, sample)


def cmd_annotate(args) -> None:
    config = load_config(args.config)
    for sample in _select_samples(config, args.sample):
        annotate_blast_results(config, sample)


def cmd_bed(args) -> None:
    config = load_config(args.config)
    for sample in _select_samples(config, args.sample):
        blast_to_bed(config, sample)


def cmd_report(args) -> None:
    config = load_config(args.config)
    generate_report(config)


def cmd_test(args) -> None:
    config = load_config(args.config)
    parity_test(config)


def cmd_run(args) -> None:
    config = load_config(args.config)
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
    parser = argparse.ArgumentParser(description="GradCAM 1D pipeline CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("cache", help="Encode sequences and cache tensors").set_defaults(func=cmd_cache)
    subparsers.add_parser("predict", help="Run model predictions").set_defaults(func=cmd_predict)

    explain_parser = subparsers.add_parser("explain", help="Compute attributions for samples")
    explain_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    explain_parser.set_defaults(func=cmd_explain)

    regions_parser = subparsers.add_parser("regions", help="Select top regions for samples")
    regions_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    regions_parser.set_defaults(func=cmd_regions)

    blast_parser = subparsers.add_parser("blast", help="Run BLAST for samples")
    blast_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    blast_parser.set_defaults(func=cmd_blast)

    annotate_parser = subparsers.add_parser("annotate", help="Fetch GenBank annotations")
    annotate_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    annotate_parser.set_defaults(func=cmd_annotate)

    bed_parser = subparsers.add_parser("bed", help="Generate BED files from BLAST hits")
    bed_parser.add_argument("--sample", action="append", help="Sample name (can repeat)")
    bed_parser.set_defaults(func=cmd_bed)

    subparsers.add_parser("report", help="Build summary report").set_defaults(func=cmd_report)
    subparsers.add_parser("test", help="Run parity test").set_defaults(func=cmd_test)

    run_parser = subparsers.add_parser("run", help="Execute full pipeline")
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
