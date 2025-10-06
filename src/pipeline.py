"""High-level orchestration helpers for pipeline stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .attribution import (
    compute_attribution,
    get_sequences,
    prepare_sample,
    save_attribution,
    topk_summary,
    trim_to_sequence,
)
from .blast_tools import convert_blast_to_bed, process_blast_results, run_blast_search
from .config import PipelineConfig
from .data import (
    InteractionDataset,
    check_cache_integrity,
    dump_manifest,
    load_datasets,
    precompute_and_cache_sequences,
    write_fasta,
)
from .logging_utils import setup_logging
from .modeling import build_model, predict_dataset, run_parity_check
from .regions import find_top_k_important_windows, save_regions_table
from .reporting import write_report


def thresholds_from_config(config: PipelineConfig) -> Dict[str, int]:
    thresholds = config.raw.get("thresholds", {})
    return {
        "phage": int(thresholds.get("phage", 200000)),
        "bacteria": int(thresholds.get("bacteria", 7000000)),
    }


def get_samples(config: PipelineConfig) -> List[Dict[str, str]]:
    samples = config.raw.get("samples", [])
    if not samples:
        raise ValueError("No samples defined in config.yaml")
    return samples


def prepare_results_dirs(results_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "cache": results_dir / "cache",
        "predictions": results_dir / "predictions",
        "attributions": results_dir / "attributions",
        "regions": results_dir / "regions",
        "fasta": results_dir / "fasta",
        "blast": results_dir / "blast",
        "annotations": results_dir / "annotations",
        "bed": results_dir / "bed",
        "logs": results_dir / "logs",
        "report": results_dir / "report",
        "tests": results_dir / "tests",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def _sequence_lengths(config: PipelineConfig) -> Dict[str, Dict[str, int]]:
    bacteria_df, phages_df, _ = load_datasets(config.raw["datasets"])
    bacteria_map = {str(row["bacterium_id"]): int(row["sequence_length"]) for _, row in bacteria_df.iterrows()}
    phage_map = {str(row["phage_id"]): int(row["sequence_length"]) for _, row in phages_df.iterrows()}
    return {"bacteria": bacteria_map, "phage": phage_map}


def cache_sequences(config: PipelineConfig) -> Path:
    paths = config.raw["paths"]
    thresholds = thresholds_from_config(config)
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], "cache", config.raw.get("logging", {}).get("level", "INFO"))

    logging.info("Loading datasets from %s", paths["data_dir"])
    bacteria_df, phages_df, couples_df = load_datasets(config.raw["datasets"])

    logging.info("Caching sequences to %s", paths["cache_dir"])
    phage_map, bacteria_map = precompute_and_cache_sequences(
        phages_df, bacteria_df, Path(paths["cache_dir"]), thresholds
    )

    logging.info("Validating cached tensors")
    check_cache_integrity(phage_map, bacteria_map, thresholds)

    manifest_path = subdirs["cache"] / "cache_manifest.json"
    dump_manifest(manifest_path, phage_map, bacteria_map)

    done_file = subdirs["cache"] / "cache.done"
    done_file.write_text("ok", encoding="utf-8")
    return done_file


def load_dataset_with_cache(config: PipelineConfig) -> InteractionDataset:
    thresholds = thresholds_from_config(config)
    bacteria_df, phages_df, couples_df = load_datasets(config.raw["datasets"])
    phage_map, bacteria_map = precompute_and_cache_sequences(
        phages_df, bacteria_df, Path(config.raw["paths"]["cache_dir"]), thresholds
    )
    return InteractionDataset(couples_df, phage_map, bacteria_map)


def run_predictions(config: PipelineConfig) -> Path:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], "predict", config.raw.get("logging", {}).get("level", "INFO"))

    dataset = load_dataset_with_cache(config)
    model = build_model(Path(paths["model_checkpoint"]))
    records = predict_dataset(model, dataset)
    couples = dataset.couples
    for record in records:
        row = couples.iloc[int(record["dataset_index"])]
        record["phage_id"] = row["phage_id"]
        record["bacterium_id"] = row["bacterium_id"]
        record["interaction_type"] = row["interaction_type"]
    frame = pd.DataFrame(records)
    output_path = subdirs["predictions"] / "predictions.csv"
    frame.to_csv(output_path, index=False)
    logging.info("Wrote predictions to %s", output_path)
    return output_path


def _sample_lookup(config: PipelineConfig, sample_name: str) -> Dict[str, str]:
    for sample in get_samples(config):
        if sample["name"] == sample_name:
            return sample
    raise KeyError(f"Sample '{sample_name}' not found in config")


def compute_sample_attribution(config: PipelineConfig, sample_name: str) -> Tuple[Path, np.ndarray]:
    paths = config.raw["paths"]
    thresholds = thresholds_from_config(config)
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], f"explain_{sample_name}", config.raw.get("logging", {}).get("level", "INFO"))

    sample_cfg = _sample_lookup(config, sample_name)
    dataset = load_dataset_with_cache(config)
    model = build_model(Path(paths["model_checkpoint"]))

    sample_index = int(sample_cfg["idx"])
    bacterium_tensor, phage_tensor, label = prepare_sample(dataset, sample_index)
    sequences = get_sequences(bacterium_tensor, phage_tensor)

    branch = sample_cfg["branch"]
    target_layer_name = config.raw["attribution"]["target_layers"][branch]
    method = config.raw["attribution"].get("method", "guided")
    module = getattr(model, f"{branch}_branch")
    target_layer = getattr(module, target_layer_name)

    logging.info(
        "Computing %s Grad-CAM for %s (idx=%s, layer=%s)",
        method,
        sample_name,
        sample_cfg["idx"],
        target_layer_name,
    )

    attribution = compute_attribution(
        model,
        branch,
        method,
        target_layer,
        bacterium_tensor,
        phage_tensor,
        thresholds,
    )

    row = dataset.couples.iloc[sample_index]
    sequence_lengths = _sequence_lengths(config)
    id_col = "bacterium_id" if branch == "bacteria" else "phage_id"
    seq_length = sequence_lengths[branch][str(row[id_col])]
    sequence = sequences[branch][:seq_length]
    trimmed = trim_to_sequence(sequence, attribution)
    attr_path = subdirs["attributions"] / f"{sample_name}.npy"
    save_attribution(trimmed, attr_path)

    meta_path = subdirs["attributions"] / f"{sample_name}.json"
    meta = {
        "sample": sample_name,
        "branch": branch,
        "idx": sample_cfg["idx"],
        "method": method,
        "target_layer": target_layer_name,
        "label": label,
        "summary": topk_summary(trimmed),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Attribution saved to %s", attr_path)
    return attr_path, trimmed


def select_top_regions(config: PipelineConfig, sample_name: str) -> Tuple[Path, Path]:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], f"regions_{sample_name}", config.raw.get("logging", {}).get("level", "INFO"))

    attr_path = subdirs["attributions"] / f"{sample_name}.npy"
    if not attr_path.exists():
        raise FileNotFoundError(f"Attribution not found for {sample_name}: run explain stage first")

    attribution = np.load(attr_path)
    sample_cfg = _sample_lookup(config, sample_name)
    dataset = load_dataset_with_cache(config)
    sample_index = int(sample_cfg["idx"])
    bacterium_tensor, phage_tensor, _ = prepare_sample(dataset, sample_index)
    sequences = get_sequences(bacterium_tensor, phage_tensor)
    branch = sample_cfg["branch"]
    row = dataset.couples.iloc[sample_index]
    sequence_lengths = _sequence_lengths(config)
    id_col = "bacterium_id" if branch == "bacteria" else "phage_id"
    seq_length = sequence_lengths[branch][str(row[id_col])]
    sequence = sequences[branch][:seq_length]

    params = config.raw["attribution"]
    regions = find_top_k_important_windows(
        sequence,
        attribution,
        k=int(params.get("num_regions", 3)),
        window_size=int(params.get("window_size", 500)),
    )

    table_path = subdirs["regions"] / f"{sample_name}.tsv"
    save_regions_table(regions, table_path, branch, sample_name)

    fasta_path = subdirs["fasta"] / f"{sample_name}.fasta"
    write_fasta(sequence, regions, fasta_path)
    logging.info("Saved regions table to %s and FASTA to %s", table_path, fasta_path)
    return table_path, fasta_path


def run_blast_stage(config: PipelineConfig, sample_name: str) -> Path:
    paths = config.raw["paths"]
    blast_cfg = config.raw.get("blast", {})
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], f"blast_{sample_name}", config.raw.get("logging", {}).get("level", "INFO"))

    if not blast_cfg.get("enabled", True):
        placeholder = subdirs["blast"] / f"{sample_name}.tsv"
        placeholder.write_text("", encoding="utf-8")
        logging.warning("BLAST disabled in config; created empty %s", placeholder)
        return placeholder

    fasta_path = subdirs["fasta"] / f"{sample_name}.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found for {sample_name}; run regions stage first")

    output_path = subdirs["blast"] / f"{sample_name}.tsv"
    run_blast_search(
        fasta_path,
        output_path,
        blast_cfg.get("database", "nt"),
        blast_cfg.get("mode", "remote"),
    )
    logging.info("BLAST results saved to %s", output_path)
    return output_path


def annotate_blast_results(config: PipelineConfig, sample_name: str) -> Path:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], f"annotate_{sample_name}", config.raw.get("logging", {}).get("level", "INFO"))

    blast_path = subdirs["blast"] / f"{sample_name}.tsv"
    annot_path = subdirs["annotations"] / f"{sample_name}.tsv"
    if not blast_path.exists() or blast_path.stat().st_size == 0:
        annot_path.write_text("", encoding="utf-8")
        logging.warning("No BLAST hits for %s; annotation skipped", sample_name)
        return annot_path

    process_blast_results(blast_path, annot_path)
    logging.info("Annotation written to %s", annot_path)
    return annot_path


def blast_to_bed(config: PipelineConfig, sample_name: str) -> Path:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], f"bed_{sample_name}", config.raw.get("logging", {}).get("level", "INFO"))

    blast_path = subdirs["blast"] / f"{sample_name}.tsv"
    bed_path = subdirs["bed"] / f"{sample_name}.bed"
    if not blast_path.exists() or blast_path.stat().st_size == 0:
        bed_path.write_text("", encoding="utf-8")
        logging.warning("No BLAST hits for %s; BED skipped", sample_name)
        return bed_path

    convert_blast_to_bed(blast_path, bed_path)
    logging.info("BED file written to %s", bed_path)
    return bed_path


def generate_report(config: PipelineConfig) -> Path:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], "report", config.raw.get("logging", {}).get("level", "INFO"))
    samples = [sample["name"] for sample in get_samples(config)]
    report_path = subdirs["report"] / "summary.md"
    write_report(results_dir, samples, report_path)
    logging.info("Report written to %s", report_path)
    return report_path


def parity_test(config: PipelineConfig) -> Path:
    paths = config.raw["paths"]
    results_dir = Path(paths["results_dir"])
    subdirs = prepare_results_dirs(results_dir)
    setup_logging(subdirs["logs"], "parity", config.raw.get("logging", {}).get("level", "INFO"))

    dataset = load_dataset_with_cache(config)
    model = build_model(Path(paths["model_checkpoint"]))
    parity_cfg = config.raw.get("parity", {})
    indices = parity_cfg.get("sample_indices", [0])
    reference_path = subdirs["tests"] / parity_cfg.get("reference", "parity_reference.json")
    run_parity_check(model, dataset, indices, reference_path)
    logging.info("Parity reference stored at %s", reference_path)
    return reference_path
