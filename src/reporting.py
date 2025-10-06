"""Simple reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def build_summary_markdown(results_dir: Path, samples: List[str]) -> str:
    lines = ["# Pipeline Summary", ""]
    predictions_path = results_dir / "predictions" / "predictions.csv"
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        lines.append("## Predictions")
        lines.append(df.head().to_markdown(index=False))
        lines.append("")

    region_dir = results_dir / "regions"
    region_tables = []
    for sample in samples:
        path = region_dir / f"{sample}.tsv"
        if path.exists():
            table = pd.read_csv(path, sep="\t")
            table.insert(0, "sample_name", sample)
            region_tables.append(table)
    if region_tables:
        combined = pd.concat(region_tables, ignore_index=True)
        lines.append("## Top Regions")
        lines.append(combined.to_markdown(index=False))

    return "\n".join(lines)


def write_report(results_dir: Path, samples: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_summary_markdown(results_dir, samples), encoding="utf-8")
