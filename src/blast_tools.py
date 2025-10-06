"""BLAST integration and downstream annotation helpers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import subprocess


BLAST_COLUMNS = [
    "query_id",
    "subject_id",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
]


def run_blast_search(
    fasta_path: Path,
    output_path: Path,
    database: str,
    mode: str = "remote",
) -> None:
    if shutil.which("blastn") is None:
        raise RuntimeError("blastn not found in PATH")

    cmd = ["blastn", "-query", str(fasta_path), "-outfmt", "6"]
    if mode == "remote":
        cmd.extend(["-db", database, "-remote"])
    else:
        cmd.extend(["-db", database])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Running BLAST: %s", " ".join(cmd))
    with output_path.open("w", encoding="utf-8") as out_handle:
        subprocess.run(cmd, check=True, stdout=out_handle)


def parse_blast_top_hits(blast_tsv: Path) -> Dict[str, Dict[str, str]]:
    if not blast_tsv.exists():
        return {}
    hits: Dict[str, Dict[str, str]] = {}
    with blast_tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < len(BLAST_COLUMNS):
                continue
            query_id = parts[0]
            if query_id not in hits:
                hits[query_id] = dict(zip(BLAST_COLUMNS[1:], parts[1:]))
    return hits


def fetch_genbank_features(
    accession: str,
    region_start: Optional[int] = None,
    region_end: Optional[int] = None,
    email: str = "arthur.babey@heig-vd.ch",
) -> List[Dict[str, str]]:
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = email
    try:
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text")
        content = handle.read()
        handle.close()
    except Exception as exc:  # pragma: no cover - network dependent
        logging.warning("Failed to fetch GenBank record for %s: %s", accession, exc)
        return []

    record = SeqIO.read(StringIO(content), "genbank")
    features = []
    for feature in record.features:
        start = int(feature.location.start)
        end = int(feature.location.end)
        if region_start is not None and region_end is not None:
            if end < region_start or start > region_end:
                continue
        features.append(
            {
                "type": feature.type,
                "location": f"{start}..{end}",
                "strand": feature.location.strand,
                "qualifiers": feature.qualifiers,
            }
        )
    return features


def classify_region(features: Iterable[Dict[str, str]]) -> str:
    coding = {"CDS"}
    noncoding = {"tRNA", "rRNA", "ncRNA", "misc_RNA"}
    types = {feat.get("type", "") for feat in features}
    if types & coding:
        return "coding"
    if types & noncoding:
        return "non-coding"
    return "intergenic"


def process_blast_results(
    blast_tsv: Path,
    output_path: Path,
    email: str = "arthur.babey@heig-vd.ch",
) -> None:
    if not blast_tsv.exists():
        raise FileNotFoundError(blast_tsv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    with blast_tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < len(BLAST_COLUMNS):
                continue
            pident = float(parts[2])
            if pident < 99:
                continue
            query_id = parts[0]
            accession = parts[1]
            sstart, send = int(parts[8]), int(parts[9])
            region_start, region_end = min(sstart, send), max(sstart, send)
            features = fetch_genbank_features(accession, region_start, region_end, email=email)
            classification = classify_region(features)
            if not features:
                rows.append(
                    {
                        "query_id": query_id,
                        "accession": accession,
                        "region_start": region_start,
                        "region_end": region_end,
                        "feature_type": "source",
                        "location": "N/A",
                        "strand": ".",
                        "gene": "",
                        "locus_tag": "",
                        "product": "",
                        "classification": classification,
                        "organism": "",
                    }
                )
            else:
                for feature in features:
                    quals = feature.get("qualifiers", {})
                    rows.append(
                        {
                            "query_id": query_id,
                            "accession": accession,
                            "region_start": region_start,
                            "region_end": region_end,
                            "feature_type": feature.get("type", ""),
                            "location": feature.get("location", ""),
                            "strand": feature.get("strand", "."),
                            "gene": quals.get("gene", [""])[0],
                            "locus_tag": quals.get("locus_tag", [""])[0],
                            "product": quals.get("product", [""])[0],
                            "classification": classification,
                            "organism": quals.get("organism", [""])[0],
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, sep="\t", index=False)


def convert_blast_to_bed(blast_tsv: Path, bed_path: Path) -> None:
    df = pd.read_csv(blast_tsv, sep="\t", names=BLAST_COLUMNS)
    df["strand"] = df.apply(lambda row: "+" if row["sstart"] <= row["send"] else "-", axis=1)
    df["start"] = df[["sstart", "send"]].min(axis=1) - 1
    df["end"] = df[["sstart", "send"]].max(axis=1)
    df["name"] = df["query_id"]
    df["score"] = df["bitscore"].astype(int)
    bed = df[["subject_id", "start", "end", "name", "score", "strand"]]
    bed_path.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(bed_path, sep="\t", header=False, index=False)
