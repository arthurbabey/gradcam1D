#!/usr/bin/env python3
import os
import sys
import argparse
import re
from io import StringIO
from Bio import Entrez, SeqIO

# Set your email (REQUIRED by NCBI Entrez)
Entrez.email = "arthur.babey@heig-vd.ch"

def fetch_genbank_features(accession, region_start=None, region_end=None):
    """
    Fetch GenBank features for a given accession and optional region.
    Returns a list of features overlapping the region.
    """
    try:
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text")
        gb_text = handle.read()
        handle.close()
        record = SeqIO.read(StringIO(gb_text), "genbank")
    except Exception as e:
        print(f"Error fetching GenBank record for {accession}: {e}")
        return []

    features = []
    for feature in record.features:
        if region_start is not None and region_end is not None:
            f_start = int(feature.location.start)
            f_end = int(feature.location.end)
            if f_end < region_start or f_start > region_end:
                continue  # skip non-overlapping
        features.append({
            "type": feature.type,
            "location": f"{int(feature.location.start)}..{int(feature.location.end)}",
            "strand": feature.location.strand,
            "qualifiers": feature.qualifiers
        })
    return features

def classify_region(features):
    """
    Classify a region based on the list of overlapping features.
    Priority: coding > non-coding > intergenic
    """
    coding_types = {"CDS"}
    noncoding_types = {"tRNA", "rRNA", "ncRNA", "misc_RNA"}

    types = set(f["type"] for f in features)
    if types & coding_types:
        return "coding"
    elif types & noncoding_types:
        return "non-coding"
    else:
        return "intergenic"

def process_blast_results(blast_tsv_path):
    """
    Read a BLAST TSV file, filter for 100% hits, fetch GenBank features, 
    classify biological context, and write results to a TSV file.
    """
    results = []
    out_dir = os.path.dirname(os.path.abspath(blast_tsv_path))
    output_file = os.path.join(out_dir, "blast_features_summary.tsv")

    with open(blast_tsv_path, "r") as infile:
        for line in infile:
            cols = line.strip().split("\t")
            if len(cols) < 12:
                continue
            query_id = cols[0]
            accession = cols[1]
            pident = cols[2]
            if float(pident) < 99:
                continue  # only keep 99% + matches
            try:
                s_start = int(cols[8])
                s_end = int(cols[9])
            except ValueError:
                continue
            region_start = min(s_start, s_end)
            region_end = max(s_start, s_end)

            features = fetch_genbank_features(accession, region_start, region_end)
            classification = classify_region(features)

            if not features:
                # Still record the region even if only source is found
                results.append({
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
                    "classification": classification
                })
            else:
                for feat in features:
                    q = feat.get("qualifiers", {})
                    results.append({
                        "query_id": query_id,
                        "accession": accession,
                        "region_start": region_start,
                        "region_end": region_end,
                        "feature_type": feat.get("type", ""),
                        "location": feat.get("location", ""),
                        "strand": feat.get("strand", "."),
                        "gene": q.get("gene", [""])[0],
                        "locus_tag": q.get("locus_tag", [""])[0],
                        "product": q.get("product", [""])[0],
                        "classification": classification,
                        "organism": q.get("organism", [""])[0],
                    })


    # Write results to file
    with open(output_file, "w") as out:
        header = ["query_id", "accession", "region_start", "region_end", 
                  "feature_type", "location", "strand", "gene", "locus_tag", "product", "classification", "organism"]
        out.write("\t".join(header) + "\n")
        for r in results:
            row = "\t".join(str(r[h]) for h in header)
            out.write(row + "\n")

    print(f"Results written to {output_file}")

    print(f"\n Classification written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="BLAST TSV => GenBank feature summarizer with classification.")
    parser.add_argument("--blast_tsv", required=True, help="Path to blast_output.tsv file.")
    args = parser.parse_args()

    if not os.path.exists(args.blast_tsv):
        print(f"File not found: {args.blast_tsv}")
        sys.exit(1)

    process_blast_results(args.blast_tsv)

if __name__ == "__main__":
    main()
