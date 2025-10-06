#!/usr/bin/env python3
import os
import sys
import re
import shutil
import logging
import argparse
import subprocess
from datetime import datetime

import torch
import pandas as pd
from captum.attr import LayerGradCam, GuidedGradCam

# Import the project modules (model definition and utilities)
from model import PerphectInteractionModel
from utils import InteractionDataset, precompute_and_cache_sequences, tensor_to_sequence
from utils import combine_attributions, BACTERIUM_THRESHOLD, PHAGE_THRESHOLD
from visualisation import find_top_k_important_windows

def load_model_and_data(public=True):
    """Load the pre-trained model and dataset (with cached one-hot sequences)."""
    # Load pre-trained model weights
    model = PerphectInteractionModel()
    model_path = "./data/saved_model/model_v1.pth"
    if not os.path.isfile(model_path):
        logging.error(f"Model weights file not found at {model_path}")
        sys.exit(1)
    # Load state dict (the model was saved with state_dict)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set model to evaluation mode

    # Load datasets (CSV files)
    data_dir = "./data/public_data_set" if public else "./data/private_data_set"
    try:
        bacteria_df = pd.read_csv(os.path.join(data_dir, "bacteria_df.csv"))
        phages_df   = pd.read_csv(os.path.join(data_dir, "phages_df.csv"))
        couples_df  = pd.read_csv(os.path.join(data_dir, "couples_df.csv"))
    except Exception as e:
        logging.error(f"Failed to load dataset CSVs in {data_dir}: {e}")
        sys.exit(1)
    # Precompute or load cached one-hot encoded sequences
    cache_dir = os.path.join(data_dir, "cached_tensors")
    phage_paths, bacterium_paths = precompute_and_cache_sequences(phages_df, bacteria_df, cache_dir)
    # Initialize the interaction dataset
    dataset = InteractionDataset(couples_df, phage_paths, bacterium_paths)
    return model, dataset

def get_sample_sequence(dataset, idx):
    """Retrieve the one-hot tensors and DNA sequences for a given sample index."""
    try:
        bacterium_tensor, phage_tensor, label = dataset[idx]
    except IndexError:
        logging.error(f"Index {idx} is out of range for the dataset.")
        sys.exit(1)
    # Add batch dimension (model expects batch-first tensors)
    bacterium_tensor = bacterium_tensor.unsqueeze(0)
    phage_tensor = phage_tensor.unsqueeze(0)
    # Convert one-hot tensor to DNA sequence string
    # (Ensure numpy array input for the conversion function)
    if torch.is_tensor(bacterium_tensor):
        bact_seq = tensor_to_sequence(bacterium_tensor.cpu().numpy())
    else:
        bact_seq = tensor_to_sequence(bacterium_tensor)
    if torch.is_tensor(phage_tensor):
        phage_seq = tensor_to_sequence(phage_tensor.cpu().numpy())
    else:
        phage_seq = tensor_to_sequence(phage_tensor)
    return bacterium_tensor, phage_tensor, bact_seq, phage_seq

def compute_gradcam_attribution(model, branch, target_layer, inputs, mode):
    """
    Compute Grad-CAM attributions using Captum for the given model and target layer.
    Returns a 1D numpy array of attribution scores (length = full sequence length).
    """
    mode = mode.lower()
    # Prepare the attribution object based on mode
    if mode == "layercam":
        # Standard Grad-CAM (LayerGradCam) on the chosen layer
        gradcam = LayerGradCam(model, target_layer)
        # Compute attribution for the given input tuple
        # Captum LayerGradCam returns a tensor for the target layer's output
        layer_attr = gradcam.attribute(inputs)  # shape expected: (1, 1, L_layer)
        # Upsample and normalize to full input length using combine_attributions
        input_len = BACTERIUM_THRESHOLD if branch == "bacteria" else PHAGE_THRESHOLD
        combined = combine_attributions(layer_attr.detach(), input_length=input_len)
        return combined  # numpy array (length = input_len)
    elif mode == "guided":
        # Guided Grad-CAM (Grad-CAM with guided backpropagation)
        guided_gc = GuidedGradCam(model, target_layer)
        # Compute attributions for both inputs; Captum will upsample to input dims
        attributions = guided_gc.attribute(inputs)  # returns a tuple for multi-input
        # Select the attribution for the branch of interest
        if isinstance(attributions, tuple) or isinstance(attributions, list):
            attr_tensor = attributions[0] if branch == "bacteria" else attributions[1]
        else:
            # In case of single input models (not in this dual-input scenario)
            attr_tensor = attributions
        # Combine channels (one-hot channels) into a single importance score per position
        combined = combine_attributions(attr_tensor.detach())
        return combined  # numpy array (length = input_len)
    else:
        raise ValueError(f"Invalid gradcam_type '{mode}'. Use 'layercam' or 'guided'.")

def save_regions_to_fasta(sequence, regions, fasta_path):
    """Save the specified sequence regions to a FASTA file."""
    with open(fasta_path, "w") as f:
        for i, (start, end, mean_val) in enumerate(regions, start=1):
            subseq = sequence[start:end]
            # Replace padding character 'Z' with 'N' (unknown nucleotide) for BLAST compatibility
            subseq = subseq.replace('Z', 'N')
            f.write(f">Seq{i}\n")
            f.write(subseq + "\n")

def run_blast_search(fasta_path, output_path):
    """Run a BLASTn search on the given FASTA file against NCBI nt (remote)."""
    cmd = [
        "blastn", "-query", fasta_path, "-db", "nt", "-remote",
        "-out", output_path, "-outfmt", "6"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        # If BLAST failed, log stderr for diagnosis
        logging.error("BLAST search failed. BLAST stderr output:")
        logging.error(result.stderr.decode('utf-8', errors='ignore'))
        raise subprocess.CalledProcessError(result.returncode, cmd)

def parse_blast_top_hits(blast_tsv_path, num_queries):
    """
    Parse the BLAST output (format 6) and return the top hit for each query sequence.
    Returns a dict: {query_id: {sacc, pident, evalue, bitscore}} for each query.
    """
    top_hits = {}
    if not os.path.isfile(blast_tsv_path):
        return top_hits
    with open(blast_tsv_path, "r") as bf:
        for line in bf:
            cols = line.strip().split("\t")
            if len(cols) < 12:
                continue  # skip any malformed lines
            qid, sseqid = cols[0], cols[1]
            pident, evalue, bitscore = cols[2], cols[10], cols[11]
            if qid not in top_hits:
                # Record the first (top-ranked) hit for this query
                try:
                    pid_val = float(pident)
                except:
                    pid_val = None
                try:
                    bits_val = float(bitscore)
                except:
                    bits_val = None
                top_hits[qid] = {
                    "sseqid": sseqid,
                    "pident": pid_val,
                    "evalue": evalue,
                    "bitscore": bits_val
                }
            if len(top_hits) == num_queries:
                # Collected top hit for all query sequences
                break
    return top_hits

def fetch_entrez_description(accession):
    """Fetch the nucleotide record title (description) for a given accession via Entrez."""
    from Bio import Entrez
    Entrez.email = "arthur.babey@heig-vd.ch"
    try:
        handle = Entrez.esummary(db="nuccore", id=accession)
        record = Entrez.read(handle)
        if record and isinstance(record, list):
            summary = record[0]
        else:
            summary = record
        title = summary.get("Title") if summary else None
        if title:
            return title  # e.g., "Escherichia coli O157 chromosome, complete genome"
        # If Title is not available, try a different approach (e.g., efetch)
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text", seq_start=1, seq_stop=1)
        # Read a small part of the record to get the definition line
        text = handle.read(200)  # read first 200 chars
        # Usually "DEFINITION" line contains the description
        if "DEFINITION" in text:
            # Extract content after "DEFINITION"
            defin_line = text.split("DEFINITION")[1].split("\n")[0].strip()
            return defin_line
    except Exception as e:
        logging.warning(f"Entrez fetch failed for {accession}: {e}")
    return None


# PUBLIC DATASET

# True Positive index: 0, sequence_length = 53124, bacterium_sequence_length = 6988208
#False Positive index: 1994, sequence_length = 54867, bacterium_sequence_length = 6988208
#True Negative index: 1993, sequence_length = 41526, bacterium_sequence_length = 6988208
#False Negative index: 91, sequence_length = 46732, bacterium_sequence_length = 6988208

# PRIVATE DATASET

#True Positive index: 0, sequence_length = 18227, bacterium_sequence_length = 2872769
#False Positive index: 13, phage_sequence_length = 43114, bacterium_sequence_length = 2692583
#True Negative index: 10, phage_sequence_length = 18227, bacterium_sequence_length = 2692583
#False Negative index: 7, phage_sequence_length = 41708, bacterium_sequence_length = 2774913


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM attribution and BLAST analysis for phage-bacterium CNN.")
    parser.add_argument("--idx", type=int, required=True, help="Index of the interaction sample to analyze.")
    parser.add_argument("--target_layer", type=str, required=True, help="Name of the target layer for Grad-CAM (e.g., conv3).")
    parser.add_argument("--branch", type=str, choices=["bacteria", "phage"], required=True, help="Which branch to analyze (bacteria or phage).")
    parser.add_argument("--num_regions", type=int, default=3, help="Number of top regions to extract.")
    parser.add_argument("--window_size", type=int, default=500, help="Window size (in bp) for each region.")
    parser.add_argument("--gradcam_type", type=str, choices=["layercam", "guided"], default="layercam", help="Type of Grad-CAM to perform: 'layercam' or 'guided'.")
    parser.add_argument("--public", action="store_true", help="Use public dataset (default is private).")
    parser.add_argument("sequence_length", type=int, help="Length of the sequence to analyze to cut for padding.")
    parser.add_argument("--dry_run", action="store_true", help="If set, skip BLAST and Entrez steps (for debugging).")
    args = parser.parse_args()

    # name 
    name = args.branch + "_" + str(args.idx) + "_" + str(args.public)
    
    # Set up output directory (include timestamp for uniqueness)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results_{name}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Configure logging to file and console
    log_file = os.path.join(out_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info("=== Grad-CAM Attribution and BLAST Analysis ===")
    logging.info(f"Parameters: idx={args.idx}, target_layer='{args.target_layer}', branch={args.branch}, "
                 f"num_regions={args.num_regions}, window_size={args.window_size}, gradcam_type={args.gradcam_type}")

    # Load model and dataset according to flag
    model, dataset = load_model_and_data(public=args.public)
    logging.info("Model and data loaded successfully.")

    # Get the specified sample's input tensors and sequences
    bacterium_tensor, phage_tensor, bact_seq, phage_seq = get_sample_sequence(dataset, args.idx)
    logging.info(f"Sample {args.idx}: Retrieved bacterium and phage sequences.")

    # Determine branch-specific inputs
    branch = args.branch.lower()
    if branch == "bacteria":
        target_seq = bact_seq
    else:
        target_seq = phage_seq

    # Get the target layer module from the model
    if branch == "bacteria":
        branch_module = model.bacteria_branch
    else:
        branch_module = model.phage_branch
    if not hasattr(branch_module, args.target_layer):
        logging.error(f"Layer '{args.target_layer}' not found in {branch} branch of model.")
        sys.exit(1)
    target_layer = getattr(branch_module, args.target_layer)
    logging.info(f"Targeting layer '{args.target_layer}' in {branch} branch for Grad-CAM.")

    # Prepare model inputs tuple for attribution
    inputs = (bacterium_tensor.requires_grad_(), phage_tensor.requires_grad_())

    # Compute Grad-CAM attributions
    try:
        attribution_scores = compute_gradcam_attribution(model, branch, target_layer, inputs, args.gradcam_type)
    except Exception as e:
        logging.error(f"Grad-CAM computation failed: {e}")
        sys.exit(1)
    logging.info(f"Computed {args.gradcam_type} attributions for layer '{args.target_layer}'.")

    # Cut attribution and sequence to drop 0 padding with sequence length
    attribution_scores = attribution_scores[:args.sequence_length]
    target_seq = target_seq[:args.sequence_length]
    
    # Identify top K important regions in the selected sequence
    top_windows = find_top_k_important_windows(target_seq, attribution_scores, k=args.num_regions, window_size=args.window_size)
    if not top_windows:
        logging.error("No regions identified (check if window_size is smaller than sequence length or attributions are all zero).")
        sys.exit(1)
    logging.info(f"Top {len(top_windows)} regions (window size {args.window_size}) in {branch} sequence:")
    for i, (start, end, mean_val) in enumerate(top_windows, start=1):
        logging.info(f"  Region {i}: {start}-{end} (mean attribution={mean_val:.6f})")

    # Save the top regions to a FASTA file
    fasta_path = os.path.join(out_dir, "top_regions.fasta")
    save_regions_to_fasta(target_seq, top_windows, fasta_path)
    logging.info(f"Saved top regions to FASTA file: {fasta_path}")

    # Run BLAST search on the extracted regions (if not in dry-run mode)
    blast_tsv = os.path.join(out_dir, "blast_output.tsv")
    if args.dry_run:
        logging.info("--dry_run specified: skipping BLAST search.")
    else:
        # Check BLAST availability
        if shutil.which("blastn") is None:
            logging.error("NCBI BLAST+ is not installed or 'blastn' not found in PATH. Please install BLAST+ to run the search.")
            sys.exit(1)
        try:
            logging.info("Running BLAST search against nt (remote)... This may take a while.")
            run_blast_search(fasta_path, blast_tsv)
            logging.info(f"BLAST search completed. Results saved to {blast_tsv}")
        except subprocess.CalledProcessError:
            logging.error("BLAST search failed. Exiting.")
            sys.exit(1)

    # Parse BLAST results and fetch top hit details
    if not args.dry_run:
        top_hits = parse_blast_top_hits(blast_tsv, num_queries=len(top_windows))
        if not top_hits:
            logging.info("No BLAST hits found for the given regions.")
        else:
            logging.info("Top BLAST hit for each region:")
            # Attempt to import Entrez for fetching additional info
            entrez_available = False
            try:
                from Bio import Entrez
                entrez_available = True
            except ImportError:
                logging.warning("Biopython Entrez not available; skipping detailed info fetch.")
            # Iterate through each query's top hit
            for qid, hit in top_hits.items():
                sseqid = hit["sseqid"]
                identity = hit["pident"]
                evalue = hit["evalue"]
                # Extract an accession from sseqid (handles 'gi|..|ref|ACC|', 'ACC.version', etc.)
                accession = None
                match = re.search(r"[A-Za-z0-9_]+\.\d+", sseqid)
                if match:
                    accession = match.group(0)
                else:
                    # Fallback: if sseqid contains pipes
                    parts = sseqid.split("|")
                    if "gi" in parts:
                        # if format is gi|GI|db|ACC|, ACC is usually at index 3
                        accession = parts[3] if len(parts) > 3 else None
                    elif len(parts) > 1:
                        # If no gi, maybe format like ref|ACC|
                        accession = parts[1]
                    else:
                        accession = sseqid  # use entire as accession if no delimiters
                # Fetch description via Entrez if possible
                description = None
                if entrez_available and accession:
                    description = fetch_entrez_description(accession)
                # Log the hit summary
                if identity is not None:
                    ident_str = f"{identity:.1f}%"
                else:
                    ident_str = "NA"
                if description:
                    logging.info(f"  {qid}: {accession} – {description} (identity {ident_str}, e-value {evalue})")
                else:
                    logging.info(f"  {qid}: {accession or sseqid} (identity {ident_str}, e-value {evalue})")
    logging.info("Analysis complete. See log.txt for details and BLAST results for alignments.")
    
if __name__ == "__main__":
    main()
