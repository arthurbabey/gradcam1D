from Bio import Entrez
import pandas as pd
import os

# ==== CONFIGURATION ====
Entrez.email = "arthur.babey@heig-vd.ch"  # <-- IMPORTANT: replace with your email
blast_file = "./results/results_phage_91_Public_20250407_124523/blast_output.tsv"    # <-- Your BLAST result file
bed_output = "./results/results_phage_91_Public_20250407_124523/blast_hits.bed"            # Output BED file
genome_folder = "phage_genomes"         

# ==== 1. DOWNLOAD GENOMES ====
def download_fasta(accession_list, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for acc in accession_list:
        fasta_path = os.path.join(output_folder, f"{acc}.fasta")
        if os.path.exists(fasta_path):
            print(f"Already exists: {acc}")
            continue
        try:
            with Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text") as handle:
                record = handle.read()
                with open(fasta_path, "w") as f:
                    f.write(record)
                print(f"Downloaded: {acc}")
        except Exception as e:
            print(f"Error downloading {acc}: {e}")

# ==== 2. CONVERT TO BED ====
def convert_blast_to_bed(blast_path, bed_path):
    colnames = [
        "query_id", "accession", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"
    ]
    df = pd.read_csv(blast_path, sep="\t", names=colnames)

    # Determine strand
    df["strand"] = df.apply(lambda row: "+" if row["sstart"] <= row["send"] else "-", axis=1)
    df["start"] = df[["sstart", "send"]].min(axis=1) - 1  # BED format: 0-based start
    df["end"] = df[["sstart", "send"]].max(axis=1)
    df["name"] = df["query_id"]
    df["score"] = df["bitscore"].astype(int)

    bed_df = df[["accession", "start", "end", "name", "score", "strand"]]
    bed_df.to_csv(bed_path, sep="\t", index=False, header=False)
    print(f"BED file saved as: {bed_path}")

# ==== MAIN ====
if __name__ == "__main__":
    print("Reading BLAST output...")
    blast_df = pd.read_csv(blast_file, sep="\t", header=None)
    accessions = blast_df[1].unique().tolist()  # column 1 is 'subject_id'

    print("\nStep 1: Downloading genomes...")
    download_fasta(accessions, genome_folder)

    print("\nStep 2: Converting BLAST to BED...")
    convert_blast_to_bed(blast_file, bed_output)

    print("\nAll done! You may now load the FASTA genomes and the BED file in IGV.")