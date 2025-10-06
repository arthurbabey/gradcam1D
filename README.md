Phage–Bacteria Interaction Pipeline
===================================

Overview
- Predict phage–bacteria interactions with a dual-branch 1D CNN (`src/modeling.py`).
- Explain predictions via Grad-CAM and derive biologically meaningful regions (`src/attribution.py`, `src/regions.py`).
- Run BLAST on top regions and annotate hits with GenBank metadata (`src/blast_tools.py`).
- Produce concise reports summarising predictions, regions, and annotations (`src/reporting.py`).

Repository Layout
- `config.yaml` – central configuration for datasets, thresholds, attribution, BLAST, logging, parity.
- `cli.py` – unified command-line entrypoint (cache, predict, explain, regions, blast, annotate, bed, report, test, run).
- `workflow/Snakefile` – Snakemake workflow wiring the pipeline end-to-end.
- `src/` – modular implementation (data handling, model, attribution, pipeline orchestration).
- `environment.yml` – Conda environment (PyTorch, Captum, Snakemake, BLAST+, Biopython, etc.).
- `notebooks/pipeline_walkthrough.ipynb` – hands-on tour of a single example (mirrors CLI stages).
- Legacy scripts (`gradcam_to_blast.py`, `blast_to_genebank.py`, `blast_to_bedd.py`) now wrap the shared modules for backward compatibility.

Setup
1. Create the environment: `conda env create -f environment.yml`
2. Activate: `conda activate gradcam1d`
3. Verify BLAST+: `blastn -version`
4. Adjust `config.yaml` if needed (paths, samples, attribution parameters, BLAST mode).

Using the CLI
```
# Step-by-step (functions can be run individually)
python3 cli.py cache
python3 cli.py predict
python3 cli.py explain --sample phage_0
python3 cli.py regions --sample phage_0
python3 cli.py blast --sample phage_0
python3 cli.py annotate --sample phage_0
python3 cli.py bed --sample phage_0
python3 cli.py report
python3 cli.py test

# Or run everything for all samples listed in config.yaml
python3 cli.py run
```
Outputs are written to `results_pipeline/` (deterministic naming per sample).

Running the Workflow (Snakemake)
```
snakemake --snakefile workflow/Snakefile --cores 4
```
Snakemake reads `config.yaml`, triggers the CLI subcommands, and produces:
- `cache/cache.done` & `cache_manifest.json`
- `predictions/predictions.csv`
- `attributions/{sample}.npy|json`
- `regions/{sample}.tsv` & `fasta/{sample}.fasta`
- `blast/{sample}.tsv`, `annotations/{sample}.tsv`, `bed/{sample}.bed`
- `report/summary.md`
- `tests/parity_reference.json`
Logs live in `results_pipeline/logs/`.

Config Highlights (`config.yaml`)
- `paths`: dataset directory, cache location, model checkpoint, results directory.
- `datasets`: CSV filenames (bacteria, phages, couples).
- `samples`: list of `{name, idx, branch}` processed by CLI/Snakemake.
- `attribution`: method (`guided` or `layercam`), target layers per branch, number/size of regions.
- `thresholds`: padding lengths for bacteria/phage sequences.
- `blast`: enable/disable, database name, remote/local mode.
- `report`: toggle report generation.
- `parity`: sample indices tracked by the parity test and where the reference is stored.

Validation & Logging
- CSV schemas are checked before use (required columns).
- Cached tensors are verified for shape/dtype before downstream work.
- Each CLI command and Snakemake rule writes a dedicated log file under `results_pipeline/logs/`.
- `python3 cli.py test` performs a parity check (stores/compares predictions for known samples).

Notebooks & Legacy Tools
- Use `notebooks/pipeline_walkthrough.ipynb` to understand the pipeline on a single sample.
- Older notebooks (`gradcam1D.ipynb`, `keras2torch.ipynb`, `predictions.ipynb`) remain for reference; prefer the new CLI or Snakemake for fresh runs.

Next Steps
- See `TODO.md` for engineering, XAI, signal-processing, and biology-focused extensions.
- Track contributions via `CONTRIBUTING.md` (coding style, tests, review flow).
