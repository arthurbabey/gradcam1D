# Roadmap & TODOs — Phage–Bacteria XAI

**Goal.** A reproducible pipeline that explains a 2-branch 1D-CNN (bacteria + phage), extracts key DNA regions, annotates them (BLAST/GenBank), and reports biological insights.

**Background**: We have a dataset of bacterial and phage DNA sequences (one-hot encoded) with labels
indicating if the pair interacts (yes/no). The model is a two-branch 1D CNN that processes
a bacterium genome (padded to ~7,000,000 bp) and a phage genome (padded to ~200,000 bp) and then
concatenates their feature representations to predict interaction. We want to use XAI techniques (GradCAM, Integrated Gradients, etc.) to find which parts of each sequence contribute most to the
prediction. By extracting those important regions and aligning them via BLAST to known sequences, we
hope to discover genes or motifs (e.g. phage tail fibers, bacterial surface receptors) that drive the
interaction.

**Artifacts.**  
`model.py` (2-branch CNN) · `utils.py` (IUPAC one-hot, caching) · `pipeline_walkthrough.ipynb`  
New modules to add: `src/xai.py`, `src/peaks.py`, `src/blast.py`, `src/report.py`, `cli.py`

---

## Track A — Engineering & Reproducibility

**Goals**: Streamline the entire analysis into a reproducible, one-command pipeline, and provide a
containerized environment to ensure consistent results across different systems. By the end of this
track, one should be able to go from raw CSV data to a final report with a single command. All
dependencies (PyTorch, Captum, Biopython, BLAST+, etc.) should be encapsulated in an environment,
and the workflow should be easy to run on any machine (or Docker) with minimal setup.

**Tasks.**
- **Pipeline.** `config.yaml` (paths, seeds, methods, BLAST); Snakemake rules: `cache → predict → explain → peaks → blast → annotate → bed → report`. Stable run names.
- **CLI.** `cli.py` subcommands: `cache/predict/explain/blast/annotate/bed/report`.
- **Env.** `environment.yml` (torch, captum, biopython, blast+, snakemake); `Dockerfile` with BLAST+; mount `data/` & `results/`.

**Deliverables.** `workflow/Snakefile`, `config.yaml`, `environment.yml`, `Dockerfile`, `cli.py`.

Running snakemake --cores 4 (or an equivalent one-command execution via CLI) on a fresh
machine takes raw input (CSV sequences) and produces a complete results directory without
any manual intervention or errors. This includes all intermediate files (cached data,
predictions, attribution scores, region files, BLAST results, annotations) and a final report. For
example, a user should be able to clone the repo, adjust the config with input file paths, and
execute one command to get results. 

---

## Track B — XAI Methods & Comparison

**Goals**: Extend the explainability of the model by implementing multiple attribution methods, and build
a unified interface to compute and compare these attributions. We want to go beyond Grad-CAM and
have methods like Saliency maps, Integrated Gradients, PositionalSHAP, etc.
The goal is to determine which methods are most reliable and informative for our application and possibly recommend a default method for routine
analysis.

**Tasks.**
- **Methods** Saliency/GuidedBP · IG · GradCAM · Positional SHAP.
- **Outputs.** Standardize to per-base 1D arrays; mask padding; return `{bacteria, phage}`.
- **Eval.** Notebook: runtime/memory; rank correlations & top-K overlaps; stability with fixed seeds.
- **Code.** In `notebooks.ipynb`, use different `method=` to have plots.

**Deliverables.** `src/xai.py` (`explain(sample, branch, method, **kw) → np.ndarray`), comparison notebook.

- A new module (e.g., src/xai.py ) that contains implementations for the attribution methods
and a unified interface to invoke them. The code should be well-documented, explaining how
each method works at a high level (with references if helpful).
-A comparison report or notebook (Jupyter Notebook) that benchmarks and visualizes the
methods. Expect to include charts of attribution scores and a summary table of metrics (e.g., a
table showing pairwise Spearman correlations between methods, time per sample, etc.).
-An update to the pipeline or CLI to allow choosing an attribution method (for example, via config
or CLI argument). 

---

## Track C — Signal Processing & Region Extraction

**Goals**: Enhance how we extract important genomic regions from the raw attribution scores. Initially, a
simple fixed-size sliding window approach was used (e.g., take the top scoring window of fixed length).
This track aims to develop a more adaptive, signal processing-informed approach to identify peaks of
importance in the attribution signals. By doing so, we hope to capture meaningful biological regions
(like genes or regulatory elements) of varying lengths, rather than arbitrary fixed-size windows. The end
result should be a procedure that, given an attribution score array for a sequence, returns a set of
significant regions (with start/end positions on the genome) that likely correspond to biologically
relevant features, with reduced false positives and more consistency.

**Tasks.**

- **Peaks.** : Implement a peak-finding algorithm on the attribution signal for
each sequence. Instead of pre-defining window size, identify local maxima (peaks) in the 1D
attribution score array (Adaptative peaks). You can use functions like SciPy’s signal.find_peaks which supports finding peaks by prominence and
other criteria.
- **Selection.** How to choose the thresholds : (percentile (top 1%) / absolute top 10 regions / local-baseline (higher than neighboor regions)).
- **Post.** Expand/contract to local minima; deduplicate overlaps; compute region metrics (mean/max/area).
- **Viz.** Sanity plots; Show attribution and signal for regions, visualisation of the signals.

**Deliverables.** `src/peaks.py`, example `*.bed`/`*.bedgraph`, short demo notebook.

A module/file src/peaks.py (or integrated into attribution.py ) containing functions for
peak detection and region extraction
Example output files for a few samples: e.g., results/sample1/regions.bed , results/
sample1/attribution.bedgraph , which demonstrate the format and content of the region
extraction.
A Jupyter notebook showcasing the improvements of this dynamic approach. This notebook should include plots of attribution signals with the new regions
overlaid.

The new region extraction method should output a reasonable number of regions per sample
(for instance, not hundreds of tiny fragments, and not just one giant region unless truly only one
area is important). "Reasonable" might be on the order of a few regions for an interacting pair,
depending on model behavior. This likely means your parameters are tuned so that noise doesn't
produce peaks and real signals do.

---

## Track D — Biology, Annotation & Reporting

**Goals.** Turn regions into interpretable biology; clear per-sample + aggregate reports.

**Tasks.**
- **BLAST.** Remote or local BLAST+ (config); filters by %id, coverage, e-value; strand handling; multi-hit support.
- **Annotate.** Fetch GenBank; extract gene/product/protein_id ± neighbor context; simple keyword tagging (e.g., tail fiber, integrase, CRISPR).
- **Report.** Per-sample HTML/PDF with attribution plots, regions, BLAST table; cohort summary (recurring genes/functions); export TSV + BED/BEDGRAPH.

**Deliverables.** `src/blast.py`, `src/report.py`, Jinja2 templates under `report/`, `blast_features_summary.tsv`.

**Acceptance.** One command yields per-sample reports + an aggregate summary with sensible annotations.

---



---

## Milestones (suggested order)
1) **A1** Pipeline skeleton + config + env.  
2) **B1** Implement IG/Saliency/DeepLIFT via `src/xai.py`; wire into pipeline.  
3) **C1** Adaptive peaks replace fixed windows; quick visual validation.  
4) **D1** Basic BLAST+annotation and per-sample report.  
5) **A2/B2/D2** Dockerize; method benchmark & defaults; polished reports & docs.

---

## Notes for the Student
- Start with the walkthrough notebook; then run the CLI/Snakemake.
- Keep seeds fixed; document parameters in outputs.
- Develop on the tiny sample dataset; scale later.
- Prefer libraries (Captum/SciPy/Biopython/BLAST+) over re-implementing.
