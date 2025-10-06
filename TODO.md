Roadmap and TODOs
=================

Purpose
- This file organizes next steps into four tracks you can assign to students.
- Each track lists goals, concrete tasks, and deliverables. Adapt as needed.

Context Artifacts
- Model: `model.py` (two-branch 1D CNN)
- Data utils: `utils.py` (encoding, caching, dataset, attribution helpers)
- Grad-CAM + BLAST CLI: `gradcam_to_blast.py`
- BLAST→GenBank: `blast_to_genebank.py`
- BLAST→BED + downloads: `blast_to_bedd.py`
- Top-window selection: `visualisation.py`
- Guided notebook: `pipeline_walkthrough.ipynb`

Track A — Engineering and Reproducibility
- Goals
  - One-command, reproducible pipeline to go from CSVs → cached tensors → predictions → attributions → top regions → BLAST → feature summaries → report.
  - Containerized environment with BLAST+, Captum, Biopython.

- Tasks
  - Pipeline
    - [x] Add a `config.yaml` describing dataset paths, thresholds, attribution config, BLAST options, output dir.
    - [x] Implement a Snakemake workflow (or Makefile) with rules: `cache`, `predict`, `attribution`, `top_regions`, `blast`, `genebank`, `bed`, `report`.
    - [ ] Support both public and private datasets via config profiles.
    - [x] Use stable, human-readable run names (e.g., `results/{branch}/{idx}`) instead of timestamps.
  - CLI unification
    - [x] Create a single entrypoint `cli.py` with subcommands: `cache`, `predict`, `explain`, `blast`, `annotate`, `bed`, `report` that call into existing functions.
    - [x] Refactor shared utilities into a `src/` package (keep function names; do not change behavior).
  - Environment
    - [x] Create `environment.yml` (conda) with exact versions (torch, captum, biopython, pandas, snakemake).
    - [ ] Create `Dockerfile` installing BLAST+ and the environment; add `docker-compose.yml` with bind mounts for `data/` and `results/`.
  - Reliability
    - [x] Add logging to files under each run directory and a top-level `logs/` folder.
    - [x] Add schema checks for CSVs (required columns and types) and cache integrity checks for `.npy` shapes.
    - [x] Add a small parity check to verify loaded `model_v1.pth` predicts identically across machines for a handful of samples.
  - Documentation
    - [x] Expand `README.md` with: install, run, pipeline diagram, expected inputs/outputs, and troubleshooting.
    - [x] Add `CONTRIBUTING.md` with coding style, how to add new steps.

- Deliverables
  - `workflow/Snakefile` + `config.yaml`, `environment.yml`, `Dockerfile`.
  - Unified `cli.py` executing end-to-end runs.

- Acceptance
  - `snakemake --cores 4` runs from raw CSVs to a finished report directory without manual edits.
  - `docker run …` executes the same and passes basic checks.

Track B — XAI Methods and Comparisons
- Goals
  - Provide multiple attribution methods for the dual-input model and a unified API to compute/import them.
  - Compare methods qualitatively and quantitatively (stability, runtime, agreement).

- Tasks
  - Implement methods (Captum)
    - [ ] Saliency and GuidedBackprop on inputs (per branch).
    - [ ] Integrated Gradients (IG) with robust baselines; try `LayerIntegratedGradients` for deeper layers.
    - [ ] DeepLift and DeepLiftShap on inputs.
    - [ ] InputXGradient; Occlusion with suitable sliding window for long sequences.
    - [ ] NoiseTunnel variants (SmoothGrad / VarGrad) for stability.
  - Multi-input handling
    - [ ] Standardize outputs to 1D per-base arrays after channel-combine (reuse `combine_attributions`).
    - [ ] Ensure methods return separate arrays for `bacteria` and `phage` branches; align lengths to true sequence lengths (not padded).
  - Evaluation
    - [ ] Add a small benchmark notebook to compare runtime and memory per method on a fixed sample set.
    - [ ] Compute rank correlations and overlap of top regions across methods.
    - [ ] Add reproducibility check (repeat runs with fixed seeds and NoiseTunnel parameters).
  - UX
    - [ ] Extend `pipeline_walkthrough.ipynb` to toggle method via a single variable and plot overlays.

- Deliverables
  - `src/xai.py` exposing `explain(sample, branch, method, target_layer, params) → 1D attribution`.
  - Comparison notebook with plots and summary table.

- Acceptance
  - At least five methods implemented and validated on both branches.
  - A summary document recommending two default methods and parameter presets.

Track C — Signal Processing and Region Extraction
- Goals
  - Improve region extraction beyond a fixed-size sliding window by adapting to signal characteristics.

- Tasks
  - Peak detection and segmentation
    - [ ] Implement dynamic peak calling (e.g., prominence-based) on the normalized attribution signal.
    - [ ] Merge adjacent peaks if closer than a minimum gap; enforce minimum width.
    - [ ] Multi-scale windows: evaluate several window sizes; keep windows that are stable across scales.
  - Thresholding strategies
    - [ ] Absolute (e.g., > 0.9), percentile-based (top 1–5%), and relative-to-local-baseline thresholds.
    - [ ] Control number of regions per sample (top-K with diversity constraint).
  - Post-processing
    - [ ] Expand/contract regions to nearest local minima to capture full motif context.
    - [ ] Deduplicate overlapping regions; track attribution mean, max, and area-under-importance as features.
  - Validation
    - [ ] Sanity plots of attribution vs. selected regions; export IGV tracks (BEDGRAPH/BED) with scores.
    - [ ] Compare region sets between methods from Track B; compute Jaccard overlaps.

- Deliverables
  - `src/peaks.py` with functions for peak calling, merging, multi-scale selection.
  - Notebook demonstrating improvements vs. fixed windows on representative samples.

- Acceptance
  - Regions are fewer, more precise, and stable across methods and runs; visual inspection looks biologically plausible.

Track D — Biological Queries and Reporting
- Goals
  - Turn BLAST hits into interpretable biology with richer annotations and clear summaries.

- Tasks
  - BLAST improvements
    - [ ] Support both remote BLAST and local BLAST+ with a pre-downloaded db, controlled by config.
    - [ ] Add hit filtering by identity, coverage, e-value; handle multi-hit regions and strand.
  - Annotation
    - [ ] Extend `blast_to_genebank.py` to extract gene name, product, protein_id, and neighboring features within ±N bp.
    - [ ] Map to functional vocabularies (e.g., gene ontology where possible, basic keyword tagging).
    - [ ] Summarize recurring genes/functional categories across samples and methods.
  - Reporting
    - [ ] Generate per-sample HTML/PDF reports with: attribution plot, selected regions, BLAST top hits, feature table.
    - [ ] Aggregate cohort report: tables of most frequent accessions/features, example IGV snapshots, and method comparisons.
    - [ ] Provide export formats: TSV/CSV and BED/BEDGRAPH for genome viewers.

- Deliverables
  - `report/` templates (Jinja2 or nbconvert) and a CLI `report` command to render.
  - Enhanced `blast_features_summary.tsv` with standardized columns and metadata.

- Acceptance
  - A single command produces both per-sample and aggregate reports with clear tables/figures suitable for slides.

Cross-Cutting and Cleanup
- [ ] Archive legacy notebooks to `notebooks/legacy/` (keep `pipeline_walkthrough.ipynb` as the main guide).
- [ ] Remove `.DS_Store` and ensure `__pycache__/` is ignored (already in `.gitignore`).
- [ ] Add minimal unit tests for: one-hot encode/decode, caching, dataset shapes, attribution combination, peak caller.
- [ ] Add small sample dataset for CI (short sequences) to validate pipeline steps without large genomes.

Suggested Order of Work (Milestones)
1) A1 Pipeline skeleton (config + Snakemake + env) and stable run naming.
2) B1 Implement 2–3 XAI methods with unified API; hook into pipeline.
3) C1 Replace fixed-window selection with peak calling; compare on examples.
4) D1 Enrich annotations and basic per-sample report.
5) A2 Dockerize; add local BLAST option; finalize docs and quickstart.

Notes for Students
- Use `pipeline_walkthrough.ipynb` to understand each step on a single example.
- For batch jobs, prefer the unified CLI or Snakemake once implemented.
- Keep runs deterministic (set seeds) when comparing XAI methods; document parameters in outputs.
