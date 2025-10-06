Contributing Guide
==================

Environment
- Use the Conda environment defined in `environment.yml` (`conda env create -f environment.yml`).
- Activate before running tools: `conda activate gradcam1d`.

Workflow
- Configuration lives in `config.yaml`. Update or add profiles rather than hardcoding paths.
- Reuse shared modules in `src/` for new features; avoid duplicating logic in standalone scripts.
- Prefer adding new CLI subcommands (`cli.py`) or Snakemake rules (`workflow/Snakefile`) so that automation stays consistent.

Coding Style
- Python: follow PEP8, use type hints where reasonable, keep functions small and documented with docstrings.
- Write concise inline comments only when the intent is non-obvious (heavy commenting is discouraged).
- Default to channels-first tensors `(4, L)` and reuse helpers from `src/data.py` for encoding/decoding.

Testing & Validation
- Run `python3 cli.py test` after significant model or data changes to ensure parity.
- For new modules, add lightweight unit or integration tests (e.g., encode/decode roundtrip, peak detection invariants).
- When adding pipeline stages, ensure Snakemake also knows about the new outputs.

Pull Requests / Reviews
- Describe the motivation, configuration changes, and testing evidence in the PR body.
- Highlight any expected long-running steps (e.g., BLAST) so reviewers know how to reproduce.
- Keep commits focused; squash noisy WIP commits before merging.

Documentation
- Update `README.md` and/or `TODO.md` when workflow or roadmap details change.
- If you add new config options, document them inline in `config.yaml` comments or README.
