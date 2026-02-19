# Claude Code Guidelines for CPR

## Working Patterns

### Before Writing Code
- **Describe your approach first** and wait for approval before implementing
- **Ask clarifying questions** if requirements are ambiguous - don't assume
- **If a task requires changes to more than 3 files**, stop and break it into smaller tasks first
- Verify current behavior matches expectations before changing anything

### While Writing Code
- Run existing tests before and after changes
- For paper reproduction, verify numbers match before claiming success
- Submit fast/reduced trials first to validate approach, then full runs

### After Writing Code
- **List what could break** and suggest tests to cover edge cases
- Run the test suite to confirm nothing regressed
- Archive (don't delete) old scripts - they may have useful patterns

### Bug Fixing
- **Start by writing a test that reproduces the bug**
- Fix the code until the test passes
- Keep the test to prevent regression

### Learning From Mistakes
- **When corrected, add a new rule to this file** so the mistake never happens again
- Document gotchas and edge cases discovered during debugging

### Session Continuity
- Check `DEVELOPMENT.md` changelog for recent work
- Check running SLURM jobs: `squeue -u ronb`
- Check `results/*.csv` for computed values
- The development log below tracks session-to-session context

---

## Bash Guidelines

### IMPORTANT: Avoid commands that cause output buffering issues
- DO NOT pipe through `head`, `tail`, `less`, or `more` when monitoring
- Use command-specific flags: `git log -n 10` not `git log | head -10`
- For log files, read directly rather than piping through filters

### IMPORTANT: Use $HOME2 for storage, not $HOME
- `$HOME` (/home/ronb) has limited quota - builds will fail
- `$HOME2` (/groups/doudna/projects/ronb/) has 2 PB storage
- Set: `APPTAINER_CACHEDIR=$HOME2/.apptainer_cache`
- Set: `PIP_CACHE_DIR=$HOME2/.pip_cache`

### IMPORTANT: Use SLURM for GPU or heavy CPU tasks
- NEVER run GPU code on login nodes - submit to SLURM
- Partitions: `standard` (CPU), `gpu` (GPU), `memory` (high-mem)
- Always use `eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"` in SLURM
- Example scripts: `scripts/slurm_*.sh`

---

## Project-Specific Guidelines

### Paper Reference
- **Title**: "Functional protein mining with conformal guarantees"
- **Journal**: Nature Communications (2025) 16:85
- **DOI**: https://doi.org/10.1038/s41467-024-55676-y

### Verified Paper Claims ✅
| Claim | Paper Value | Verified Value |
|-------|-------------|----------------|
| Syn3.0 annotation (α=0.1) | 39.6% (59/149) | 39.6% (59/149) |
| FDR threshold (α=0.1) | 0.9999802250 | 0.9999801 |
| DALI TPR | 82.8% | 81.8% |
| DALI DB reduction | 31.5% | 31.5% |
| CLEAN loss ≤ α | 1.0 | 0.97 |

### Core Algorithms (in `protein_conformal/util.py`)
- `get_thresh_FDR()` / `get_thresh_new_FDR()` - FDR threshold
- `get_thresh_new()` - FNR threshold
- `simplifed_venn_abers_prediction()` - Calibrated probabilities
- `scope_hierarchical_loss()` - Hierarchical loss
- `load_database()` / `query()` - FAISS operations

### ⚠️ Data Leakage Warning
**DO NOT USE** `conformal_pfam_with_lookup_dataset.npy` from backup directories.
**USE** `pfam_new_proteins.npy` from Zenodo - produces correct threshold.

---

## Key Files Reference

### CLI
- `protein_conformal/cli.py` - Main CLI (`cpr embed`, `cpr search`, `cpr verify`)

### Threshold Computation
- `scripts/compute_fdr_table.py` - FDR thresholds (use `--partial` for partial match)
- `scripts/compute_fnr_table.py` - FNR thresholds
- `scripts/slurm_compute_fdr_thresholds.sh` - SLURM wrapper
- `scripts/slurm_compute_fnr_thresholds.sh` - SLURM wrapper

### Verification
- `scripts/verify_syn30.py` - JCVI Syn3.0 (Figure 2A)
- `scripts/verify_dali.py` - DALI prefiltering (Tables 4-6)
- `scripts/verify_clean.py` - CLEAN enzyme (Tables 1-2)

### Results
- `results/fdr_thresholds.csv` - FDR thresholds with stats
- `results/fnr_thresholds.csv` - FNR exact match thresholds
- `results/fnr_thresholds_partial.csv` - FNR partial match thresholds
- `results/dali_thresholds.csv` - DALI prefiltering results

### Documentation
- `GETTING_STARTED.md` - User quick-start (most important)
- `DEVELOPMENT.md` - Dev status and changelog
- `DATA.md` - Data file documentation
- `REPO_ORGANIZATION.md` - Paper figures → code mapping

---

## Development Log

### 2026-02-18 - Deploy Fix + AFDB + CLEAN

**Completed:**
- Fixed baked file path conflict: CLEAN centroid files moved from `/app/data/clean/` to `/app/bundled/clean/` (was creating real dir that blocked volume symlink)
- Fixed Gradio 6 dropdown validation: converted float choices to strings (Gradio 6.5.1 strict type comparison)
- Deployed AFDB + CLEAN modes to production
- All 56 tests passing
- Production URL: `https://doudna-lab--cpr-gradio-ui.modal.run`

**Architecture (3 Modal containers):**
- `Embedder` — ProtTrans T5 + Protein-Vec on T4 GPU
- `CLEANEmbedder` — ESM-1b + LayerNormNet on A10G GPU
- `ui` — Gradio on CPU, calls `.remote()` to GPU containers

**Branch:** `gradio`

---

### 2026-02-11 - Gradio UI/UX Overhaul (Phase A + B)

**Completed:**
- Phase A: Mounted partial threshold CSVs in Modal, clamped FDR threshold to [0,1], renamed "UniProt" → "Swiss-Prot (540K)"
- Phase B: Per-query dropdown filter, dual exact/partial Pfam probabilities, Probability Filter mode, hide uncharacterized checkbox, Syn3.0 full 149-protein FASTA, embedding + FAISS result caching, probability vs. rank line plot, "Save Embeddings (.npy)" button, detail panel for full sequences/metadata
- Fixed: float32 artifact where `sims.max()` = 1.000000238 (clamped with `min(lhat, 1.0)`)
- Fixed: `raw_matches` now calibrated before caching so probability plot has real data
- Fixed: `filter_by_query` TypeError (wrong args to `_build_prob_plot`)
- Removed α=0.001 from dropdown (threshold always hits >1.0)

**Open issue — Table cell clipping (see HANDOFF.md):**
- User wants Google Sheets-like cell behavior: clipped text, click to expand in-place
- `wrap=False` doesn't constrain column width in Gradio 5 — cells expand to fit content
- CSS `table-layout: fixed` didn't work — Gradio 5 scoped styles override external CSS
- Current workaround: explicit string truncation with "…" (readable but loses click-to-expand)
- **Next step**: Try JavaScript via `gr.Blocks(js=...)` for inline styles (highest specificity)
- See `HANDOFF.md` for full details and untried approaches

**Branch:** `gradio` (all changes uncommitted)
**Deploy:** `modal deploy modal_app.py`

---

### 2026-02-03 - Cleanup & Consolidation

**Completed:**
- Archived 16 redundant scripts to `scripts/archive/`
- Archived duplicate Python files from `notebooks/pfam/`
- Consolidated threshold CSVs (removed "simple" versions)
- Added full threshold tables to `GETTING_STARTED.md`
- Merged `SESSION_SUMMARY.md` into `DEVELOPMENT.md`
- Archived outdated `docs/QUICKSTART.md`
- Updated this file with working patterns

**FDR Job Status:**
- Job 1012664 (fdr-fast): 20 trials, α=0.1 verified as 0.99998006

**Final Structure:**
- 4 SLURM scripts (build, embed, fdr, fnr)
- 4 results CSVs (fdr, fnr, fnr_partial, dali)
- 51 tests passing

---

### 2026-02-02 - Verification & CLI

**Completed:**
- Verified Syn3.0: 59/149 = 39.6% ✅
- Fixed FDR bug (1D/2D array handling)
- Created CLI with `embed`, `search`, `verify` commands
- Created verification scripts for DALI, CLEAN
- Investigated data leakage in backup dataset

**Environment:**
- Conda: `conformal-s` (Python 3.11.10)
- Packages: faiss 1.9.0, torch 2.5.0, numpy 1.26.4

---

### 2026-01-28 - Initial Session

- Removed duplicate `src/protein_conformal/`
- Created `pyproject.toml` and test infrastructure
- Created initial documentation

---

## Best Practices

### Testing
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_util.py -v        # Just util tests
pytest tests/test_cli.py -v         # Just CLI tests
```

### Git Workflow
- Work on feature branches, not main
- Run tests before committing
- Use descriptive commits referencing paper figures/tables

### SLURM Jobs
```bash
squeue -u ronb                      # Check running jobs
cat logs/job_*.log | tail -20       # Check recent output (use Read tool)
scancel JOBID                       # Cancel a job
```

### Code Style
- Follow patterns in `protein_conformal/util.py`
- Use numpy for numerical operations
- Use FAISS for similarity search
- Notebooks for analysis, package for algorithms

### Gradio 6 Gotchas
- **`wrap=False` does NOT clip text** — it only sets `white-space: nowrap`. Columns still expand to fit content. You MUST also constrain column width for clipping to work.
- **External CSS often doesn't work** — Gradio uses scoped Svelte styles that override external CSS even with `!important`. Use `gr.Blocks(js=...)` with inline styles via JavaScript instead.
- **`interactive=True` on Dataframe** — clicking a cell enters edit mode, showing full content. This IS the click-to-expand behavior if the initial display is clipped.
- **Dropdown choices must be strings** — Gradio 6 does strict type comparison in dropdown preprocess. Use `["0.1", "0.2"]` not `[0.1, 0.2]`, and `float()` in the handler.
- **Test on compute nodes** — `pytest tests/` takes ~5 min, prefer SLURM `standard` partition.
- **Gradio version** — Modal currently installs Gradio 6.5.1. Check specific version behavior before assuming CSS/JS patterns work.

### Modal Deployment
- `modal deploy modal_app.py` — takes ~2s, deploys Gradio + GPU embedder
- `modal serve modal_app.py` — local testing with live logs (use for debugging)
- Container idle timeout: 20 min (`scaledown_window=60*20`)
- First request after cold start: loads ~1GB lookup embeddings from volume → builds FAISS index → slow. Subsequent requests within same container are fast (in-memory cache).
- Baked files: `results/*.csv`, `protein_conformal/`, `data/gene_unknown/unknown_aa_seqs.fasta` (at `/app/bundled/syn30.fasta`), `data/clean/ec_centroid_*` (at `/app/bundled/clean/`)
- **NEVER bake files to `/app/data/`** — it creates a real directory that blocks the `/app/data -> /vol/data` symlink. Always bake to `/app/bundled/` and copy at startup.
- Volume (`/vol`): lookup embeddings, metadata TSV, Protein-Vec models, HF cache, AFDB data
- **CLEAN_repo dependency** — `CLEAN_repo/app/data/pretrained/split100.pth` must exist at deploy time (baked into CLEANEmbedder image)
