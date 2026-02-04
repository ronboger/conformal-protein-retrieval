# Claude Code Guidelines for CPR

## Working Patterns That Help

### Verification-First Development
- Before changing code, verify current behavior matches expectations
- Run existing tests before and after changes
- For paper reproduction, verify numbers match before claiming success
- Use `scripts/verify_*.py` to check paper claims

### Incremental Validation
- When running long jobs, check intermediate results (e.g., α=0.1 before waiting for all α levels)
- Use SLURM job logs to monitor progress: `cat logs/job_*.log | tail -20`
- Submit fast/reduced trials first to validate approach, then full runs

### Cleanup as You Go
- Archive (don't delete) old scripts - they may have useful patterns
- Use `scripts/archive/` and `notebooks/*/archive/` for superseded code
- Keep only essential SLURM scripts in main directories
- Consolidate documentation rather than creating new files

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
