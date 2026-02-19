# Development Notes: CPR Refactoring Project

This document tracks the ongoing refactoring of the Conformal Protein Retrieval (CPR) codebase.

**Paper**: [Functional protein mining with conformal guarantees](https://www.nature.com/articles/s41467-024-55676-y) (Nature Communications, 2025)

**Authors**: Ron S. Boger, Seyone Chithrananda, Anastasios N. Angelopoulos, Peter H. Yoon, Michael I. Jordan, Jennifer A. Doudna

---

## Current Status

**Branch**: `refactor/cpr-cleanup-and-tests`

### Verified Paper Results

| Claim | Paper | Reproduced | Status |
|-------|-------|------------|--------|
| Syn3.0 annotation | 39.6% (59/149) | 39.6% (59/149) | ✅ EXACT |
| FDR threshold (α=0.1) | 0.9999802250 | 0.9999801 | ✅ Match |
| DALI TPR | 82.8% | 81.8% | ✅ ~1% diff |
| DALI reduction | 31.5% | 31.5% | ✅ EXACT |
| CLEAN loss | ≤ α=1.0 | 0.97 | ✅ Pass |

### Completed Work

#### Phase 1: Code Cleanup ✅
- Removed duplicate `src/protein_conformal/` directory
- Archived 16 redundant SLURM/shell scripts
- Archived duplicate Python files from notebooks
- Fixed FDR threshold bug (1D/2D array handling)
- Fixed numpy deprecation warnings

#### Phase 2: CLI Implementation ✅
- Created `cpr` CLI with subcommands: `embed`, `search`, `verify`
- Unified `cpr search` accepts both FASTA and embeddings
- Added `--fdr`, `--fnr`, `--threshold`, `--no-filter` options
- Multi-model support: `--model protein-vec` or `--model clean`

#### Phase 3: Testing ✅
- 51 tests total (27 util + 24 CLI)
- All tests passing
- Regression tests for paper-critical values

#### Phase 4: Documentation ✅
- `GETTING_STARTED.md` - comprehensive user guide
- `DATA.md` - data file documentation
- `REPO_ORGANIZATION.md` - paper figures → code mapping
- Full threshold tables in docs

#### Phase 5: Containerization (Partial)
- Created `Dockerfile` and `apptainer.def`
- Apptainer build blocked by glibc mismatch (needs PyTorch 2.4+ base)

---

## File Structure

```
conformal-protein-retrieval/
├── protein_conformal/           # Main package
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point (`cpr` command)
│   ├── util.py                  # Core algorithms
│   ├── embed_protein_vec.py     # Protein-Vec embedding
│   ├── scope_utils.py           # SCOPe utilities
│   └── backend/                 # Gradio interface
├── scripts/                     # Standalone scripts
│   ├── compute_fdr_table.py     # FDR threshold computation
│   ├── compute_fnr_table.py     # FNR threshold computation
│   ├── verify_*.py              # Verification scripts
│   └── slurm_*.sh               # SLURM job scripts (4 kept)
├── notebooks/                   # Analysis notebooks
│   ├── pfam/                    # Pfam/Syn3.0 analysis
│   ├── scope/                   # SCOPe/DALI analysis
│   ├── clean_selection/         # CLEAN enzyme analysis
│   └── ec/                      # EC classification
├── tests/                       # Test suite
│   ├── conftest.py
│   ├── test_util.py             # 27 tests
│   └── test_cli.py              # 24 tests
├── results/                     # Computed thresholds
│   ├── fdr_thresholds.csv
│   ├── fnr_thresholds.csv
│   ├── fnr_thresholds_partial.csv
│   └── dali_thresholds.csv
└── data/                        # Data files (see DATA.md)
```

---

## Data Files

### ⚠️ Data Leakage Warning

**DO NOT USE** `conformal_pfam_with_lookup_dataset.npy` from backup directories. This dataset has data leakage:
- First 50 samples all have the same Pfam family "PF01266;"
- Positive rate is 3.00% (vs 0.22% in correct dataset)
- Produces incorrect FDR threshold

**USE**: `pfam_new_proteins.npy` from Zenodo with:
- 1,864 diverse samples
- 0.22% positive rate
- Produces threshold matching paper

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=protein_conformal --cov-report=html
```

---

## Remaining Work

1. **Complete FDR threshold table** - job running, α=0.1 verified
2. **Fix Apptainer build** - update to PyTorch 2.4+ base image
3. **Merge to main** - after final verification

---

## Changelog

### 2026-02-18
- Fixed deployed Modal app: baked CLEAN centroid files to `/app/bundled/clean/` instead of `/app/data/clean/` (was blocking `/app/data -> /vol/data` symlink)
- Fixed Gradio 6 dropdown validation: converted float choices to strings (Gradio 6.5.1 does strict type comparison)
- Migrated from HuggingFace downloads to Modal volume for all data storage
- Added AFDB (AlphaFold DB) clustered database option
- Added CLEAN enzyme classification mode (ESM-1b + LayerNormNet on A10G GPU)
- Added AFDB metadata conversion script
- All 56 tests passing
- Deployed to `https://doudna-lab--cpr-gradio-ui.modal.run`

### 2026-02-03
- Archived 16 redundant scripts to `scripts/archive/`
- Consolidated threshold CSVs, added full tables to GETTING_STARTED.md
- Removed duplicate Python files from notebooks

### 2026-02-02
- Verified JCVI Syn3.0 result: 59/149 = 39.6% ✅
- Fixed FDR threshold bug in `get_thresh_FDR()`
- Created CLI: `cpr embed`, `cpr search`, `cpr verify`
- All 51 tests passing

### 2026-01-28
- Initial cleanup session
- Removed duplicate `src/protein_conformal/`
- Created `pyproject.toml` and test infrastructure
