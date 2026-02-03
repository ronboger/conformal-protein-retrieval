# Development Notes: CPR Refactoring Project

This document tracks the ongoing refactoring of the Conformal Protein Retrieval (CPR) codebase to make it more usable, testable, and maintainable.

**Paper**: [Functional protein mining with conformal guarantees](https://www.nature.com/articles/s41467-024-55676-y) (Nature Communications, 2024)

**Authors**: Ron S. Boger, Seyone Chithrananda, Anastasios N. Angelopoulos, Peter H. Yoon, Michael I. Jordan, Jennifer A. Doudna

---

## Current Status (Branch: `refactor/cpr-cleanup-and-tests`)

### Completed Work

1. **Merged Gradio UI branch** (`origin/gradio-ron` → `main`)
   - Added Gradio web interface in `protein_conformal/backend/`
   - Added Dockerfile and `environment.yml`
   - Reorganized notebooks into `notebooks/` directory
   - Added FDR/FNR threshold precomputation scripts
   - Added `requirements.txt`

2. **Removed duplicate code**
   - Deleted `src/protein_conformal/` (duplicate of `protein_conformal/`)
   - Deleted `pfam/tmp.py` (temporary debug file)

3. **Set up modern Python packaging**
   - Created `pyproject.toml` with:
     - Package metadata and dependencies
     - CLI entry point: `cpr` command
     - Optional dependency groups: `[gui]`, `[api]`, `[dev]`, `[all]`
     - pytest and code quality tool configuration

4. **Created test infrastructure**
   - `tests/` directory with pytest fixtures
   - `tests/conftest.py` - shared fixtures for testing
   - `tests/test_util.py` - comprehensive test suite for `protein_conformal/util.py`

### Test Coverage

The test suite covers:

| Module | Functions Tested |
|--------|------------------|
| FASTA parsing | `read_fasta()` |
| FAISS operations | `load_database()`, `query()` |
| Risk metrics | `risk()`, `risk_1d()`, `calculate_false_negatives()`, `calculate_true_positives()` |
| Conformal thresholds | `get_thresh_new()`, `get_thresh_new_FDR()`, `get_thresh_FDR()` |
| Venn-Abers | `simplifed_venn_abers_prediction()`, `get_isotone_regression()` |
| Hierarchical loss | `scope_hierarchical_loss()` |
| Validation | `validate_lhat_new()` |

### Known Test Cases from Notebooks

From `notebooks/scope/analyze_scope_protein_vec.ipynb`:
- **FDR threshold**: `alpha=0.1, delta=0.5, N=100` → `lhat=0.999987906879849`, `risk=0.0358`
- **Data shape**: 400 queries × 14,777 lookup proteins
- **Similarity range**: 0.9992... to 0.9999...
- **Hierarchical loss**: `scope_hierarchical_loss('a.1.1.1', 'a.1.1.1')` → `(0, True)`

---

## Planned Work

### Phase 1: Validate Current Code (This Branch)

1. **Run test suite** and fix any failures
2. **Add integration tests** with small sample data
3. **Verify numerical reproducibility** against notebook outputs

### Phase 2: CLI Implementation

Create a clean CLI interface:

```bash
# Embedding
cpr embed input.fasta -o embeddings.npy --model protein-vec

# Search with FDR control
cpr search query.npy --lookup lookup.npy --fdr 0.1 -o results.csv

# Search with FNR control
cpr search query.npy --lookup lookup.npy --fnr 0.1 -o results.csv

# Compute probabilities
cpr probs results.csv --calibration pfam_new_proteins.npy --partial

# Launch GUI
cpr gui --port 7860
```

### Phase 3: Documentation

1. **Installation guide** with all dependencies
2. **Quick start** with example workflows
3. **Data download instructions** (Zenodo files)
4. **API reference** for programmatic use

### Phase 4: Code Cleanup

1. Remove remaining duplicate/dead code
2. Standardize imports and module structure
3. Add type hints to public functions
4. Ensure consistent error handling

---

## Required Data Files

### From Zenodo (https://zenodo.org/records/14272215)

| File | Size | Purpose |
|------|------|---------|
| `pfam_new_proteins.npy` | 2.5 GB | **CORRECT** calibration data for FDR/FNR control |

#### ⚠️ Data Leakage Warning

**DO NOT USE** `conformal_pfam_with_lookup_dataset.npy` from the backup directory. This dataset has **data leakage**:
- First 50 samples all have the same Pfam family "PF01266;" repeated
- Positive rate is 3.00% (vs 0.22% in correct dataset)
- Produces incorrect FDR threshold (~0.999965 vs paper's ~0.999980)

The correct dataset is `pfam_new_proteins.npy` with:
- 1,864 diverse samples with different Pfam families
- 0.22% positive rate matching expected calibration distribution
- Produces threshold ~0.999982 matching paper's 0.9999802250

See `scripts/quick_fdr_check.py` for verification.
| `lookup_embeddings.npy` | 1.1 GB | UniProt protein embeddings (lookup database) |
| `lookup_embeddings_meta_data.tsv` | 560 MB | Metadata for lookup proteins |
| `afdb_embeddings_protein_vec.npy` | 4.7 GB | AlphaFold DB embeddings |
| `AFDB_sequences.fasta` | 671 MB | AlphaFold DB sequences |

### Protein-Vec Model Weights

**TODO**: Document where to obtain Protein-Vec model weights. The embedding script expects:
- `protein_vec_models/protein_vec.ckpt`
- `protein_vec_models/protein_vec_params.json`

These appear to come from the Protein-Vec repository (need to verify source).

---

## File Structure (Current)

```
conformal-protein-retrieval/
├── protein_conformal/           # Main package
│   ├── __init__.py
│   ├── util.py                  # Core algorithms (FDR/FNR, Venn-Abers, FAISS)
│   ├── embed_protein_vec.py     # Protein-Vec embedding
│   ├── scope_utils.py           # SCOPe-specific utilities
│   ├── gradio_app.py            # GUI launcher
│   └── backend/                 # Gradio backend
│       ├── gradio_interface.py
│       ├── collaborative.py
│       └── visualization.py
├── scripts/                     # CLI scripts (to be replaced by `cpr` command)
│   ├── search.py
│   ├── get_probs.py
│   ├── precompute_SVA_probs.py
│   └── ...
├── notebooks/                   # Analysis notebooks
│   ├── pfam/
│   ├── scope/
│   ├── ec/
│   └── ...
├── tests/                       # Test suite
│   ├── conftest.py
│   └── test_util.py
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies (legacy)
├── environment.yml              # Conda environment
└── dockerfile                   # Docker support
```

## File Structure (Planned)

```
conformal-protein-retrieval/
├── protein_conformal/           # Main package
│   ├── __init__.py
│   ├── cli.py                   # NEW: CLI entry point
│   ├── core/                    # NEW: Core algorithms
│   │   ├── conformal.py         # FDR/FNR threshold calculations
│   │   ├── venn_abers.py        # Probability calibration
│   │   └── faiss_ops.py         # FAISS operations
│   ├── embed/                   # NEW: Embedding backends
│   │   ├── protein_vec.py
│   │   └── base.py
│   ├── io/                      # NEW: I/O utilities
│   │   ├── fasta.py
│   │   └── results.py
│   └── gui/                     # Gradio interface (moved from backend/)
├── tests/
├── docs/                        # NEW: Documentation
│   ├── installation.md
│   ├── quickstart.md
│   └── api.md
└── examples/                    # NEW: Example scripts
```

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

## Contributing

1. Create a feature branch from `main`
2. Make changes with tests
3. Ensure all tests pass
4. Submit PR for review

---

## Notes

- **Do not merge to `main`** until all tests pass and numerical outputs are verified
- The original scripts in `scripts/` should continue to work during the transition
- Gradio UI should remain functional throughout refactoring
