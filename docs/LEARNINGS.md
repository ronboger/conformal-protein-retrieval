# CPR Project Learnings

This document captures key learnings, gotchas, and patterns discovered while working on the Conformal Protein Retrieval project.

---

## Git & Branch Management

### Branch Structure
- **`main`** - Your fork's main branch
- **`upstream/main`** - Ron Boger's original repo (https://github.com/ronboger/conformal-protein-retrieval)
- **`gradio`** - Active development branch for the Gradio web interface
- **`huggingface`** - Remote for HuggingFace Spaces deployment

### Syncing with Upstream
```bash
git fetch upstream
git log gradio..upstream/main --oneline  # Show commits in upstream NOT in gradio
git merge upstream/main                   # Merge upstream into current branch
```

---

## Data & Thresholds

### Critical: Data Leakage Warning
**DO NOT USE** `conformal_pfam_with_lookup_dataset.npy` from backup directories.
- First 50 samples all have the same Pfam family "PF01266;"
- Positive rate: 3.00% (vs 0.22% correct)
- Produces WRONG FDR threshold

**USE**: `pfam_new_proteins.npy` from Zenodo
- 1,864 diverse samples
- 0.22% positive rate
- Matches paper threshold: 0.9999802

### Threshold Files
| File | Purpose |
|------|---------|
| `results/fdr_thresholds.csv` | FDR (exact match) thresholds |
| `results/fnr_thresholds.csv` | FNR (exact match) thresholds |
| `results/fnr_thresholds_partial.csv` | FNR (partial match) thresholds |
| `results/calibration_probs.csv` | Venn-Abers probability calibration data |

### Verified Paper Claims
| Claim | Paper Value | Verified Value |
|-------|-------------|----------------|
| Syn3.0 annotation (alpha=0.1) | 39.6% (59/149) | 39.6% (59/149) |
| FDR threshold (alpha=0.1) | 0.9999802250 | 0.9999801 |
| DALI TPR | 82.8% | 81.8% |
| DALI DB reduction | 31.5% | 31.5% |
| CLEAN loss <= alpha | 1.0 | 0.97 |

---

## Code Patterns

### FDR vs FNR Thresholds
- **FDR** (False Discovery Rate): Higher threshold = stricter = fewer but more confident results
- **FNR** (False Negative Rate): Lower threshold = more permissive = more results, fewer misses
- For FNR: lower alpha -> lower threshold (opposite intuition!)

### Array Dimension Bug (Fixed)
The `get_thresh_FDR()` function failed on 1D arrays. Fixed by checking array dimensions:
```python
is_1d = len(labels.shape) == 1
if is_1d:
    # Use risk_1d function
else:
    # Use standard risk function
```

### Gradio File Uploads
Gradio may pass file objects in different formats:
- File-like objects with `.read()`
- Temp files with `.name` attribute
- Plain filesystem paths (when type='filepath')
- Dicts with 'path'/'name' metadata

Handle all cases with fallback logic (see `_persist_uploaded_file()` in gradio_interface.py).

---

## Environment & Dependencies

### Python Environment
- Conda environment: `conformal-s` (Python 3.11.10)
- Key packages: faiss 1.9.0, torch 2.5.0, numpy 1.26.4

### Missing Dependencies (Not in requirements.txt)
- `pytorch-lightning` - for Protein-Vec model loading
- `h5py` - for utils_search.py
- `gradio` - for web interface

### NumPy Compatibility
NumPy 1.22+ renamed `interpolation=` to `method=` in `np.quantile()`. Use `method=` for compatibility.

---

## Gradio Interface

### Current UI Features (as of 2026-02)
1. **FASTA input**: Text area + file upload
2. **Risk control**: FDR/FNR toggle with alpha slider
3. **Match type**: Exact vs Partial Pfam matching
4. **Database selection**: UniProt, SCOPE, or Custom upload
5. **Results**: Sortable table with export (CSV/JSON)
6. **Probability calibration**: Uses pre-computed Venn-Abers data

### HuggingFace Deployment
- Set `HF_DATASET_ID` environment variable for automatic data download
- Uses `huggingface_hub.hf_hub_download()` for large files
- Files are cached locally after first download

### Performance Optimizations
- `LOOKUP_RESOURCE_CACHE`: Caches FAISS index + metadata by file path + mtime
- `@lru_cache`: Caches threshold CSV parsing
- `StageTimer`: Logs timing for each pipeline stage

---

## Common Issues & Solutions

### Issue: "No module named 'protein_conformal'"
**Solution**: Install in development mode: `pip install -e .`

### Issue: Gradio import fails
**Solution**: Made gradio import optional in `__init__.py` with try/except

### Issue: FDR threshold doesn't match paper
**Solution**: Check if using correct calibration data (pfam_new_proteins.npy, NOT backup files)

### Issue: NumPy deprecation warning for quantile
**Solution**: Use `method='lower'` instead of `interpolation='lower'`

### Issue: setup.py references non-existent src/ directory
**Solution**: Simplified to defer to pyproject.toml

---

## Testing

### Run Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_util.py -v        # Just util tests
pytest tests/test_cli.py -v         # Just CLI tests
pytest tests/ --cov=protein_conformal --cov-report=html  # With coverage
```

### Test Count
- 27 util tests
- 24 CLI tests
- All passing as of 2026-02-03

---

## Files to Know

### Core Algorithm
- `protein_conformal/util.py` - All conformal prediction algorithms

### CLI
- `protein_conformal/cli.py` - `cpr embed`, `cpr search`, `cpr verify`

### Gradio
- `protein_conformal/gradio_app.py` - Entry point
- `protein_conformal/backend/gradio_interface.py` - Main UI logic

### Threshold Computation
- `scripts/compute_fdr_table.py` - FDR thresholds
- `scripts/compute_fnr_table.py` - FNR thresholds

### Verification
- `scripts/verify_syn30.py` - JCVI Syn3.0 (Figure 2A)
- `scripts/verify_dali.py` - DALI prefiltering
- `scripts/verify_clean.py` - CLEAN enzyme

---

## HuggingFace Spaces Deployment

### Key Lesson: Optional Imports
When deploying to HuggingFace Spaces, **wrap optional module imports in try/except**. The Space only installs what's in `requirements.txt`, so unused modules with extra dependencies will crash the app on import.

```python
# Bad - crashes if py3Dmol not installed
from .visualization import create_structure_with_heatmap

# Good - gracefully handles missing deps
try:
    from .visualization import create_structure_with_heatmap
except ImportError:
    create_structure_with_heatmap = None
```

### Requirements.txt Best Practices
1. Include ALL imports used by the main app path
2. Comment out optional deps with clear notes
3. Test locally with a fresh venv before pushing

### Dataset Integration
- Set `HF_DATASET_ID` env variable in Space settings
- Dataset structure must match paths in `app.py` `ensure_assets()`
- Files downloaded on first run, then cached

---

## Session History

### 2026-02-05 (Gradio branch)
- Confirmed gradio branch is synced with upstream/main
- No remaining changes to integrate
- Last 3 commits were Gradio UI improvements

### 2026-02-03
- Archived 16 redundant scripts
- Consolidated threshold CSVs
- Added full tables to GETTING_STARTED.md

### 2026-02-02
- Verified Syn3.0: 59/149 = 39.6%
- Fixed FDR bug (1D/2D array handling)
- Created CLI with embed, search, verify commands

### 2026-01-28
- Initial cleanup
- Removed duplicate src/protein_conformal/
- Created pyproject.toml and test infrastructure
