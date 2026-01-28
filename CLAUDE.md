# Claude Code Guidelines for CPR

## Bash Guidelines

### IMPORTANT: Avoid commands that cause output buffering issues

- DO NOT pipe output through `head`, `tail`, `less`, or `more` when monitoring or checking command output
- DO NOT use `| head -n X` or `| tail -n X` to truncate output - these cause buffering problems
- Instead, let commands complete fully, or use `-max-lines` flags if the command supports them
- For log monitoring, prefer reading files directly rather than piping through filters

### When checking command output:

- Run commands directly without pipes when possible
- If you need to limit output, use command-specific flags (e.g., `git log -n 10` instead of `git log | head -10`)
- Avoid chained pipes that can cause output to buffer indefinitely

---

## Project-Specific Guidelines

### Paper Reference
- **Title**: "Functional protein mining with conformal guarantees"
- **Journal**: Nature Communications (2025) 16:85
- **DOI**: https://doi.org/10.1038/s41467-024-55676-y
- **Authors**: Ron S. Boger, Seyone Chithrananda, Anastasios N. Angelopoulos, Peter H. Yoon, Michael I. Jordan, Jennifer A. Doudna

### Key Claims to Verify
1. **Figure 2A**: 39.6% of JCVI Syn3.0 genes (59/149) annotated at FDR α=0.1
2. **Tables 1-2**: CLEAN enzyme classification (New-392, Price-149)
3. **Tables 4-6**: DALI prefiltering (82.8% TPR, 31.5% DB reduction)
4. **Figure 2H**: Venn-Abers calibration (|p̂⁰ - p̂¹| ≈ 0)

### Core Algorithms (in `protein_conformal/util.py`)
- `get_thresh_FDR()` / `get_thresh_new_FDR()` - FDR threshold via conformal risk control
- `get_thresh_new()` - FNR threshold calculation
- `simplifed_venn_abers_prediction()` - Calibrated probability assignment
- `scope_hierarchical_loss()` - Hierarchical loss for SCOPe/EC classification
- `load_database()` / `query()` - FAISS operations for similarity search

### Data Files (Zenodo: https://zenodo.org/records/14272215)
- `pfam_new_proteins.npy` (2.5 GB) - Pfam calibration data
- `lookup_embeddings.npy` (1.1 GB) - UniProt embeddings
- `afdb_embeddings_protein_vec.npy` (4.7 GB) - AFDB embeddings

### Protein-Vec Model Weights
- Location: Google Drive (to be added to repo)
- Required files: `protein_vec.ckpt`, `protein_vec_params.json`, `model_protein_moe.py`, `utils_search.py`

---

## Development Log

### 2025-01-28 14:30 PST - Initial Session

**Completed:**
- [x] Merged `origin/gradio-ron` into local (NOT pushed to origin/main)
- [x] Removed duplicate `src/protein_conformal/` directory (2,280 lines)
- [x] Removed `pfam/tmp.py` temp file
- [x] Created `pyproject.toml` with modern packaging and `cpr` CLI entry point
- [x] Created test infrastructure: `tests/conftest.py`, `tests/test_util.py`
- [x] Created documentation: `DEVELOPMENT.md`, `docs/INSTALLATION.md`, `docs/QUICKSTART.md`
- [x] Created `REPO_ORGANIZATION.md` mapping paper figures to code
- [x] Added `docker-compose.yml`
- [x] Read and analyzed the full Nature Communications paper

**Current Branch:** `refactor/cpr-cleanup-and-tests` (4 commits ahead of origin/main)

**Key Findings:**
- CLEAN data file exists and is NOT empty (84 MB)
- `results/fdr_thresholds.csv` is nearly empty (just headers)
- Local main has merge but origin/main is clean - can reset if needed

**Pending Tasks:**
1. Download Zenodo data files
2. Verify JCVI Syn3.0 results (39.6% annotation rate) - HIGHEST PRIORITY
3. Run test suite and fix failures
4. Add Protein-Vec model weights (user will provide)
5. Create CLI entry point

**Questions for User:**
- Protein-Vec weights location? → User will add to folder
- Zenodo download? → User asked Claude to download

---

### Session Notes Template

```
### YYYY-MM-DD HH:MM TZ - Session Description

**Completed:**
- [ ] Task 1
- [ ] Task 2

**Blocked By:**
- Issue 1

**Next Steps:**
- Step 1
- Step 2
```

---

## Best Practices for This Codebase

### Testing
- Always run `pytest tests/ -v` before committing
- Add regression tests for paper-critical numbers
- Use fixtures from `tests/conftest.py` for consistent test data

### Git Workflow
- Work on feature branches, NOT main
- Don't push to main until results are verified
- Use descriptive commit messages referencing paper figures/tables

### Code Style
- Follow existing patterns in `protein_conformal/util.py`
- Use numpy for numerical operations
- Use FAISS for similarity search (not sklearn)

### Notebooks
- Notebooks are for analysis/visualization, not core logic
- Core algorithms should be in `protein_conformal/`
- Notebooks should import from the package, not duplicate code

### Documentation
- Update `REPO_ORGANIZATION.md` when adding new notebooks
- Keep this log updated with timestamped entries
- Document any deviations from paper methods
