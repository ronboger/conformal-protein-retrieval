# Upload Checklist: What Goes Where

This document specifies exactly what files go to GitHub vs Zenodo.

## Summary

| Location | What | Why |
|----------|------|-----|
| **GitHub** | Code, small data (<1MB), configs | Version control, collaboration |
| **Zenodo** | Large data files (>1MB), embeddings | Long-term archival, DOI |
| **User obtains** | Protein-Vec model weights | Large binary, separate distribution |

---

## GitHub Repository (You Commit This)

### Code & Configuration
```
protein_conformal/          # All Python code
├── __init__.py
├── cli.py
├── util.py
├── scope_utils.py
├── embed_protein_vec.py
├── gradio_app.py
└── backend/

scripts/                    # Helper scripts
├── verify_*.py
├── compute_fdr_table.py
├── slurm_*.sh
└── *.py

tests/                      # Test suite
notebooks/                  # Analysis notebooks
docs/                       # Documentation
```

### Small Data Files (<1MB each)
```
data/gene_unknown/
├── unknown_aa_seqs.fasta   # 56 KB - JCVI Syn3.0 sequences
├── unknown_aa_seqs.npy     # 299 KB - Pre-computed embeddings
└── jcvi_syn30_unknown_gene_hits.csv  # 61 KB - Results

results/
├── fdr_thresholds.csv      # ~2 KB - Threshold lookup table
├── fnr_thresholds.csv      # ~7 KB - FNR thresholds
└── sim2prob_lookup.csv     # ~8 KB - Probability lookup
```

### Configuration & Docs
```
pyproject.toml
setup.py
Dockerfile
apptainer.def
README.md
GETTING_STARTED.md
DATA.md
CLAUDE.md
docs/REPRODUCIBILITY.md
.gitignore
```

### Model Code (NOT weights)
```
protein_vec_models/
├── model_protein_moe.py      # Model architecture code
├── utils_search.py           # Embedding utilities
├── data_protein_vec.py       # Data loading code
├── embed_structure_model.py
├── model_protein_vec_single_variable.py
├── train_protein_vec.py
├── __init__.py
└── *.json                    # Config files only
```

---

## Zenodo Repository (You Upload This)

**Zenodo URL**: https://zenodo.org/records/14272215

### Essential Files (Required for paper verification)

| File | Size | Description |
|------|------|-------------|
| `lookup_embeddings.npy` | **1.1 GB** | UniProt database embeddings (540K proteins) |
| `lookup_embeddings_meta_data.tsv` | **535 MB** | Protein metadata (names, Pfam domains, etc.) |
| `pfam_new_proteins.npy` | **2.4 GB** | Calibration data for FDR/probability |

### Optional Files (For extended experiments)

| File | Size | Description |
|------|------|-------------|
| `afdb_embeddings_protein_vec.npy` | 4.7 GB | AlphaFold DB embeddings |
| CLEAN enzyme data | varies | For Tables 1-2 reproduction |
| SCOPe/DALI data | varies | For Tables 4-6 reproduction |

---

## User Must Obtain Separately

### Protein-Vec Model Weights (~3 GB)

These are NOT in GitHub or Zenodo. Users get them by:

1. **Option A**: Contact authors for `protein_vec_models.gz`
2. **Option B**: Use pre-computed embeddings from Zenodo (no weights needed for searching)

Files needed if embedding new sequences:
```
protein_vec_models/
├── protein_vec.ckpt          # 804 MB - Main model
├── protein_vec_params.json   # Config
├── aspect_vec_*.ckpt         # 200-400 MB each - Aspect models
└── tm_vec_swiss_model_large.ckpt  # 391 MB
```

### CLEAN Model Weights (if using --model clean)

Get from: https://github.com/tttianhao/CLEAN

---

## .gitignore Must Include

```gitignore
# Large data files (on Zenodo)
data/*.npy
data/*.tsv
data/*.pkl

# Model weights (user obtains separately)
protein_vec_models/*.ckpt
protein_vec_models.gz

# Build artifacts
*.sif
.apptainer_cache/
logs/
.claude/
```

---

## Verification: Is Everything Set Up Correctly?

Run this after cloning + downloading:

```bash
# Check GitHub files present
ls data/gene_unknown/unknown_aa_seqs.fasta  # Should exist
ls results/fdr_thresholds.csv               # Should exist

# Check Zenodo files downloaded
ls -lh data/lookup_embeddings.npy           # Should be ~1.1 GB
ls -lh data/pfam_new_proteins.npy           # Should be ~2.4 GB

# Check model weights (if embedding)
ls protein_vec_models/protein_vec.ckpt      # Should exist if embedding

# Run verification
cpr verify --check syn30
# Expected: 58-60/149 hits (39.6%)
```

---

## For Repository Maintainers

### When releasing a new version:

1. **GitHub**:
   - Commit all code changes
   - Update `results/fdr_thresholds.csv` with new calibration
   - Tag release: `git tag v1.x.x`

2. **Zenodo**:
   - Upload updated embedding files if changed
   - Create new version linked to GitHub release

### Files to NEVER commit to GitHub:
- Any `.npy` file > 1 MB
- Any `.ckpt` file (model weights)
- Any `.pkl` file > 1 MB
- Any `.tsv` or `.csv` > 1 MB
