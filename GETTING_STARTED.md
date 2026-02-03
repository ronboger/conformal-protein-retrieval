# Getting Started with CPR

This guide will get you from zero to running protein searches with conformal guarantees.

## Statistical Guarantees

CPR provides rigorous statistical guarantees based on conformal prediction:

| Guarantee | Meaning | How to Use |
|-----------|---------|------------|
| **Expected Marginal FDR ≤ α** | On average, at most α fraction of your hits are false positives | Use `--fdr 0.1` for 10% expected FDR |
| **FNR Control** | Controls the expected fraction of true matches you miss | Use `--fnr 0.1` to miss ≤10% of true hits |
| **Calibrated Probabilities** | Venn-Abers calibration provides valid probability estimates | Output includes `probability` column |

**Key insight**: Unlike p-values or arbitrary thresholds, our FDR guarantees are *marginal* guarantees that hold across all queries in expectation. See the [paper](https://doi.org/10.1038/s41467-024-55676-y) for theoretical details.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval
pip install -e .

# 2. Download required data (see wget commands below)

# 3. Search with your sequences (FASTA or embeddings)
cpr search --input your_sequences.fasta --output results.csv --fdr 0.1
```

---

## What You Need

### Already Included (GitHub clone)

| File | Size | Description |
|------|------|-------------|
| `data/gene_unknown/unknown_aa_seqs.fasta` | 56 KB | JCVI Syn3.0 test sequences (149 proteins) |
| `data/gene_unknown/unknown_aa_seqs.npy` | 299 KB | Pre-computed embeddings for test sequences |
| `results/fdr_thresholds.csv` | ~2 KB | FDR thresholds at standard alpha levels |
| `protein_conformal/*.py` | ~100 KB | All the code |

### Download from Zenodo (Required)

**Zenodo URL**: https://zenodo.org/records/14272215

```bash
# Download all required files with wget
cd data/

# Database embeddings (1.1 GB) - 540K UniProt protein embeddings
wget "https://zenodo.org/records/14272215/files/lookup_embeddings.npy?download=1" -O lookup_embeddings.npy

# Database metadata (535 MB) - protein names, Pfam domains, etc.
wget "https://zenodo.org/records/14272215/files/lookup_embeddings_meta_data.tsv?download=1" -O lookup_embeddings_meta_data.tsv

# Calibration data (2.4 GB) - Pfam data for FDR/probability computation
wget "https://zenodo.org/records/14272215/files/pfam_new_proteins.npy?download=1" -O pfam_new_proteins.npy

# Verify downloads
ls -lh lookup_embeddings.npy lookup_embeddings_meta_data.tsv pfam_new_proteins.npy
# Expected: 1.1G, 535M, 2.4G
```

Or with curl:
```bash
cd data/
curl -L -o lookup_embeddings.npy "https://zenodo.org/records/14272215/files/lookup_embeddings.npy?download=1"
curl -L -o lookup_embeddings_meta_data.tsv "https://zenodo.org/records/14272215/files/lookup_embeddings_meta_data.tsv?download=1"
curl -L -o pfam_new_proteins.npy "https://zenodo.org/records/14272215/files/pfam_new_proteins.npy?download=1"
```

### Optional Downloads

| File | Size | When you need it |
|------|------|------------------|
| `afdb_embeddings_protein_vec.npy` | 4.7 GB | Searching AlphaFold Database |
| Protein-Vec model weights | 3 GB | Computing new embeddings from FASTA |
| CLEAN model weights | 1 GB | Enzyme classification with CLEAN |

---

## CLI Commands

### `cpr search` - Search with Conformal Guarantees

The main command for protein search. Accepts both FASTA files and pre-computed embeddings:

```bash
# From FASTA (embeds automatically using Protein-Vec)
cpr search --input proteins.fasta --output results.csv --fdr 0.1

# From pre-computed embeddings
cpr search --input embeddings.npy --output results.csv --fdr 0.1
```

When given a FASTA file, `cpr search` will:
1. Embed your sequences using Protein-Vec (or CLEAN with `--model clean`)
2. Search the UniProt database (540K proteins)
3. Filter to confident hits at your specified FDR
4. Add calibrated probability estimates
5. Include Pfam/functional annotations

**More examples:**

```bash
# With FNR control instead (control false negatives)
cpr search --input proteins.fasta --output results.csv --fnr 0.1

# With a specific threshold you've computed
cpr search --input proteins.fasta --output results.csv --threshold 0.999980

# Use CLEAN model for enzyme classification
cpr search --input enzymes.fasta --output results.csv --model clean --fdr 0.1

# Exploratory: get all neighbors without filtering
cpr search --input proteins.fasta --output results.csv --no-filter
```

**Threshold options** (mutually exclusive):
- `--fdr ALPHA`: Look up threshold for target FDR level (e.g., `--fdr 0.1` for 10% FDR)
- `--fnr ALPHA`: Look up threshold for target FNR level
- `--threshold VALUE`: Use a specific similarity threshold you provide
- `--no-filter`: Return all k nearest neighbors without filtering

### `cpr embed` - Generate Embeddings

Convert FASTA sequences to embeddings:

```bash
# Using Protein-Vec (default, general-purpose)
cpr embed --input proteins.fasta --output embeddings.npy --model protein-vec

# Using CLEAN (enzyme-specific)
cpr embed --input enzymes.fasta --output embeddings.npy --model clean
```

### `cpr verify` - Verify Paper Results

```bash
cpr verify --check syn30    # Verify JCVI Syn3.0 result (39.6% annotation)
cpr verify --check all      # Run all verification checks
```

---

## FDR/FNR Threshold Reference

These thresholds control the trade-off between hits and false positives:

| α Level | FDR Threshold (λ) | FNR Threshold (λ) | Use Case |
|---------|-------------------|-------------------|----------|
| 0.001 | ~0.999995 | ~0.99979 | Ultra-stringent |
| 0.01 | ~0.999992 | ~0.99984 | Very stringent |
| 0.05 | ~0.999987 | ~0.99988 | Stringent |
| **0.1** | **0.999980** | **~0.99990** | **Paper default** |
| 0.15 | ~0.999975 | ~0.99991 | Relaxed |
| 0.2 | ~0.999970 | ~0.99992 | Discovery-focused |

Full computed tables in `results/fdr_thresholds.csv` and `results/fnr_thresholds.csv`.

---

## CLEAN Enzyme Classification

For enzyme-specific searches with EC number predictions:

### Setup

```bash
# 1. Clone CLEAN repository
git clone https://github.com/tttianhao/CLEAN.git CLEAN_repo
cd CLEAN_repo && pip install -e . && cd ..

# 2. Install ESM dependency
pip install fair-esm

# 3. Download CLEAN weights (if not included)
# Weights should be at: CLEAN_repo/app/data/pretrained/CLEAN_pretrained/
```

### Usage with CPR

```bash
# Generate CLEAN embeddings (128-dim)
cpr embed --input enzymes.fasta --output clean_embeddings.npy --model clean

# Search with CLEAN
cpr search --input enzymes.fasta --output enzyme_results.csv --model clean --fdr 0.1
```

### Verify CLEAN Results (Paper Tables 1-2)

```bash
# Run CLEAN verification script
python scripts/verify_clean.py

# Expected output:
# Mean test loss: 0.97 ± 0.XX
# ✓ VERIFICATION PASSED - Risk controlled at α=1.0
```

---

## DALI Structural Prefiltering

For structural homology search (DALI + AFDB), we use z-score thresholds:

| Metric | Value | Description |
|--------|-------|-------------|
| **elbow_z** | **~5.1** | Z-score threshold for prefiltering |
| TPR | 81.8% | True Positive Rate at elbow threshold |
| FNR | 18.2% | False Negative Rate (miss rate) |
| DB Reduction | 31.5% | Fraction of database filtered out |

Pre-computed results in `results/dali_thresholds.csv` (73 trials from paper experiments).

**Usage**: When running DALI, filter candidates with z-score ≥ 5.1 to achieve ~82% TPR while reducing database size by ~31%.

---

## Legacy Scripts

These scripts from the original paper analysis can be used for advanced workflows:

### FDR/FNR Threshold Computation

```bash
# Compute FDR thresholds at custom alpha levels
python scripts/compute_fdr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/my_fdr_thresholds.csv \
    --n-trials 100 \
    --alpha-levels 0.01,0.05,0.1,0.2

# Compute FNR thresholds
python scripts/compute_fnr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/my_fnr_thresholds.csv \
    --n-trials 100

# Use partial matches (at least one Pfam domain matches)
python scripts/compute_fdr_table.py --partial ...
```

### Verification Scripts

```bash
# Verify JCVI Syn3.0 annotation (Paper Figure 2A)
python scripts/verify_syn30.py

# Verify DALI prefiltering (Paper Tables 4-6)
python scripts/verify_dali.py

# Verify CLEAN enzyme classification (Paper Tables 1-2)
python scripts/verify_clean.py

# Verify FDR algorithm correctness
python scripts/verify_fdr_algorithm.py
```

### Probability Computation

```bash
# Precompute SVA probabilities for a database
python scripts/precompute_SVA_probs.py \
    --calibration data/pfam_new_proteins.npy \
    --output data/sva_probabilities.csv

# Get probabilities for search results
python scripts/get_probs.py \
    --input results.csv \
    --calibration data/pfam_new_proteins.npy \
    --output results_with_probs.csv
```

### Original Paper Scripts (in `scripts/pfam/`)

```bash
# Original FDR threshold generation (paper methodology)
python scripts/pfam/generate_fdr.py

# Original FNR threshold generation
python scripts/pfam/generate_fnr.py

# SVA reliability analysis
python scripts/pfam/sva_results.py
```

---

## Model Weights

### Protein-Vec (General Protein Search)

**Option 1: Contact authors** for the `protein_vec_models.gz` archive.

**Option 2: Use pre-computed embeddings** from Zenodo (no weights needed).

If you have the weights:
```bash
tar -xzf protein_vec_models.gz
# Creates protein_vec_models/ with:
#   protein_vec.ckpt (804 MB)
#   protein_vec_params.json
#   aspect_vec_*.ckpt (200-400 MB each)
```

---

## Troubleshooting

### "FileNotFoundError: data/lookup_embeddings.npy"
→ Download from Zenodo (see wget commands above)

### "ModuleNotFoundError: No module named 'faiss'"
→ Install FAISS: `pip install faiss-cpu` (or `conda install faiss-gpu` for GPU)

### "Got 58 hits, expected 59"
→ This is expected! See `docs/REPRODUCIBILITY.md` - varies by ±1 due to threshold boundary effects.

### "CUDA out of memory"
→ Use CPU: `--cpu` flag or reduce batch size

### "ModuleNotFoundError: No module named 'fair_esm'"
→ For CLEAN embeddings: `pip install fair-esm`

---

## Output Columns

Search results include:

| Column | Description |
|--------|-------------|
| `query_name` | Your sequence ID from FASTA |
| `similarity` | Cosine similarity score |
| `probability` | Calibrated probability of functional match |
| `uncertainty` | Venn-Abers uncertainty interval |
| `match_name` | Matched protein name |
| `match_pfam` | Pfam domain annotations |

---

## What's Next?

- **Read the paper**: [Nature Communications (2025) 16:85](https://doi.org/10.1038/s41467-024-55676-y)
- **Explore notebooks**: `notebooks/pfam/genes_unknown.ipynb` shows the full Syn3.0 analysis
- **Run verification**: `cpr verify --check all` tests all paper claims
- **Get help**: Open an issue at https://github.com/ronboger/conformal-protein-retrieval/issues

---

## Files Checklist

| Source | Files | Size | Status |
|--------|-------|------|--------|
| **GitHub** | Code, test data, thresholds | ~1 MB | ✓ Included |
| **Zenodo** | lookup_embeddings.npy | 1.1 GB | ☐ Download |
| **Zenodo** | lookup_embeddings_meta_data.tsv | 535 MB | ☐ Download |
| **Zenodo** | pfam_new_proteins.npy | 2.4 GB | ☐ Download |
| **Optional** | protein_vec_models/ | 3 GB | ☐ For new embeddings |
| **Optional** | afdb_embeddings_protein_vec.npy | 4.7 GB | ☐ For AFDB search |
