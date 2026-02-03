# Getting Started with CPR

This guide will get you from zero to running protein searches with conformal guarantees in under 10 minutes.

## TL;DR (Easiest Path)

```bash
# 1. Clone and install
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval
pip install -e .

# 2. Download data (4GB total) from https://zenodo.org/records/14272215:
#    → lookup_embeddings.npy (1.1 GB) → data/
#    → lookup_embeddings_meta_data.tsv (535 MB) → data/
#    → pfam_new_proteins.npy (2.4 GB) → data/

# 3. Get Protein-Vec model weights (contact authors or see below)
#    → Extract protein_vec_models.gz to protein_vec_models/

# 4. Run search on your sequences (ONE COMMAND!)
cpr find --input your_sequences.fasta --output results.csv --fdr 0.1

# That's it! results.csv contains:
#   - Functional annotations for each protein
#   - Calibrated probabilities
#   - Uncertainty estimates
```

### Don't have model weights? Use pre-computed embeddings:

```bash
# If you already have embeddings (.npy), skip to search:
cpr search --query your_embeddings.npy --output results.csv --fdr 0.1
```

---

## What You Need

### Already Included (GitHub clone)

When you clone the repository, you automatically get:

| File | Size | Description |
|------|------|-------------|
| `data/gene_unknown/unknown_aa_seqs.fasta` | 56 KB | JCVI Syn3.0 test sequences (149 proteins) |
| `data/gene_unknown/unknown_aa_seqs.npy` | 299 KB | Pre-computed embeddings for test sequences |
| `results/fdr_thresholds.csv` | ~2 KB | FDR thresholds at standard alpha levels |
| `protein_conformal/*.py` | ~100 KB | All the code |

### Download from Zenodo (Required)

Download these from **https://zenodo.org/records/14272215**:

| File | Size | What it is | Where to put it |
|------|------|------------|-----------------|
| `lookup_embeddings.npy` | **1.1 GB** | UniProt database (540K protein embeddings) | `data/` |
| `lookup_embeddings_meta_data.tsv` | **535 MB** | Protein metadata (names, Pfam domains, etc.) | `data/` |
| `pfam_new_proteins.npy` | **2.4 GB** | Calibration data for FDR/probability computation | `data/` |

**Total download: ~4 GB**

### Optional Downloads

| File | Size | When you need it |
|------|------|------------------|
| `afdb_embeddings_protein_vec.npy` | 4.7 GB | Searching AlphaFold Database |
| Protein-Vec model weights | 3 GB | Computing new embeddings from FASTA |
| CLEAN model weights | 1 GB | Enzyme classification with CLEAN |

---

## Step-by-Step Setup

### Step 1: Clone and Install

```bash
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval
pip install -e .
```

**Verify installation:**
```bash
cpr --help
# Should show: embed, search, prob, calibrate, verify commands
```

### Step 2: Download Data

Go to **https://zenodo.org/records/14272215** and download:

1. `lookup_embeddings.npy` (1.1 GB)
2. `lookup_embeddings_meta_data.tsv` (535 MB)
3. `pfam_new_proteins.npy` (2.4 GB)

Move them to the `data/` directory:

```bash
mv ~/Downloads/lookup_embeddings.npy data/
mv ~/Downloads/lookup_embeddings_meta_data.tsv data/
mv ~/Downloads/pfam_new_proteins.npy data/
```

**Verify files:**
```bash
ls -lh data/*.npy data/*.tsv
# lookup_embeddings.npy         1.1G
# lookup_embeddings_meta_data.tsv   535M
# pfam_new_proteins.npy         2.4G
```

### Step 3: Verify Setup

```bash
cpr verify --check syn30
```

**Expected output:**
```
JCVI Syn3.0 Annotation Verification
Total queries:     149
Confident hits:    59    (might be 58-60, see docs/REPRODUCIBILITY.md)
Hit rate:          39.6%
FDR threshold:     λ = 0.999980225003
✓ VERIFICATION PASSED
```

---

## Your First Search

### Easiest: One Command from FASTA (Recommended)

```bash
cpr find --input your_proteins.fasta --output results.csv --fdr 0.1
```

This single command:
1. Embeds your sequences using Protein-Vec
2. Searches the UniProt database (540K proteins)
3. Filters to confident hits at 10% FDR
4. Adds calibrated probability estimates
5. Includes Pfam/functional annotations

**Output columns:**
- `query_name`: Your sequence ID from FASTA
- `similarity`: Cosine similarity score
- `probability`: Calibrated probability of functional match
- `uncertainty`: Venn-Abers uncertainty interval
- `match_*`: Pfam domains, protein names, etc.

### Control FDR Level

```bash
# Stringent: 1% FDR (fewer but more confident hits)
cpr find --input proteins.fasta --output results.csv --fdr 0.01

# Default: 10% FDR (balanced)
cpr find --input proteins.fasta --output results.csv --fdr 0.1

# Discovery: 20% FDR (more hits, some false positives)
cpr find --input proteins.fasta --output results.csv --fdr 0.2
```

### Alternative: Manual Workflow (Advanced)

If you need more control or already have embeddings:

```bash
# Step 1: Embed (if starting from FASTA)
cpr embed --input seqs.fasta --output embeddings.npy --model protein-vec

# Step 2: Search with FDR control
cpr search --query embeddings.npy --output hits.csv --fdr 0.1

# Step 3: Add probabilities (optional, for detailed analysis)
cpr prob --input hits.csv --output hits_with_probs.csv
```

---

## FDR Threshold Reference

Use these thresholds for your desired false discovery rate:

| FDR Level | Threshold (λ) | Use Case |
|-----------|---------------|----------|
| 1% | 0.999990 | Very stringent |
| 5% | 0.999985 | Stringent |
| **10%** | **0.999980** | **Paper default** |
| 15% | 0.999975 | Relaxed |
| 20% | 0.999970 | Discovery-focused |

Full table in `results/fdr_thresholds.csv`.

---

## Model Weights

### Protein-Vec (General Protein Search)

**Option 1: Contact authors** for the `protein_vec_models.gz` archive.

**Option 2: Use pre-computed embeddings** from Zenodo (no weights needed for searching).

If you have the weights:
```bash
tar -xzf protein_vec_models.gz
# Creates protein_vec_models/ directory with:
#   protein_vec.ckpt (804 MB)
#   protein_vec_params.json
#   aspect_vec_*.ckpt (200-400 MB each)
```

### CLEAN (Enzyme Classification)

For enzyme-specific searches, get CLEAN from: https://github.com/tttianhao/CLEAN

---

## Directory Structure After Setup

```
conformal-protein-retrieval/
├── data/
│   ├── lookup_embeddings.npy          ← Download from Zenodo (1.1 GB)
│   ├── lookup_embeddings_meta_data.tsv ← Download from Zenodo (535 MB)
│   ├── pfam_new_proteins.npy          ← Download from Zenodo (2.4 GB)
│   └── gene_unknown/                  ← Included in GitHub
│       ├── unknown_aa_seqs.fasta
│       └── unknown_aa_seqs.npy
├── protein_vec_models/                ← Optional: for new embeddings
│   ├── protein_vec.ckpt
│   └── ...
├── protein_conformal/                 ← Code (included)
├── results/                           ← Your outputs go here
└── scripts/                           ← Helper scripts
```

---

## Troubleshooting

### "FileNotFoundError: data/lookup_embeddings.npy"
→ Download from Zenodo: https://zenodo.org/records/14272215

### "ModuleNotFoundError: No module named 'faiss'"
→ Install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU)

### "Got 58 hits, expected 59"
→ This is expected! See `docs/REPRODUCIBILITY.md` - the result varies by ±1 due to threshold boundary effects.

### "CUDA out of memory"
→ Use CPU: `--device cpu` or reduce batch size with `--batch-size 16`

---

## What's Next?

- **Read the paper**: [Nature Communications (2025) 16:85](https://doi.org/10.1038/s41467-024-55676-y)
- **Explore notebooks**: `notebooks/pfam/genes_unknown.ipynb` shows the full Syn3.0 analysis
- **Run verification**: `cpr verify --check all` tests all paper claims
- **Get help**: Open an issue at https://github.com/ronboger/conformal-protein-retrieval/issues

---

## Summary: Files Checklist

| Source | Files | Size | Status |
|--------|-------|------|--------|
| **GitHub** | Code, test data, thresholds | ~1 MB | ✓ Included |
| **Zenodo** | lookup_embeddings.npy | 1.1 GB | ☐ Download |
| **Zenodo** | lookup_embeddings_meta_data.tsv | 535 MB | ☐ Download |
| **Zenodo** | pfam_new_proteins.npy | 2.4 GB | ☐ Download |
| **Optional** | protein_vec_models/ | 3 GB | ☐ For new embeddings |
| **Optional** | afdb_embeddings_protein_vec.npy | 4.7 GB | ☐ For AFDB search |
