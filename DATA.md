# Data Requirements

This document describes the data files needed to run CPR (Conformal Protein Retrieval) and reproduce the paper results.

## Quick Start

```bash
# 1. Download required data files
cd data/
wget "https://zenodo.org/records/14272215/files/lookup_embeddings.npy?download=1" -O lookup_embeddings.npy
wget "https://zenodo.org/records/14272215/files/lookup_embeddings_meta_data.tsv?download=1" -O lookup_embeddings_meta_data.tsv
wget "https://zenodo.org/records/14272215/files/pfam_new_proteins.npy?download=1" -O pfam_new_proteins.npy
cd ..

# 2. Download and extract Protein-Vec model weights (for embedding new sequences)
wget "https://zenodo.org/records/18478696/files/protein_vec_models.gz?download=1" -O protein_vec_models.gz
tar -xzf protein_vec_models.gz

# 3. Verify setup
cpr verify --check syn30
```

## Data Sources

### Zenodo (https://zenodo.org/records/14272215)

Large data files that should NOT be committed to git:

| File | Size | Description | Location |
|------|------|-------------|----------|
| `lookup_embeddings.npy` | 1.1 GB | UniProt protein embeddings (540K proteins) | `data/` |
| `pfam_new_proteins.npy` | 2.4 GB | Pfam calibration data | `data/` |
| `lookup_embeddings_meta_data.tsv` | 535 MB | UniProt metadata (Pfam, protein names, etc.) | `data/` |

### GitHub Repository

Small files that ARE committed to git:

| File | Size | Description |
|------|------|-------------|
| `data/gene_unknown/unknown_aa_seqs.fasta` | 56 KB | JCVI Syn3.0 unknown gene sequences |
| `data/gene_unknown/unknown_aa_seqs.npy` | 299 KB | Pre-computed embeddings for Syn3.0 genes |
| `data/gene_unknown/jcvi_syn30_unknown_gene_hits.csv` | 61 KB | Results: 59 annotated genes |

### Protein-Vec Models ([Zenodo #18478696](https://zenodo.org/records/18478696))

Model weights (2.9 GB compressed):

```bash
wget "https://zenodo.org/records/18478696/files/protein_vec_models.gz?download=1" -O protein_vec_models.gz
tar -xzf protein_vec_models.gz
```

| File | Size | Required For |
|------|------|--------------|
| `protein_vec.ckpt` | 804 MB | Core embedding model |
| `protein_vec_params.json` | 240 B | Model configuration |
| `aspect_vec_*.ckpt` | ~200-400 MB each | Aspect-specific models |
| `tm_vec_swiss_model_large.ckpt` | 391 MB | TM-Vec model |

## Directory Structure

```
conformal-protein-retrieval/
├── data/
│   ├── lookup_embeddings.npy          # [Zenodo] UniProt embeddings
│   ├── lookup_embeddings_meta_data.tsv # [Zenodo] UniProt metadata
│   ├── pfam_new_proteins.npy          # [Zenodo] Calibration data
│   ├── gene_unknown/
│   │   ├── unknown_aa_seqs.fasta      # [GitHub] Syn3.0 sequences
│   │   ├── unknown_aa_seqs.npy        # [GitHub] Syn3.0 embeddings
│   │   └── jcvi_syn30_unknown_gene_hits.csv  # [GitHub] Results
│   └── ec/                            # CLEAN enzyme data
├── protein_vec_models/                # [Archive] Model weights
│   ├── protein_vec.ckpt
│   ├── protein_vec_params.json
│   ├── model_protein_moe.py           # Model code
│   ├── utils_search.py                # Embedding utilities
│   └── ...
└── results/                           # Output directory
```

## Reproducing Paper Results

### Figure 2A: JCVI Syn3.0 Annotation (39.6%)

**Required files:**
- `data/gene_unknown/unknown_aa_seqs.npy`
- `data/lookup_embeddings.npy`
- `data/lookup_embeddings_meta_data.tsv`
- `data/pfam_new_proteins.npy`

**Run:**
```bash
cpr verify --check syn30
# Expected: 59/149 = 39.6% hits at FDR α=0.1
```

### Tables 1-2: CLEAN Enzyme Classification

**Required files:**
- `clean_selection/clean_new_v_ec_cluster.npy`
- Additional CLEAN data from Zenodo

### Tables 4-6: DALI Prefiltering

**Required files:**
- SCOPe domain data
- DALI Z-scores
- AFDB embeddings

## What to Add to Zenodo

If you're updating Zenodo, include:

1. **Essential (required for paper verification):**
   - `lookup_embeddings.npy`
   - `lookup_embeddings_meta_data.tsv`
   - `pfam_new_proteins.npy`

2. **Optional (for full experiments):**
   - `afdb_embeddings_protein_vec.npy` (4.7 GB) - AlphaFold DB embeddings
   - CLEAN embeddings
   - SCOPe/DALI data

## What to Add to GitHub

Keep in GitHub (small files):
- `data/gene_unknown/*.fasta` - Query sequences
- `data/gene_unknown/*.npy` - Pre-computed query embeddings (< 1 MB)
- `results/*.csv` - Result summaries
- `protein_vec_models/*.py` - Model code (NOT weights)
- `protein_vec_models/*.json` - Model configs

Add to `.gitignore` (large files):
```
*.ckpt
data/*.npy
data/*.tsv
protein_vec_models.gz
```

## Verification Checklist

After setting up data, verify with:

```bash
# Check file sizes
ls -lh data/*.npy

# Expected:
# lookup_embeddings.npy      ~1.1 GB
# pfam_new_proteins.npy      ~2.4 GB

# Run verification
cpr verify --check fdr    # Tests algorithm
cpr verify --check syn30  # Tests paper result (39.6%)
```
