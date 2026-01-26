# Quick Start Guide

This guide shows how to use Conformal Protein Retrieval for common tasks.

## Overview

CPR provides **statistically rigorous protein search** with two key guarantees:

1. **False Discovery Rate (FDR) Control**: Limit the fraction of incorrect matches among your results
2. **False Negative Rate (FNR) Control**: Limit the fraction of true matches you miss

Additionally, CPR provides **calibrated probabilities** for each hit using Venn-Abers prediction.

---

## Basic Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│   Embed     │────▶│   Search    │────▶│   Results   │
│   FASTA     │     │  (Protein-  │     │  (FAISS +   │     │   + Probs   │
│             │     │   Vec)      │     │  Conformal) │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Example 1: Search with FDR Control

Find protein homologs while controlling the false discovery rate at 10%.

### Step 1: Embed your query proteins

```bash
python protein_conformal/embed_protein_vec.py \
    --input_file my_proteins.fasta \
    --output_file my_proteins_embeddings.npy \
    --path_to_protein_vec protein_vec_models
```

### Step 2: Search with FDR control

```bash
python scripts/search.py \
    --query_embedding my_proteins_embeddings.npy \
    --query_fasta my_proteins.fasta \
    --lookup_embedding data/lookup_embeddings.npy \
    --lookup_fasta data/lookup_embeddings_meta_data.tsv \
    --fdr \
    --fdr_lambda 0.99996425 \
    --output results.csv
```

The `--fdr_lambda` value is a pre-computed threshold that ensures FDR ≤ 10% for exact Pfam matches.

### Step 3: Get calibrated probabilities

```bash
python scripts/get_probs.py \
    --precomputed \
    --precomputed_path data/pfam_sims_to_probs.csv \
    --input results.csv \
    --output results_with_probs.csv \
    --partial
```

---

## Example 2: Search with FNR Control

Find protein homologs while ensuring you don't miss more than 10% of true matches.

```bash
python scripts/search.py \
    --query_embedding my_proteins_embeddings.npy \
    --query_fasta my_proteins.fasta \
    --lookup_embedding data/lookup_embeddings.npy \
    --lookup_fasta data/lookup_embeddings_meta_data.tsv \
    --fnr \
    --fnr_lambda 0.99974871 \
    --output results_fnr.csv
```

---

## Example 3: Using the Gradio GUI

Launch the web interface for interactive exploration:

```bash
python -m protein_conformal.gradio_app --port 7860
```

Then open http://localhost:7860 in your browser.

Features:
- Paste sequences or upload FASTA files
- Choose FDR or FNR control
- Visualize results with 3D structures
- Export results to CSV

---

## Example 4: Programmatic Use (Python)

```python
import numpy as np
from protein_conformal.util import (
    load_database,
    query,
    read_fasta,
    get_thresh_FDR,
    risk,
    simplifed_venn_abers_prediction
)

# Load your query embeddings
query_embeddings = np.load('my_proteins_embeddings.npy')

# Load the lookup database
lookup_embeddings = np.load('data/lookup_embeddings.npy')
index = load_database(lookup_embeddings)

# Search (k nearest neighbors)
D, I = query(index, query_embeddings, k=100)

# D contains similarity scores
# I contains indices into the lookup database

# Filter by FDR threshold
fdr_threshold = 0.99996425
hits = D >= fdr_threshold

# Get probabilities for a specific hit
# (requires calibration data)
cal_data = np.load('data/pfam_new_proteins.npy', allow_pickle=True)
# ... extract X_cal, Y_cal from cal_data ...
p0, p1 = simplifed_venn_abers_prediction(X_cal, Y_cal, similarity_score)
probability = (p0 + p1) / 2
```

---

## Pre-computed Thresholds

For convenience, here are pre-computed thresholds for common use cases:

### Pfam (Exact Match)

| Alpha (Error Rate) | FDR Lambda | FNR Lambda |
|--------------------|------------|------------|
| 0.01 (1%) | TBD | TBD |
| 0.05 (5%) | TBD | TBD |
| 0.10 (10%) | 0.99996425 | 0.99974871 |
| 0.20 (20%) | TBD | TBD |

### Pfam (Partial Match)

Partial matches tolerate hits to the same clan/superfamily even if the exact Pfam domain differs.

| Alpha (Error Rate) | FDR Lambda | FNR Lambda |
|--------------------|------------|------------|
| 0.10 (10%) | TBD | TBD |

---

## Understanding the Output

### Search Results CSV

| Column | Description |
|--------|-------------|
| `query_seq` | Query protein sequence |
| `query_meta` | Query metadata from FASTA header |
| `lookup_seq` | Matched protein sequence |
| `D_score` | Cosine similarity score (0-1) |
| `lookup_entry` | UniProt entry ID |
| `lookup_pfam` | Pfam domain annotations |
| `lookup_protein_names` | Protein name |

### With Probabilities

| Column | Description |
|--------|-------------|
| `prob_exact_p0` | Lower bound probability of exact match |
| `prob_exact_p1` | Upper bound probability of exact match |
| `prob_partial_p0` | Lower bound probability of partial match |
| `prob_partial_p1` | Upper bound probability of partial match |

The **calibrated probability** of a match is typically taken as the average: `(p0 + p1) / 2`

---

## Tips

1. **Start with FDR control** if you want high-confidence hits (fewer false positives)
2. **Use FNR control** if you want comprehensive coverage (don't miss true hits)
3. **Check the similarity distribution** in your results to calibrate expectations
4. **Use partial match probabilities** when exact Pfam annotation isn't critical

---

## Next Steps

- See [notebooks/pfam/](../notebooks/pfam/) for detailed Pfam analysis
- See [notebooks/scope/](../notebooks/scope/) for structural classification examples
- See [notebooks/ec/](../notebooks/ec/) for enzyme classification examples
