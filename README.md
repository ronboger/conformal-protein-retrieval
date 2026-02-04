# Conformal Protein Retrieval

Code and notebooks from [Functional protein mining with conformal guarantees](https://www.nature.com/articles/s41467-024-55676-y) (Nature Communications, 2025). This package provides statistically rigorous methods for protein database search with false discovery rate (FDR) and false negative rate (FNR) control.

**[→ GETTING STARTED](GETTING_STARTED.md)** - Quick setup guide (10 minutes)

## Quick Setup

```bash
# 1. Clone and install
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval
pip install -e .

# 2. Download data from Zenodo (4GB total)
# https://zenodo.org/records/14272215
#   → lookup_embeddings.npy (1.1 GB) → data/
#   → lookup_embeddings_meta_data.tsv (535 MB) → data/
#   → pfam_new_proteins.npy (2.4 GB) → data/

# 3. Verify setup
cpr verify --check syn30
# Expected: 59/149 = 39.6% hits at FDR α=0.1
```

See **[GETTING_STARTED.md](GETTING_STARTED.md)** for detailed instructions.

## Repository Structure

```
conformal-protein-retrieval/
├── protein_conformal/     # Core library (FDR/FNR control, Venn-Abers)
├── notebooks/             # Analysis notebooks organized by experiment
│   ├── pfam/             # Pfam domain annotation (Figure 2)
│   ├── scope/            # SCOPe structural classification
│   ├── ec/               # EC number classification
│   └── clean_selection/  # CLEAN enzyme experiments (Tables 1-2)
├── scripts/              # CLI scripts and SLURM jobs
├── data/                 # Data files (see GETTING_STARTED.md)
├── results/              # Pre-computed thresholds and outputs
└── docs/                 # Additional documentation
```

## Quick Start

The `cpr` CLI provides five main commands for functional protein mining:

### 1. Embed protein sequences

```bash
# Embed with Protein-Vec (for general protein search)
cpr embed --input sequences.fasta --output embeddings.npy --model protein-vec

# Embed with CLEAN (for enzyme classification)
cpr embed --input sequences.fasta --output embeddings.npy --model clean
```

### 2. Search for similar proteins with conformal guarantees

The `cpr search` command accepts **both FASTA files and pre-computed embeddings**:

```bash
# From FASTA file (auto-embeds with Protein-Vec)
cpr search --input sequences.fasta --output results.csv --fdr 0.1

# From pre-computed embeddings
cpr search --input embeddings.npy --output results.csv --fdr 0.1

# With FNR control instead of FDR
cpr search --input sequences.fasta --output results.csv --fnr 0.1

# With explicit threshold
cpr search --input sequences.fasta --output results.csv --threshold 0.99998

# Exploratory mode (no filtering, return all k neighbors)
cpr search --input sequences.fasta --output results.csv --no-filter
```

### 3. Convert similarity scores to calibrated probabilities

```bash
# Add Venn-Abers calibrated probabilities to search results
cpr prob \
    --input results.csv \
    --calibration data/pfam_new_proteins.npy \
    --output results_with_probs.csv \
    --n-calib 1000
```

### 4. Calibrate FDR/FNR thresholds for a new embedding model

```bash
# Compute thresholds from your own calibration data
cpr calibrate \
    --calibration my_calibration_data.npy \
    --output thresholds.csv \
    --alpha 0.1 \
    --n-trials 100 \
    --n-calib 1000
```

### 5. Verify paper results

```bash
# Reproduce key results from the paper
cpr verify --check syn30   # JCVI Syn3.0 annotation (39.6% at FDR α=0.1)
cpr verify --check fdr     # FDR threshold calibration
cpr verify --check dali    # DALI prefiltering (82.8% TPR, 31.5% DB reduction)
cpr verify --check clean   # CLEAN enzyme classification
```

## Data Files

Download the following files from [Zenodo](https://zenodo.org/records/14272215) and place in the `data/` directory:

- `pfam_new_proteins.npy` (2.5 GB) - Pfam calibration data for FDR/FNR control
- `lookup_embeddings.npy` (1.1 GB) - UniProt database embeddings (Protein-Vec)
- `lookup_embeddings_meta_data.tsv` - Metadata for lookup database
- `afdb_embeddings_protein_vec.npy` (4.7 GB) - AlphaFold DB embeddings (optional)

## Protein-Vec vs CLEAN Models

### Protein-Vec (general protein search)
- Trained on UniProt with multi-task objectives (Pfam, EC, GO, transmembrane, etc.)
- Best for: broad functional annotation, domain identification, general homology search
- Output: 128-dimensional embeddings
- FDR threshold at α=0.1: λ ≈ 0.9999802

### CLEAN (enzyme classification)
- Trained specifically for EC number classification
- Best for: enzyme function prediction, detailed catalytic annotation
- Output: 128-dimensional embeddings
- Requires ESM embeddings as input (computed automatically)
- See `ec/` directory for CLEAN-specific notebooks

## Creating Custom Calibration Datasets

To calibrate FDR/FNR thresholds for your own protein search tasks:

1. Create a calibration dataset with ground-truth labels (see `data/create_pfam_data.ipynb`)
2. Embed sequences using your chosen model (`cpr embed`)
3. Compute similarity scores and labels (save as .npy with shape `(n_samples, 3)`: `[sim, label_exact, label_partial]`)
4. Run calibration: `cpr calibrate --calibration my_data.npy --output thresholds.csv --alpha 0.1`

**Important:** Ensure your calibration dataset is outside the training data of your embedding model to avoid data leakage.

## Complete Workflow Example

Here's a full example searching viral domains against the Pfam database with FDR control:

```bash
# Option A: One-step search from FASTA (embeds automatically)
cpr search --input viral_domains.fasta --output viral_hits.csv --fdr 0.1

# Option B: Two-step with explicit embedding
cpr embed --input viral_domains.fasta --output viral_embeddings.npy
cpr search --input viral_embeddings.npy --output viral_hits.csv --fdr 0.1
```

The output CSV will contain:
- `query_idx`: Query sequence index
- `match_idx`: Database match index
- `similarity`: Cosine similarity score
- `match_*`: Metadata columns from database (UniProt ID, Pfam domains, etc.)
- `probability`: Calibrated probability of functional match
- `uncertainty`: Venn-Abers uncertainty interval (|p1 - p0|)

## Advanced Usage

### Using Legacy Scripts

For advanced use cases, the original Python scripts are still available in `scripts/`:

```bash
# Legacy search script with more options
python scripts/search.py \
    --fdr \
    --fdr_lambda 0.99998 \
    --output results.csv \
    --query_embedding query.npy \
    --query_fasta query.fasta \
    --lookup_embedding data/lookup_embeddings.npy \
    --lookup_fasta data/lookup_embeddings_meta_data.tsv \
    --k 1000

# Precompute similarity-to-probability lookup table
python scripts/precompute_SVA_probs.py \
    --cal_data data/pfam_new_proteins.npy \
    --output data/pfam_sims_to_probs.csv \
    --partial \
    --n_bins 1000 \
    --n_calib 1000

# Apply precomputed probabilities (faster than on-the-fly computation)
python scripts/get_probs.py \
    --precomputed \
    --precomputed_path data/pfam_sims_to_probs.csv \
    --input results.csv \
    --output results_with_probs.csv \
    --partial
```

## Key Paper Results

This repository reproduces the following results from the paper:

| Claim | Paper | CLI Command | Status |
|-------|-------|-------------|--------|
| JCVI Syn3.0 annotation (Fig 2A) | 39.6% (59/149) at FDR α=0.1 | `cpr verify --check syn30` | ✓ Exact |
| FDR threshold | λ = 0.9999802250 at α=0.1 | `cpr verify --check fdr` | ✓ (~0.002% diff) |
| DALI prefiltering TPR (Table 4-6) | 82.8% | `cpr verify --check dali` | ✓ (~1% diff) |
| DALI database reduction | 31.5% | `cpr verify --check dali` | ✓ Exact |
| CLEAN enzyme loss (Table 1-2) | ≤ α=1.0 | `cpr verify --check clean` | ✓ (0.97) |

## Repository Structure

- `protein_conformal/` - Core utilities for conformal prediction and search
- `scripts/` - Verification scripts and legacy search tools
- `scope/` - SCOPe structural classification experiments
- `pfam/` - Pfam domain annotation notebooks
- `ec/` - EC number classification with CLEAN model
- `data/` - Data processing notebooks and scripts
- `clean_selection/` - CLEAN enzyme selection pipeline
- `tests/` - Test suite (run with `pytest tests/ -v`)

## Contributing & Feature Requests

If you'd like expanded support for specific models or search tasks, please open an issue describing:
1. The embedding model you'd like to use
2. The search/annotation task you're working on
3. Any specific conformal guarantees you need (FDR, FNR, coverage, etc.)

We welcome contributions and look forward to hearing from you!

## Citation

If you use this code or method in your work, please cite:

```bibtex
@article{boger2025functional,
  title={Functional protein mining with conformal guarantees},
  author={Boger, Ron S and Chithrananda, Seyone and Angelopoulos, Anastasios N and Yoon, Peter H and Jordan, Michael I and Doudna, Jennifer A},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={85},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-024-55676-y}
}
```

## License

See LICENSE file for details.
