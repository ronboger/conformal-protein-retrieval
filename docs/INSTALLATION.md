# Installation Guide

This guide covers how to install Conformal Protein Retrieval (CPR) and download the required data files.

## Prerequisites

- Python 3.9 or higher
- ~15 GB disk space for full dataset
- GPU recommended for embedding (but CPU works)

## Quick Install

```bash
# Clone the repository
git clone https://github.com/ronboger/conformal-protein-retrieval.git
cd conformal-protein-retrieval

# Install the package
pip install -e .

# Or with GUI support
pip install -e ".[gui]"

# Or with all optional dependencies
pip install -e ".[all]"
```

## Conda Environment (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml
conda activate cpr

# Install the package
pip install -e .
```

## Docker

```bash
# Build the image
docker build -t cpr .

# Run with GUI
docker run -p 7860:7860 cpr python -m protein_conformal.gradio_app
```

---

## Downloading Data

All data files are hosted on Zenodo: https://zenodo.org/records/14272215

### Required Files (Minimum)

For basic FDR/FNR-controlled search against Pfam:

| File | Size | Download |
|------|------|----------|
| `pfam_new_proteins.npy` | 2.5 GB | [Download](https://zenodo.org/records/14272215/files/pfam_new_proteins.npy) |

### For UniProt Search

| File | Size | Download |
|------|------|----------|
| `lookup_embeddings.npy` | 1.1 GB | [Download](https://zenodo.org/records/14272215/files/lookup_embeddings.npy) |
| `lookup_embeddings_meta_data.tsv` | 560 MB | [Download](https://zenodo.org/records/14272215/files/lookup_embeddings_meta_data.tsv) |

### For AlphaFold DB Search

| File | Size | Download |
|------|------|----------|
| `afdb_embeddings_protein_vec.npy` | 4.7 GB | [Download](https://zenodo.org/records/14272215/files/afdb_embeddings_protein_vec.npy) |
| `AFDB_sequences.fasta` | 671 MB | [Download](https://zenodo.org/records/14272215/files/AFDB_sequences.fasta) |

### Supplementary Data

| File | Size | Description |
|------|------|-------------|
| `scope_supplement.zip` | 800 MB | SCOPe hierarchical risk data |
| `ec_supplement.zip` | 199 MB | EC number classification data |
| `clean_selection.zip` | 1.6 GB | Improved enzyme classification data |

### Download Script

```bash
# Create data directory
mkdir -p data

# Download minimum required files
cd data

# Pfam calibration data (required for FDR/FNR control)
wget https://zenodo.org/records/14272215/files/pfam_new_proteins.npy

# UniProt lookup database (for general protein search)
wget https://zenodo.org/records/14272215/files/lookup_embeddings.npy
wget https://zenodo.org/records/14272215/files/lookup_embeddings_meta_data.tsv
```

---

## Protein-Vec Model Weights

To generate embeddings for new proteins, you need the Protein-Vec model weights.

### Option 1: Download Pre-trained Weights

**TODO**: Add download link for Protein-Vec weights

The model files should be placed in `protein_vec_models/`:
```
protein_vec_models/
├── protein_vec.ckpt           # Model checkpoint
├── protein_vec_params.json    # Model configuration
├── model_protein_moe.py       # Model definition
└── utils_search.py            # Utility functions
```

### Option 2: Use Pre-computed Embeddings

If you only need to search against existing databases (UniProt, AFDB), you can skip the embedding step and use the pre-computed embeddings from Zenodo.

---

## Verifying Installation

```bash
# Check that the package is installed
python -c "import protein_conformal; print('OK')"

# Run the test suite
pip install pytest
pytest tests/ -v

# Launch the GUI (if installed with [gui])
python -m protein_conformal.gradio_app
```

---

## Directory Structure

After downloading, your directory should look like:

```
conformal-protein-retrieval/
├── data/
│   ├── pfam_new_proteins.npy          # Calibration data
│   ├── lookup_embeddings.npy          # UniProt embeddings
│   └── lookup_embeddings_meta_data.tsv
├── protein_vec_models/                 # Model weights (if embedding)
│   ├── protein_vec.ckpt
│   └── protein_vec_params.json
├── protein_conformal/                  # Source code
└── ...
```

---

## Troubleshooting

### FAISS Installation Issues

If you encounter issues with `faiss-cpu`:

```bash
# Try conda instead of pip
conda install -c pytorch faiss-cpu

# Or for GPU support
conda install -c pytorch faiss-gpu
```

### Memory Issues

The calibration data (`pfam_new_proteins.npy`) is large. If you run into memory issues:

1. Use a machine with at least 8 GB RAM
2. Consider using memory-mapped arrays:
   ```python
   data = np.load('pfam_new_proteins.npy', mmap_mode='r', allow_pickle=True)
   ```

### PyTorch/Transformers Issues

For embedding, ensure compatible versions:

```bash
pip install torch>=2.0.0 transformers>=4.30.0
```

---

## Next Steps

- See [Quick Start](quickstart.md) for usage examples
- See [API Reference](api.md) for programmatic use
- See the [notebooks/](../notebooks/) directory for detailed analysis examples
