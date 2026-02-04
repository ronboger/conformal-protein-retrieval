# Repository Organization

This document maps the codebase to the paper: [Functional protein mining with conformal guarantees](https://www.nature.com/articles/s41467-024-55676-y) (Nature Communications, 2024).

---

## Paper Figure/Table to Code Mapping

| Paper Element | Description | Notebook/Script | Data Required |
|--------------|-------------|-----------------|---------------|
| **Figure 2A** | JCVI Syn3.0 annotation (39.6%) | `notebooks/pfam/genes_unknown.ipynb` | Zenodo: lookup_embeddings.npy |
| **Figure 2B-G** | FDR/FNR trade-off curves | `notebooks/pfam/analyze_protein_vec_results.ipynb` | pfam_new_proteins.npy |
| **Figure 2H** | Venn-Abers probability calibration | `notebooks/pfam/sva_reliability.ipynb` | calibration_probs.csv |
| **Figure 3A-B** | CLEAN enzyme violin plots | `notebooks/clean_selection/analyze_new_price_pppl.ipynb` | clean_new_v_ec_cluster.npy |
| **Figure 4A** | DALI prefiltering correlation | `notebooks/scope/test_scope_conformal_retrieval.ipynb` | SCOPe data from Zenodo |
| **Table 1** | New-392 enzyme classification | `notebooks/clean_selection/analyze_new_price_pppl.ipynb` | CLEAN embeddings |
| **Table 2** | Price-149 generalizability | `notebooks/clean_selection/analyze_new_price_pppl.ipynb` | CLEAN embeddings |
| **Tables 4-6** | DALI prefiltering results | `notebooks/scope/*.ipynb` | SCOPe + AFDB data |
| **Supp Fig 1** | ECE calibration plot | `notebooks/pfam/sva_reliability.ipynb` | Calibration data |

---

## Directory Structure

```
conformal-protein-retrieval/
├── protein_conformal/           # Core Python package
│   ├── __init__.py
│   ├── util.py                  # Core algorithms: FDR/FNR, Venn-Abers, FAISS
│   ├── embed_protein_vec.py     # Protein-Vec embedding generation
│   ├── scope_utils.py           # SCOPe hierarchical classification
│   ├── gradio_app.py            # GUI launcher
│   └── backend/                 # Gradio web interface
│       ├── gradio_interface.py  # Main UI logic
│       ├── collaborative.py     # Session management, API
│       └── visualization.py     # 3D structure, plots
│
├── scripts/                     # CLI scripts
│   ├── search.py                # Main search with FDR/FNR control
│   ├── get_probs.py             # Venn-Abers probability assignment
│   ├── precompute_SVA_probs.py  # Precompute calibration
│   ├── embed_fasta.sh           # Batch embedding
│   └── pfam/                    # Pfam-specific scripts
│       ├── generate_fdr.py      # FDR threshold computation
│       └── generate_fnr.py      # FNR threshold computation
│
├── notebooks/                   # Analysis notebooks (paper figures)
│   ├── pfam/                    # Pfam domain analysis
│   │   ├── analyze_protein_vec_results.ipynb  # Fig 2B-G
│   │   ├── genes_unknown.ipynb               # Fig 2A (JCVI)
│   │   ├── sva_reliability.ipynb             # Fig 2H, Supp Fig 1
│   │   └── multidomain_search.ipynb          # Multi-domain queries
│   ├── clean_selection/         # Enzyme classification (Tables 1-2)
│   │   ├── analyze_new_price_pppl.ipynb      # Tables 1-2, Fig 3
│   │   └── analyze_clean_hierarchical_loss_protein_vec.ipynb
│   ├── scope/                   # Structural classification (Tables 4-6)
│   │   ├── test_scope_conformal_retrieval.ipynb  # Fig 4
│   │   └── analyze_scope_hierarchical_loss_protein_vec.ipynb
│   ├── ec/                      # EC number classification
│   └── afdb/                    # AlphaFold DB analysis
│
├── clean_selection/             # CLEAN enzyme data
│   ├── clean_new_v_ec_cluster.npy  # 84MB - enzyme embeddings
│   ├── dists.pkl                # Distance matrices
│   ├── sorted_dict.pkl          # Sorted results
│   └── true_labels.pkl          # Ground truth labels
│
├── data/                        # Data files (download from Zenodo)
│   └── ec/                      # EC lookup data
│
├── results/                     # Output results
│   ├── calibration_probs.csv    # Venn-Abers calibration
│   ├── fdr_thresholds.csv       # Pre-computed FDR λ values
│   └── fnr_thresholds.csv       # Pre-computed FNR λ values
│
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   └── test_util.py             # Unit tests for core functions
│
├── docs/                        # Documentation
│   ├── INSTALLATION.md          # Installation guide
│   └── QUICKSTART.md            # Usage examples
│
├── DEVELOPMENT.md               # Developer guide & roadmap
├── pyproject.toml               # Package configuration
├── environment.yml              # Conda environment
├── dockerfile                   # Docker build
└── docker-compose.yml           # Docker compose
```

---

## Core Algorithms

### 1. Conformal Risk Control (FDR)

**Location**: `protein_conformal/util.py` → `get_thresh_FDR()`, `get_thresh_new_FDR()`

**Paper Section**: Methods - "Learn then Test (LTT)"

```python
# Finds threshold λ such that FDR ≤ α with probability ≥ 1-δ
lhat = get_thresh_FDR(labels, sims, alpha=0.1, delta=0.5, N=100)
```

### 2. Conformal Risk Control (FNR)

**Location**: `protein_conformal/util.py` → `get_thresh_new()`

**Paper Section**: Methods - "FNR Control"

```python
# Finds threshold λ such that FNR ≤ α
lhat = get_thresh_new(sims, labels, alpha=0.1)
```

### 3. Venn-Abers Prediction

**Location**: `protein_conformal/util.py` → `simplifed_venn_abers_prediction()`

**Paper Section**: Methods - "Inductive Venn-Abers Predictors"

```python
# Returns calibrated probability bounds [p0, p1]
p0, p1 = simplifed_venn_abers_prediction(X_cal, Y_cal, x_test)
probability = (p0 + p1) / 2  # Point estimate
```

### 4. Hierarchical Loss

**Location**: `protein_conformal/util.py` → `scope_hierarchical_loss()`

**Paper Section**: Methods - "Hierarchical Risk"

```python
# Returns loss based on SCOPe hierarchy depth
loss, is_exact = scope_hierarchical_loss('a.1.1.1', 'a.1.2.1')
# loss=2 (superfamily mismatch), is_exact=False
```

---

## Key Results to Verify

### Figure 2A: JCVI Syn3.0 Annotation
- **Claim**: 39.6% of 149 genes got exact functional hits at FDR α=0.1
- **Expected**: 59 hits / 149 genes
- **Notebook**: `notebooks/pfam/genes_unknown.ipynb`

### Tables 1-2: Enzyme Classification
- **Claim Table 1** (New-392): Precision=56.80±1.64, Recall=63.71±0.29
- **Claim Table 2** (Price-149): Precision=55.98, Recall=49.34
- **Notebook**: `notebooks/clean_selection/analyze_new_price_pppl.ipynb`

### Tables 4-6: DALI Prefiltering
- **Claim**: 82.8% TPR, 31.5% database reduction, FNR=0.182
- **Notebook**: `notebooks/scope/test_scope_conformal_retrieval.ipynb`

---

## Data Sources

### Zenodo (https://zenodo.org/records/14272215)
- `pfam_new_proteins.npy` (2.5 GB) - Pfam calibration
- `lookup_embeddings.npy` (1.1 GB) - UniProt embeddings
- `afdb_embeddings_protein_vec.npy` (4.7 GB) - AFDB embeddings
- `scope_supplement.zip` - SCOPe data
- `ec_supplement.zip` - EC classification data
- `clean_selection.zip` - CLEAN enzyme data

### Protein-Vec Model
- Source: [TODO - add link]
- Files needed: `protein_vec.ckpt`, `protein_vec_params.json`
