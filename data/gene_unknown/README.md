# JCVI Syn3.0 Unknown Genes

This directory contains protein sequences from the JCVI Syn3.0 minimal bacterial genome that were annotated as "unknown function" or "generic".

## Source

**JCVI Syn3.0** is the minimal bacterial genome created by the J. Craig Venter Institute:

> Hutchison CA 3rd, et al. "Design and synthesis of a minimal bacterial genome."
> Science. 2016 Mar 25;351(6280):aad6253.
> DOI: [10.1126/science.aad6253](https://doi.org/10.1126/science.aad6253)

The 473-gene genome was systematically reduced from *Mycoplasma mycoides* to identify the minimal set of genes required for life.

## Files

| File | Description |
|------|-------------|
| `unknown_aa_seqs.fasta` | 149 protein sequences with unknown/generic function |
| `unknown_aa_seqs.npy` | Pre-computed Protein-Vec embeddings (149 × 512) |

## Gene Naming

- `MMSYN1_XXXX` - Gene identifier in Syn3.0
- `1=Unknown` - Gene with unknown function
- `2=Generic` - Gene with generic/broad annotation

## Results

Using conformal protein retrieval at 10% FDR (α=0.1):
- **59/149 (39.6%)** of unknown genes can be confidently annotated
- Results reproduced in `notebooks/pfam/genes_unknown.ipynb`
- See paper Figure 2A for visualization

## Citation

If using this data, please cite both the CPR paper and the original Syn3.0 paper:

```bibtex
@article{boger2025conformal,
  title={Functional protein mining with conformal guarantees},
  author={Boger, Ron S and Chithrananda, Seyone and Angelopoulos, Anastasios N and Yoon, Peter H and Jordan, Michael I and Doudna, Jennifer A},
  journal={Nature Communications},
  volume={16},
  pages={85},
  year={2025},
  doi={10.1038/s41467-024-55676-y}
}

@article{hutchison2016design,
  title={Design and synthesis of a minimal bacterial genome},
  author={Hutchison, Clyde A and Chuang, Ray-Yuan and Noskov, Vladimir N and others},
  journal={Science},
  volume={351},
  number={6280},
  pages={aad6253},
  year={2016},
  doi={10.1126/science.aad6253}
}
```
