#!/usr/bin/env python
"""Quick FDR calibration check - compare the two datasets."""
import numpy as np
import sys
sys.path.insert(0, '.')
from protein_conformal.util import get_sims_labels, get_thresh_FDR

print("Quick FDR Calibration Check")
print("=" * 50)

# Load both datasets
pfam_new = np.load('data/pfam_new_proteins.npy', allow_pickle=True)
backup = np.load('/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy', allow_pickle=True)

print(f"pfam_new: {len(pfam_new)} samples")
print(f"backup:   {len(backup)} samples")
print()

# Compare similarity and label distributions
sims1, labels1 = get_sims_labels(pfam_new[:500], partial=False)
sims2, labels2 = get_sims_labels(backup[:500], partial=False)

print("Stats from first 500 samples:")
print(f"  pfam_new - positives: {labels1.sum()}/{labels1.size} ({100*labels1.mean():.2f}%)")
print(f"  backup   - positives: {labels2.sum()}/{labels2.size} ({100*labels2.mean():.2f}%)")
print()

# Run a single FDR calibration on each
print("Single FDR trial (n_calib=1000, alpha=0.1):")
np.random.seed(42)

# pfam_new
np.random.shuffle(pfam_new)
X1, y1 = get_sims_labels(pfam_new[:1000], partial=False)
lhat1, _ = get_thresh_FDR(y1, X1, alpha=0.1, delta=0.5, N=100)

# backup
np.random.shuffle(backup)
X2, y2 = get_sims_labels(backup[:1000], partial=False)
lhat2, _ = get_thresh_FDR(y2, X2, alpha=0.1, delta=0.5, N=100)

print(f"  pfam_new λ: {lhat1:.10f}")
print(f"  backup   λ: {lhat2:.10f}")
print(f"  Paper    λ: 0.9999802250 (from pfam_fdr_2024-06-25.npy)")
print()

# Which is closer to paper?
diff1 = abs(lhat1 - 0.9999802250)
diff2 = abs(lhat2 - 0.9999802250)
print(f"Difference from paper threshold:")
print(f"  pfam_new: {diff1:.10f}")
print(f"  backup:   {diff2:.10f}")
print()

if diff1 < diff2:
    print("→ pfam_new is closer to paper threshold")
else:
    print("→ backup is closer to paper threshold")
