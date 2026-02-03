#!/usr/bin/env python
"""
Investigate FDR calibration discrepancy between datasets.
Checks for data leakage and compares calibration results.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from protein_conformal.util import get_sims_labels, get_thresh_FDR

print("=" * 60)
print("FDR Calibration Dataset Investigation")
print("=" * 60)
print()

# Load both calibration datasets
print("Loading datasets...")
pfam_new = np.load('data/pfam_new_proteins.npy', allow_pickle=True)
backup_data = np.load('/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy', allow_pickle=True)

print(f'pfam_new_proteins.npy: {len(pfam_new)} samples')
print(f'backup dataset: {len(backup_data)} samples')
print()

# Check for overlap (potential leakage)
print("Checking for overlap between datasets...")
# Meta is an array, so convert to tuple for hashing
pfam_metas = set(tuple(d['meta'].tolist()) if hasattr(d['meta'], 'tolist') else (d['meta'],) for d in pfam_new)
backup_metas = set(tuple(d['meta'].tolist()) if hasattr(d['meta'], 'tolist') else (d['meta'],) for d in backup_data)
overlap = pfam_metas & backup_metas
print(f"  Unique query sets in pfam_new: {len(pfam_metas)}")
print(f"  Unique query sets in backup: {len(backup_metas)}")
print(f"  Overlap: {len(overlap)} ({len(overlap)/len(pfam_metas)*100:.1f}% of pfam_new)")
print()

# Compare similarity distributions
print("Similarity score distributions:")
sims_new, labels_new = get_sims_labels(pfam_new[:500], partial=False)
sims_backup, labels_backup = get_sims_labels(backup_data[:500], partial=False)

print(f"  pfam_new (500 samples):")
print(f"    Similarity: min={sims_new.min():.6f}, max={sims_new.max():.6f}, mean={sims_new.mean():.6f}")
print(f"    Labels: {labels_new.sum()}/{labels_new.size} positive ({labels_new.mean()*100:.1f}%)")
print()
print(f"  backup (500 samples):")
print(f"    Similarity: min={sims_backup.min():.6f}, max={sims_backup.max():.6f}, mean={sims_backup.mean():.6f}")
print(f"    Labels: {labels_backup.sum()}/{labels_backup.size} positive ({labels_backup.mean()*100:.1f}%)")
print()

# Run FDR calibration on both with same parameters
print("Running FDR calibration (alpha=0.1, n_calib=1000, 10 trials)...")
print()

def run_fdr_trials(data, name, n_trials=10, n_calib=1000):
    lhats = []
    risks = []
    tprs = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        np.random.shuffle(data)
        cal_data = data[:n_calib]
        test_data = data[n_calib:n_calib+500]

        X_cal, y_cal = get_sims_labels(cal_data, partial=False)
        X_test, y_test = get_sims_labels(test_data, partial=False)

        lhat, fdr_cal = get_thresh_FDR(y_cal, X_cal, alpha=0.1, delta=0.5, N=100)
        lhats.append(lhat)

        # Calculate test risk and TPR
        preds = (X_test >= lhat).astype(int)
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        risk = fp / (fp + tp) if (fp + tp) > 0 else 0

        tprs.append(tpr)
        risks.append(risk)

    print(f"{name}:")
    print(f"  λ (threshold): {np.mean(lhats):.10f} ± {np.std(lhats):.10f}")
    print(f"  Risk (FDR):    {np.mean(risks):.4f} ± {np.std(risks):.4f}")
    print(f"  TPR:           {np.mean(tprs)*100:.1f}% ± {np.std(tprs)*100:.1f}%")
    print()
    return lhats, risks, tprs

lhats_new, risks_new, tprs_new = run_fdr_trials(pfam_new.copy(), "pfam_new_proteins")
lhats_backup, risks_backup, tprs_backup = run_fdr_trials(backup_data.copy(), "backup_dataset")

print("=" * 60)
print("CONCLUSION")
print("=" * 60)
if abs(np.mean(lhats_new) - np.mean(lhats_backup)) < 0.00001:
    print("✓ Thresholds are similar - datasets likely compatible")
else:
    print("⚠ Thresholds differ significantly!")
    print(f"  Difference: {abs(np.mean(lhats_new) - np.mean(lhats_backup)):.10f}")

if len(overlap) > len(pfam_metas) * 0.5:
    print("⚠ High overlap between datasets - potential data source")
else:
    print("✓ Low overlap - datasets appear independent")
