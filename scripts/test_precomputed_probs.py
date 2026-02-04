#!/usr/bin/env python
"""
Test that precomputed probability lookup gives same results as computing from scratch.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from protein_conformal.util import simplifed_venn_abers_prediction, get_sims_labels

print("=" * 60)
print("Precomputed Probability Verification")
print("=" * 60)
print()

# Load calibration data
print("Loading calibration data...")
cal_data = np.load('data/pfam_new_proteins.npy', allow_pickle=True)
np.random.seed(42)
np.random.shuffle(cal_data)
cal_subset = cal_data[:100]

X_cal, y_cal = get_sims_labels(cal_subset, partial=False)
X_cal = X_cal.flatten()
y_cal = y_cal.flatten()
print(f"  Calibration pairs: {len(X_cal)}")
print(f"  Similarity range: [{X_cal.min():.6f}, {X_cal.max():.6f}]")
print()

# Create precomputed lookup table
print("Creating precomputed lookup table (100 bins)...")
min_sim, max_sim = X_cal.min(), X_cal.max()
bins = np.linspace(min_sim, max_sim, 100)

lookup = []
for sim in bins:
    p0, p1 = simplifed_venn_abers_prediction(X_cal, y_cal, sim)
    lookup.append({'similarity': sim, 'p0': p0, 'p1': p1, 'prob': (p0+p1)/2})

lookup_df = pd.DataFrame(lookup)
print(f"  Lookup table: {len(lookup_df)} entries")
print()

# Test on random similarity values
print("Testing lookup vs direct computation on 20 random values...")
test_sims = np.random.uniform(min_sim, max_sim, 20)

print(f"{'Similarity':>12} | {'Direct':>8} | {'Lookup':>8} | {'Diff':>8}")
print("-" * 50)

max_diff = 0
for sim in test_sims:
    # Direct computation
    p0, p1 = simplifed_venn_abers_prediction(X_cal, y_cal, sim)
    prob_direct = (p0 + p1) / 2

    # Lookup with interpolation
    lower = lookup_df[lookup_df['similarity'] <= sim].iloc[-1] if len(lookup_df[lookup_df['similarity'] <= sim]) > 0 else lookup_df.iloc[0]
    upper = lookup_df[lookup_df['similarity'] >= sim].iloc[0] if len(lookup_df[lookup_df['similarity'] >= sim]) > 0 else lookup_df.iloc[-1]
    prob_lookup = (lower['prob'] + upper['prob']) / 2

    diff = abs(prob_direct - prob_lookup)
    max_diff = max(max_diff, diff)
    print(f"{sim:12.8f} | {prob_direct:8.4f} | {prob_lookup:8.4f} | {diff:8.4f}")

print()
print("=" * 60)
if max_diff < 0.01:
    print(f"✓ VERIFICATION PASSED (max diff: {max_diff:.4f})")
    print("  Precomputed lookup matches direct computation")
else:
    print(f"⚠ VERIFICATION WARNING (max diff: {max_diff:.4f})")
    print("  Consider using more bins for better accuracy")
print("=" * 60)

# Save the lookup table
output_path = 'data/sim2prob_lookup.csv'
lookup_df.to_csv(output_path, index=False)
print(f"\nSaved lookup table to: {output_path}")
