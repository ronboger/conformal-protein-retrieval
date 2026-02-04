#!/usr/bin/env python
"""
Verify DALI Prefiltering Results (Paper Tables 4-6)

Expected results:
- TPR (True Positive Rate): ~82.8%
- Database Reduction: ~31.5%

This script analyzes pre-computed DALI results from the backup data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("DALI Prefiltering Verification (Paper Tables 4-6)")
    print("=" * 60)
    print()

    # Load DALI results
    repo_root = Path(__file__).parent.parent
    dali_csv = repo_root / "results" / "dali_thresholds.csv"

    if not dali_csv.exists():
        print(f"ERROR: DALI results not found at {dali_csv}")
        sys.exit(1)

    df = pd.read_csv(dali_csv)
    print(f"Loaded {len(df)} trials from {dali_csv.name}")
    print()

    # Compute key metrics
    tpr_mean = df["TPR_elbow"].mean() * 100
    tpr_std = df["TPR_elbow"].std() * 100

    frac_kept = df["frac_samples_above_lambda"].mean()
    db_reduction = (1 - frac_kept) * 100

    fnr_mean = df["FNR_elbow"].mean() * 100
    fdr_mean = df["FDR_elbow"].mean()
    elbow_z_mean = df["elbow_z"].mean()
    elbow_z_std = df["elbow_z"].std()

    # Paper claims
    paper_tpr = 82.8
    paper_db_reduction = 31.5

    print("Results:")
    print("-" * 40)
    print(f"TPR (True Positive Rate): {tpr_mean:.1f}% ± {tpr_std:.1f}%")
    print(f"  Paper claims: {paper_tpr}%")
    print(f"  Difference: {abs(tpr_mean - paper_tpr):.1f}%")
    print()
    print(f"Database Reduction: {db_reduction:.1f}%")
    print(f"  Paper claims: {paper_db_reduction}%")
    print(f"  Difference: {abs(db_reduction - paper_db_reduction):.1f}%")
    print()
    print(f"FNR (Miss Rate): {fnr_mean:.1f}%")
    print(f"FDR at elbow: {fdr_mean:.6f}")
    print(f"Elbow z-score: {elbow_z_mean:.1f} ± {elbow_z_std:.1f}")
    print()

    # Verification
    tpr_ok = abs(tpr_mean - paper_tpr) < 2.0  # Within 2%
    db_ok = abs(db_reduction - paper_db_reduction) < 1.0  # Within 1%

    print("=" * 60)
    if tpr_ok and db_ok:
        print("✓ VERIFICATION PASSED")
        print(f"  TPR {tpr_mean:.1f}% matches paper ({paper_tpr}%)")
        print(f"  DB reduction {db_reduction:.1f}% matches paper ({paper_db_reduction}%)")
    else:
        print("⚠ VERIFICATION WARNING")
        if not tpr_ok:
            print(f"  TPR {tpr_mean:.1f}% differs from paper ({paper_tpr}%)")
        if not db_ok:
            print(f"  DB reduction {db_reduction:.1f}% differs from paper ({paper_db_reduction}%)")
    print("=" * 60)

    return 0 if (tpr_ok and db_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
