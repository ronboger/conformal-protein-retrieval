#!/usr/bin/env python
"""Merge individual FDR threshold results into single CSV files."""
import pandas as pd
from pathlib import Path
import sys

def merge_results(pattern: str, output: str):
    """Merge CSV files matching pattern into single output."""
    results_dir = Path('results')
    files = sorted(results_dir.glob(pattern))

    if not files:
        print(f"No files matching {pattern}")
        return None

    print(f"Merging {len(files)} files matching {pattern}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {f.name}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('alpha').reset_index(drop=True)

    output_path = results_dir / output
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to {output_path}")
    return merged

if __name__ == '__main__':
    print("=== Merging FDR Threshold Results ===\n")

    # Merge exact match results
    exact = merge_results('fdr_exact_alpha_*.csv', 'fdr_thresholds.csv')
    print()

    # Merge partial match results
    partial = merge_results('fdr_partial_alpha_*.csv', 'fdr_thresholds_partial.csv')
    print()

    if exact is not None:
        print("=== Exact Match FDR Thresholds ===")
        print(exact[['alpha', 'threshold_mean', 'threshold_std']].to_string(index=False))

    if partial is not None:
        print("\n=== Partial Match FDR Thresholds ===")
        print(partial[['alpha', 'threshold_mean', 'threshold_std']].to_string(index=False))
