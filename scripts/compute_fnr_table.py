#!/usr/bin/env python
"""
Compute FNR thresholds at standard alpha levels for the lookup table.

This script computes False Negative Rate (FNR) controlling thresholds using
conformal risk control. FNR thresholds ensure that the fraction of true
positives missed is controlled at level alpha.

The thresholds are computed by:
1. Sampling calibration data multiple times (n_trials)
2. Computing the FNR threshold for each trial
3. Averaging across trials to get a stable estimate

Note on reproducibility:
- Due to random sampling of calibration data, results may vary slightly between runs
- The standard deviation across trials indicates the expected variability
- For exact reproduction, use the same random seed

Usage:
    python scripts/compute_fnr_table.py --calibration data/pfam_new_proteins.npy
    python scripts/compute_fnr_table.py --calibration data/pfam_new_proteins.npy --partial
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_conformal.util import get_thresh_new, get_sims_labels


def compute_fnr_threshold(cal_data, alpha: float, n_trials: int = 100,
                          n_calib: int = 1000, seed: int = None,
                          partial: bool = False) -> dict:
    """
    Compute FNR threshold at a given alpha level.

    Parameters:
        cal_data: Calibration data array
        alpha: Target FNR level (e.g., 0.1 means at most 10% false negatives)
        n_trials: Number of trials for averaging
        n_calib: Number of calibration samples per trial
        seed: Random seed for reproducibility
        partial: If True, use partial matches (at least one Pfam domain matches)

    Returns dict with:
        - mean_threshold: Average threshold across trials
        - std_threshold: Standard deviation across trials
    """
    if seed is not None:
        np.random.seed(seed)

    thresholds = []

    for trial in range(n_trials):
        # Shuffle and sample calibration data
        np.random.shuffle(cal_data)
        trial_data = cal_data[:n_calib]

        # Get similarity scores and labels
        X_cal, y_cal = get_sims_labels(trial_data, partial=partial)

        # Compute FNR threshold
        l_hat = get_thresh_new(X_cal, y_cal, alpha)

        thresholds.append(l_hat)

    return {
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'min_threshold': np.min(thresholds),
        'max_threshold': np.max(thresholds),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute FNR thresholds at standard alpha levels'
    )
    parser.add_argument(
        '--calibration', '-c',
        type=Path,
        required=True,
        help='Path to calibration data (.npy file)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output CSV file (default: results/fnr_thresholds.csv or results/fnr_thresholds_partial.csv)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of calibration trials (default: 100)'
    )
    parser.add_argument(
        '--n-calib',
        type=int,
        default=1000,
        help='Number of calibration samples per trial (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--partial',
        action='store_true',
        help='Use partial matches (at least one Pfam domain matches)'
    )

    args = parser.parse_args()

    # Set default output path based on partial flag
    if args.output is None:
        suffix = '_partial' if args.partial else ''
        args.output = Path(f'results/fnr_thresholds{suffix}.csv')

    # Standard alpha levels that users commonly need
    alpha_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    match_type = "partial" if args.partial else "exact"
    print(f"Computing FNR thresholds ({match_type} matches)")
    print(f"Loading calibration data from {args.calibration}...")
    cal_data = np.load(args.calibration, allow_pickle=True)
    print(f"  Loaded {len(cal_data)} calibration samples")

    print(f"\nComputing thresholds at {len(alpha_levels)} alpha levels...")
    print(f"  Trials per alpha: {args.n_trials}")
    print(f"  Calibration samples per trial: {args.n_calib}")
    print(f"  Random seed: {args.seed}")
    print(f"  Match type: {match_type}")
    print()

    results = []
    for alpha in alpha_levels:
        print(f"  α = {alpha:.3f}...", end=" ", flush=True)

        # Use different seed offset for each alpha to ensure independence
        trial_seed = args.seed + int(alpha * 10000)

        stats = compute_fnr_threshold(
            cal_data.copy(),  # Copy to avoid mutation
            alpha=alpha,
            n_trials=args.n_trials,
            n_calib=args.n_calib,
            seed=trial_seed,
            partial=args.partial
        )

        results.append({
            'alpha': alpha,
            'threshold_mean': stats['mean_threshold'],
            'threshold_std': stats['std_threshold'],
            'threshold_min': stats['min_threshold'],
            'threshold_max': stats['max_threshold'],
            'match_type': match_type,
        })

        print(f"λ = {stats['mean_threshold']:.10f} ± {stats['std_threshold']:.2e}")

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Add human-readable notes
    print(f"\n{'='*70}")
    print(f"FNR Threshold Lookup Table ({match_type} matches)")
    print(f"{'='*70}")
    print(f"{'Alpha':<8} {'Threshold (λ)':<20} {'Std Dev':<12}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['alpha']:<8.3f} {row['threshold_mean']:<20.12f} {row['threshold_std']:<12.2e}")
    print(f"{'='*70}")

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Also save a simple version for easy lookup
    simple_output = args.output.parent / f'fnr_thresholds{"_partial" if args.partial else ""}_simple.csv'
    df[['alpha', 'threshold_mean']].rename(
        columns={'threshold_mean': 'lambda_threshold'}
    ).to_csv(simple_output, index=False)
    print(f"Simple lookup table saved to {simple_output}")

    return df


if __name__ == '__main__':
    main()
