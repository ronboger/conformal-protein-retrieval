#!/usr/bin/env python
"""
Compute CLEAN hierarchical thresholds at standard alpha levels.

This script uses the hierarchical conformal prediction from the paper to compute
thresholds for CLEAN enzyme classification. Uses max hierarchical loss with
euclidean distances, matching the paper's approach for Tables 1-2.

The thresholds are computed by:
1. Sampling calibration data multiple times (n_trials)
2. Computing the hierarchical threshold for each trial
3. Averaging across trials to get a stable estimate

Usage:
    python scripts/compute_clean_table.py --calibration data/clean/clean_new_v_ec_cluster.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_conformal.util import get_thresh_max_hierarchical, get_hierarchical_max_loss


def compute_clean_threshold(cal_data, alpha: float, n_trials: int = 20,
                            n_calib: int = 300, seed: int = None) -> dict:
    """
    Compute CLEAN hierarchical threshold at a given alpha level.

    Returns dict with:
        - mean_threshold: Average threshold across trials
        - std_threshold: Standard deviation across trials
        - mean_test_loss: Average test loss across trials
        - std_test_loss: Standard deviation of test loss
    """
    if seed is not None:
        np.random.seed(seed)

    # Get distance range for lambda grid
    all_dists = np.concatenate([q['S_i'] for q in cal_data])
    lambdas = np.linspace(all_dists.min(), all_dists.max(), 500)

    thresholds = []
    test_losses = []

    for trial in range(n_trials):
        np.random.shuffle(cal_data)
        trial_cal = cal_data[:n_calib]
        trial_test = cal_data[n_calib:]

        lhat, _ = get_thresh_max_hierarchical(
            trial_cal, lambdas, alpha, sim="euclidean"
        )

        if lhat is not None:
            test_loss = get_hierarchical_max_loss(trial_test, lhat, sim="euclidean")
            thresholds.append(lhat)
            test_losses.append(test_loss)

    if not thresholds:
        return {
            'mean_threshold': np.nan,
            'std_threshold': np.nan,
            'mean_test_loss': np.nan,
            'std_test_loss': np.nan,
            'n_valid_trials': 0,
        }

    return {
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'min_threshold': np.min(thresholds),
        'max_threshold': np.max(thresholds),
        'n_valid_trials': len(thresholds),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute CLEAN hierarchical thresholds at standard alpha levels'
    )
    parser.add_argument(
        '--calibration', '-c',
        type=Path,
        required=True,
        help='Path to CLEAN calibration data (.npy file)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results/clean_thresholds.csv'),
        help='Output CSV file'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of calibration trials (default: 20, matching paper)'
    )
    parser.add_argument(
        '--n-calib',
        type=int,
        default=300,
        help='Number of calibration samples per trial (default: 300, matching paper)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--alpha-levels',
        type=str,
        default=None,
        help='Comma-separated alpha levels (default: 0.5,1.0,1.5,2.0,2.5,3.0)'
    )

    args = parser.parse_args()

    # Parse alpha levels
    if args.alpha_levels:
        alpha_levels = [float(x.strip()) for x in args.alpha_levels.split(',')]
    else:
        # Standard alpha levels for hierarchical loss (0 = exact, 4 = class mismatch)
        alpha_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("Computing CLEAN Hierarchical Thresholds")
    print(f"Loading calibration data from {args.calibration}...")
    cal_data = np.load(args.calibration, allow_pickle=True)
    print(f"  Loaded {len(cal_data)} samples (New-392 dataset)")

    print(f"\nComputing thresholds at {len(alpha_levels)} alpha levels...")
    print(f"  Trials per alpha: {args.n_trials}")
    print(f"  Calibration samples per trial: {args.n_calib}")
    print(f"  Random seed: {args.seed}")
    print()

    results = []
    for alpha in alpha_levels:
        print(f"  alpha = {alpha:.1f}...", end=" ", flush=True)

        trial_seed = args.seed + int(alpha * 1000)

        stats = compute_clean_threshold(
            cal_data.copy(),
            alpha=alpha,
            n_trials=args.n_trials,
            n_calib=args.n_calib,
            seed=trial_seed,
        )

        results.append({
            'alpha': alpha,
            'threshold_mean': stats['mean_threshold'],
            'threshold_std': stats['std_threshold'],
            'threshold_min': stats.get('min_threshold', np.nan),
            'threshold_max': stats.get('max_threshold', np.nan),
            'test_loss_mean': stats['mean_test_loss'],
            'test_loss_std': stats['std_test_loss'],
            'n_valid_trials': stats['n_valid_trials'],
        })

        print(f"lambda = {stats['mean_threshold']:.2f} +/- {stats['std_threshold']:.2f}, "
              f"test_loss = {stats['mean_test_loss']:.2f} ({stats['n_valid_trials']}/{args.n_trials} trials)")

    # Create DataFrame and save
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("CLEAN Hierarchical Threshold Lookup Table")
    print(f"{'='*70}")
    print(f"{'Alpha':<8} {'Threshold':<16} {'Std Dev':<12} {'Test Loss':<12} {'Trials':<8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['alpha']:<8.1f} {row['threshold_mean']:<16.4f} "
              f"{row['threshold_std']:<12.4f} {row['test_loss_mean']:<12.4f} "
              f"{int(row['n_valid_trials']):<8}")
    print(f"{'='*70}")

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    return df


if __name__ == '__main__':
    main()
