"""
Precompute FDR lambda thresholds for different alpha values.

This script generates a lookup table of lambda threshold values for different alpha levels
(FDR control parameters), computing both exact and partial FDR values in a single CSV file.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_conformal.util import get_thresh_FDR, get_sims_labels, risk

def main(args):
    fdr_thresholds = pd.DataFrame(columns=["alpha", "lambda_threshold", "exact_fdr", "partial_fdr"])
    
    # Load calib data
    print(f"Loading calibration data from {args.cal_data}")
    data = np.load(args.cal_data, allow_pickle=True)
    
    # Use a fixed number of calib points
    n_calib = args.n_calib
    print(f"Using {n_calib} calibration points")
    np.random.shuffle(data)
    cal_data = data[:n_calib]
    
    # Get calibration features and labels for both exact and partial matches
    X_cal_exact, y_cal_exact = get_sims_labels(cal_data, partial=False)
    X_cal_partial, y_cal_partial = get_sims_labels(cal_data, partial=True)
    
    alpha_values = np.linspace(0.01, 0.2, args.n_alpha)
    
    print(f"Computing thresholds for {len(alpha_values)} alpha values between {alpha_values[0]} and {alpha_values[-1]}")
    for alpha in tqdm.tqdm(alpha_values, desc="Computing FDR thresholds"):
        lambda_threshold_exact, exact_fdr = get_thresh_FDR(
            y_cal_exact, X_cal_exact, alpha, args.delta, N=args.N
        )
        lambda_threshold_partial, partial_fdr = get_thresh_FDR(
            y_cal_partial, X_cal_partial, alpha, args.delta, N=args.N
        )   

        new_row = pd.DataFrame({
            "alpha": [alpha],
            "lambda_threshold": [lambda_threshold_exact],  # Use exact match threshold
            "exact_fdr": [exact_fdr],
            "partial_fdr": [partial_fdr]  # FDR for partial matches using exact threshold
        })
        fdr_thresholds = pd.concat([fdr_thresholds, new_row], ignore_index=True)
    
    print(f"Saving FDR thresholds to {args.output}")
    fdr_thresholds.to_csv(args.output, index=False)
    
    print("Done! Example of results:")
    print(fdr_thresholds.head())

def parse_args():
    parser = argparse.ArgumentParser("Precompute FDR lambda thresholds for different alpha values with both exact and partial FDR")
    
    parser.add_argument(
        "--cal_data",
        type=str,
        default="./data/calibration/pfam_new_proteins.npy",
        help="Path to calibration data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/fdr_thresholds.csv",
        help="Output file for the combined FDR threshold table (exact and partial)"
    )
    parser.add_argument(
        "--n_alpha",
        type=int,
        default=100,
        help="Number of alpha values to compute thresholds for (between 0.01 and 0.2, same use as n_bins from precompute_SVA_probs.py~)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="Delta parameter for FDR control"
        # for my own reference: delta is the confidence level on the alpha ceiling error rate guarantee
        # it controls how sure we are that—on any new, unseen query—the actual long–run FDR will indeed stay at or below alpha
        # "confidence of our confidence"
        # If we  set delta = 0.5 (the default), we only get a weak “≥ 50% chance” that our threshold will achieve FDR ≤ α on new queries.
        # If we want a stronger statement—say, “with 95% confidence our FDR will be ≤ α”— we would use delta = 0.05.
    )
    parser.add_argument(
        "--n_calib",
        type=int,
        default=100,
        help="Number of calibration data points from the dataset to use, same use as n_calib in precompute_SVA_probs.py"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=50,
        help="Number of search iterations for get_thresh_FDR"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args) 