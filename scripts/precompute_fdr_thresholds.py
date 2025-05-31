"""
Precompute FDR lambda thresholds for different alpha values.

This script generates a lookup table of lambda threshold values for different alpha levels
(FDR control parameters), saving them to a CSV file for easy use in the search pipeline.
"""

import numpy as np
import pandas as pd
import argparse
import tqdm
from protein_conformal.util import get_thresh_FDR, get_sims_labels

def main(args):
    # Create DataFrame to store results
    fdr_thresholds = pd.DataFrame(columns=["alpha", "lambda_threshold", "empirical_fdr"])
    
    # Load calibration data
    print(f"Loading calibration data from {args.cal_data}")
    data = np.load(args.cal_data, allow_pickle=True)
    
    # Use a fixed number of calibration points
    n_calib = args.n_calib
    print(f"Using {n_calib} calibration points")
    np.random.shuffle(data)
    cal_data = data[:n_calib]
    
    # Get calibration features and labels
    X_cal, y_cal = get_sims_labels(cal_data, partial=False)
    
    # Loop through alpha values from 0.01 to 0.2
    alpha_values = np.linspace(0.01, 0.2, args.n_alpha)
    
    print(f"Computing thresholds for {len(alpha_values)} alpha values between {alpha_values[0]} and {alpha_values[-1]}")
    for alpha in tqdm.tqdm(alpha_values, desc="Computing FDR thresholds"):
        # Compute lambda threshold using get_thresh_FDR
        lambda_threshold, empirical_fdr = get_thresh_FDR(
            y_cal, X_cal, alpha, args.delta, N=args.N
        )
        
        # Add to DataFrame
        new_row = pd.DataFrame({
            "alpha": [alpha],
            "lambda_threshold": [lambda_threshold],
            "empirical_fdr": [empirical_fdr]
        })
        fdr_thresholds = pd.concat([fdr_thresholds, new_row], ignore_index=True)
    
    # Save results to CSV
    print(f"Saving FDR thresholds to {args.output}")
    fdr_thresholds.to_csv(args.output, index=False)
    
    print("Done! Example of results:")
    print(fdr_thresholds.head())

def parse_args():
    parser = argparse.ArgumentParser("Precompute FDR lambda thresholds for different alpha values")
    
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
        help="Output file for the FDR threshold table"
    )
    parser.add_argument(
        "--n_alpha",
        type=int,
        default=20,
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
        default=1000,
        help="Number of calibration data points from the dataset to use, same use as n_calib in precompute_SVA_probs.py"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=100,
        help="Number of search iterations for get_thresh_FDR"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args) 