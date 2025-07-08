#!/bin/bash

# default parameters
CAL_DATA="${CAL_DATA:-./data/calibration/pfam_new_proteins.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
N_ALPHA="${N_ALPHA:-100}"
DELTA="${DELTA:-0.5}"
N_CALIB="${N_CALIB:-100}"
N="${N:-50}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== Precomputing FDR Thresholds ==="
echo "Calibration data: $CAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Number of alpha values: $N_ALPHA"
echo "Delta parameter: $DELTA"
echo "Number of calibration points: $N_CALIB"
echo "Number of search iterations: $N"
echo

# Precompute FDR thresholds for both exact and partial matches
echo "Computing FDR thresholds for both exact and partial matches..."
python -c "
import numpy as np
import pandas as pd
import sys
import os
sys.path.append('.')

# Import directly from the util module to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location('util', './protein_conformal/util.py')
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

get_thresh_FDR = util.get_thresh_FDR
get_sims_labels = util.get_sims_labels
risk = util.risk

import tqdm

# Parameters
cal_data_path = '$CAL_DATA'
output_path = '$OUTPUT_DIR/fdr_thresholds.csv'
n_alpha = $N_ALPHA
delta = $DELTA
n_calib = $N_CALIB
N = $N

print(f'Loading calibration data from {cal_data_path}')
data = np.load(cal_data_path, allow_pickle=True)

print(f'Using {n_calib} calibration points')
np.random.shuffle(data)
cal_data = data[:n_calib]

# Get calib features and labels for both exact and partial matches
X_cal_exact, y_cal_exact = get_sims_labels(cal_data, partial=False)
X_cal_partial, y_cal_partial = get_sims_labels(cal_data, partial=True)

# Create df to store results with exact and partial FDR
fdr_thresholds = pd.DataFrame(columns=['alpha', 'lambda_threshold', 'exact_fdr', 'partial_fdr'])

# Loop through alpha values from 0.01 to 0.2
alpha_values = np.linspace(0.01, 0.2, n_alpha)

print(f'Computing thresholds for {len(alpha_values)} alpha values between {alpha_values[0]:.3f} and {alpha_values[-1]:.3f}')
for alpha in tqdm.tqdm(alpha_values, desc='Computing FDR thresholds'):
    # Compute lambda threshold and FDR for exact matches
    lambda_threshold_exact, exact_fdr = get_thresh_FDR(
        y_cal_exact, X_cal_exact, alpha, delta, N=N
    )
    
    # Compute lambda threshold and FDR for partial matches
    lambda_threshold_partial, partial_fdr = get_thresh_FDR(
        y_cal_partial, X_cal_partial, alpha, delta, N=N
    )
    
    # Add to DataFrame (using exact match threshold as the main threshold)
    new_row = pd.DataFrame({
        'alpha': [alpha],
        'lambda_threshold': [lambda_threshold_exact],  
        'exact_fdr': [exact_fdr],
        'partial_fdr': [partial_fdr] 
    })
    fdr_thresholds = pd.concat([fdr_thresholds, new_row], ignore_index=True)

# Save results to CSV
print(f'Saving FDR thresholds to {output_path}')
fdr_thresholds.to_csv(output_path, index=False)

print('Example of results:')
print(fdr_thresholds.head())
"

echo "FDR thresholds computation completed successfully!" 