#!/bin/bash

# Precompute FNR lambda thresholds for different alpha values
# This script generates lookup tables for FNR thresholds using inline Python

# Set default parameters
CAL_DATA="${CAL_DATA:-./data/calibration/pfam_new_proteins.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
N_ALPHA="${N_ALPHA:-20}"
N_CALIB="${N_CALIB:-100}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== Precomputing FNR Thresholds ==="
echo "Calibration data: $CAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Number of alpha values: $N_ALPHA"
echo "Number of calibration points: $N_CALIB"
echo

# Precompute FNR thresholds for both exact and partial matches
echo "Computing FNR thresholds for both exact and partial matches..."
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

get_thresh_new = util.get_thresh_new
get_sims_labels = util.get_sims_labels
calculate_false_negatives = util.calculate_false_negatives

import tqdm

# Parameters
cal_data_path = '$CAL_DATA'
output_path = '$OUTPUT_DIR/fnr_thresholds.csv'
n_alpha = $N_ALPHA
n_calib = $N_CALIB

print(f'Loading calibration data from {cal_data_path}')
data = np.load(cal_data_path, allow_pickle=True)

print(f'Using {n_calib} calibration points')
np.random.shuffle(data)
cal_data = data[:n_calib]

# Get calib features and labels for both exact, partial matches
X_cal_exact, y_cal_exact = get_sims_labels(cal_data, partial=False)
X_cal_partial, y_cal_partial = get_sims_labels(cal_data, partial=True)

# Create df to store results with exact and partial FNR
fnr_thresholds = pd.DataFrame(columns=['alpha', 'lambda_threshold', 'exact_fnr', 'partial_fnr'])

# Loop through alpha values from 0.01 to 0.2
alpha_values = np.linspace(0.01, 0.2, n_alpha)

print(f'Computing thresholds for {len(alpha_values)} alpha values between {alpha_values[0]:.3f} and {alpha_values[-1]:.3f}')
for alpha in tqdm.tqdm(alpha_values, desc='Computing FNR thresholds'):
    # Compute lambda threshold using get_thresh_new for exact matches
    lambda_threshold_exact = get_thresh_new(X_cal_exact, y_cal_exact, alpha)
    
    # Calculate empirical FNR for exact matches using exact threshold
    exact_fnr = calculate_false_negatives(X_cal_exact, y_cal_exact, lambda_threshold_exact)
    
    # Calculate what the actual FNR would be for partial matches using the exact threshold
    actual_partial_fnr = calculate_false_negatives(X_cal_partial, y_cal_partial, lambda_threshold_exact)
    
    # Add to DataFrame
    new_row = pd.DataFrame({
        'alpha': [alpha],
        'lambda_threshold': [lambda_threshold_exact],  # Use exact match threshold
        'exact_fnr': [exact_fnr],
        'partial_fnr': [actual_partial_fnr]  # FNR for partial matches using exact threshold
    })
    fnr_thresholds = pd.concat([fnr_thresholds, new_row], ignore_index=True)

# Save results to CSV
print(f'Saving FNR thresholds to {output_path}')
fnr_thresholds.to_csv(output_path, index=False)

print('Example of results:')
print(fnr_thresholds.head())
"
