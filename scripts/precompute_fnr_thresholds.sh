#!/bin/bash

# Script to precompute FNR thresholds for different alpha values
# Usage: ./precompute_fnr_thresholds.sh [OPTIONS]

# Default parameters - can be modified as needed
MIN_ALPHA=0.01
MAX_ALPHA=1.0
NUM_ALPHA_VALUES=100
NUM_TRIALS=100
N_CALIB=1000
OUTPUT_DIR="../results"
TEMP_DIR="./temp_fnr_results"
CSV_OUTPUT="$OUTPUT_DIR/fnr_thresholds.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-alpha)
            MIN_ALPHA="$2"
            shift 2
            ;;
        --max-alpha)
            MAX_ALPHA="$2"
            shift 2
            ;;
        --num-values)
            NUM_ALPHA_VALUES="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --n-calib)
            N_CALIB="$2"
            shift 2
            ;;
        --output)
            CSV_OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --min-alpha FLOAT      Minimum alpha value (default: $MIN_ALPHA)"
            echo "  --max-alpha FLOAT      Maximum alpha value (default: $MAX_ALPHA)"
            echo "  --num-values INT       Number of alpha values to test (default: $NUM_ALPHA_VALUES)"
            echo "  --num-trials INT       Number of trials per alpha (default: $NUM_TRIALS)"
            echo "  --n-calib INT          Calibration set size (default: $N_CALIB)"
            echo "  --output PATH          Output CSV file (default: $CSV_OUTPUT)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Initialize CSV file with header
echo "alpha,lambda_threshold,exact_fnr,partial_fnr" > "$CSV_OUTPUT"

echo "Precomputing FNR thresholds..."
echo "Alpha range: $MIN_ALPHA to $MAX_ALPHA"
echo "Number of alpha values: $NUM_ALPHA_VALUES"
echo "Trials per alpha: $NUM_TRIALS"
echo "Calibration set size: $N_CALIB"
echo "Output file: $CSV_OUTPUT"
echo ""

# Generate alpha values using Python
ALPHA_VALUES=$(python3 -c "
import numpy as np
alphas = np.linspace($MIN_ALPHA, $MAX_ALPHA, $NUM_ALPHA_VALUES)
print(' '.join([str(a) for a in alphas]))
")

# Counter for progress tracking
counter=0
total=$NUM_ALPHA_VALUES

# Loop over alpha values
for alpha in $ALPHA_VALUES; do
    counter=$((counter + 1))
    echo "Processing alpha=$alpha ($counter/$total)..."
    
    # Run FNR generation for exact matches
    echo "  Running exact matches..."
    python3 ../pfam/generate_fnr.py \
        --alpha "$alpha" \
        --partial false \
        --num_trials "$NUM_TRIALS" \
        --n_calib "$N_CALIB" \
        --output "$TEMP_DIR/fnr_exact_$alpha" \
        --add_date false
    
    # Run FNR generation for partial matches
    echo "  Running partial matches..."
    python3 ../pfam/generate_fnr.py \
        --alpha "$alpha" \
        --partial true \
        --num_trials "$NUM_TRIALS" \
        --n_calib "$N_CALIB" \
        --output "$TEMP_DIR/fnr_partial_$alpha" \
        --add_date false
    
    # Extract results and append to CSV using Python
    python3 -c "
import numpy as np
import sys

try:
    # Load exact match results
    exact_data = np.load('$TEMP_DIR/fnr_exact_$alpha.npy', allow_pickle=True).item()
    exact_lhat = np.mean(exact_data['lhats'])
    exact_fnr = np.mean(exact_data['fnrs'])
    
    # Load partial match results  
    partial_data = np.load('$TEMP_DIR/fnr_partial_$alpha.npy', allow_pickle=True).item()
    partial_fnr = np.mean(partial_data['fnrs'])
    
    # Write to CSV
    with open('$CSV_OUTPUT', 'a') as f:
        f.write(f'$alpha,{exact_lhat},{exact_fnr},{partial_fnr}\n')
    
    print(f'  Results: lambda={exact_lhat:.6f}, exact_fnr={exact_fnr:.6f}, partial_fnr={partial_fnr:.6f}')
    
except Exception as e:
    print(f'Error processing alpha=$alpha: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "Error processing alpha=$alpha" >&2
        exit 1
    fi
done

# Clean up temporary files
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "FNR threshold precomputation completed!"
echo "Results saved to: $CSV_OUTPUT"
echo "Total alpha values processed: $total"

