#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Parameters
MIN_ALPHA=0.01
MAX_ALPHA=0.2
NUM_ALPHA_VALUES=100
NUM_TRIALS=100
N_CALIB=100
DELTA=0.5
OUTPUT_DIR="$PROJECT_ROOT/results"
TEMP_DIR="$SCRIPT_DIR/temp_fdr_results"
CSV_OUTPUT="$OUTPUT_DIR/fdr_thresholds.csv"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"


mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Initialize CSV file with header
echo "alpha,lambda_threshold,exact_fdr,partial_fdr" > "$CSV_OUTPUT"

# Generate alpha values using Python
ALPHA_VALUES=$(python -c "
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
    
    # Run FDR generation for exact matches
    python "$PROJECT_ROOT/pfam/generate_fdr.py" \
        --alpha "$alpha" \
        --num_trials "$NUM_TRIALS" \
        --n_calib "$N_CALIB" \
        --delta "$DELTA" \
        --output "$TEMP_DIR/fdr_exact_$alpha"
    
    # Run FDR generation for partial matches
    echo "  Running partial matches..."
    python "$PROJECT_ROOT/pfam/generate_fdr.py" \
        --alpha "$alpha" \
        --partial \
        --num_trials "$NUM_TRIALS" \
        --n_calib "$N_CALIB" \
        --delta "$DELTA" \
        --output "$TEMP_DIR/fdr_partial_$alpha"
    
    # Extract results and append to CSV using Python
    python -c "
import numpy as np
import sys
import os

# Convert MINGW64 paths to Windows format
def mingw_to_windows_path(path):
    if path.startswith('/c/'):
        return 'C:/' + path[3:]
    return path

try:
    temp_dir = mingw_to_windows_path('$TEMP_DIR')
    csv_output = mingw_to_windows_path('$CSV_OUTPUT')
    
    # Load exact match results
    exact_file = os.path.join(temp_dir, 'fdr_exact_$alpha.npy')
    exact_data = np.load(exact_file, allow_pickle=True).item()
    exact_lhat = np.mean(exact_data['lhats'])
    exact_fdr = np.mean(exact_data['risks'])
    
    # Load partial match results  
    partial_file = os.path.join(temp_dir, 'fdr_partial_$alpha.npy')
    partial_data = np.load(partial_file, allow_pickle=True).item()
    partial_fdr = np.mean(partial_data['risks'])
    
    # Write to CSV
    with open(csv_output, 'a') as f:
        f.write(f'$alpha,{exact_lhat},{exact_fdr},{partial_fdr}\n')
    
    print(f'  Results: lambda={exact_lhat:.6f}, exact_fdr={exact_fdr:.6f}, partial_fdr={partial_fdr:.6f}')
    
except Exception as e:
    print(f'Error processing alpha=$alpha: {e}', file=sys.stderr)
    sys.exit(1)
"
done

# Clean up temporary files
rm -rf "$TEMP_DIR"
echo "Results saved to: $CSV_OUTPUT"
echo "Total alpha values processed: $total"

