#!/bin/bash
# Combine per-alpha FDR results into the final threshold CSVs.
# Run after all slurm_fdr_parallel.sh jobs complete.
#
# Usage: bash scripts/combine_fdr_results.sh

set -euo pipefail

PROJDIR=/groups/doudna/projects/ronb/conformal-protein-retrieval
PARTSDIR=$PROJDIR/results/fdr_parts

# Combine exact match results
echo "Combining exact match FDR thresholds..."
HEADER=""
for f in "$PARTSDIR"/fdr_exact_*.csv; do
    if [ -z "$HEADER" ]; then
        HEADER=$(head -1 "$f")
        echo "$HEADER" > "$PROJDIR/results/fdr_thresholds.csv"
    fi
    tail -n +2 "$f" >> "$PROJDIR/results/fdr_thresholds.csv"
done
sort -t, -k1 -n -o "$PROJDIR/results/fdr_thresholds.csv" <(head -1 "$PROJDIR/results/fdr_thresholds.csv") <(tail -n +2 "$PROJDIR/results/fdr_thresholds.csv")
echo "  -> results/fdr_thresholds.csv"
cat "$PROJDIR/results/fdr_thresholds.csv"

# Combine partial match results
echo ""
echo "Combining partial match FDR thresholds..."
HEADER=""
for f in "$PARTSDIR"/fdr_partial_*.csv; do
    if [ -z "$HEADER" ]; then
        HEADER=$(head -1 "$f")
        echo "$HEADER" > "$PROJDIR/results/fdr_thresholds_partial.csv"
    fi
    tail -n +2 "$f" >> "$PROJDIR/results/fdr_thresholds_partial.csv"
done
sort -t, -k1 -n -o "$PROJDIR/results/fdr_thresholds_partial.csv" <(head -1 "$PROJDIR/results/fdr_thresholds_partial.csv") <(tail -n +2 "$PROJDIR/results/fdr_thresholds_partial.csv")
echo "  -> results/fdr_thresholds_partial.csv"
cat "$PROJDIR/results/fdr_thresholds_partial.csv"

echo ""
echo "Done. Update the Gradio FDR_ALPHAS list to include all alpha levels."
