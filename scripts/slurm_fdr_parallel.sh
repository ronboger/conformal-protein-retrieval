#!/bin/bash
# Submit parallel FDR threshold jobs — one per alpha level per match type.
# Each job computes 20 trials for a single alpha and writes its own CSV.
# After all complete, run: scripts/combine_fdr_results.sh
#
# Usage: bash scripts/slurm_fdr_parallel.sh

set -euo pipefail

PROJDIR=/groups/doudna/projects/ronb/conformal-protein-retrieval
CALDATA=$PROJDIR/data/pfam_new_proteins.npy
OUTDIR=$PROJDIR/results/fdr_parts
LOGDIR=$PROJDIR/logs
N_TRIALS=20
N_CALIB=1000
SEED=42

mkdir -p "$OUTDIR" "$LOGDIR"

ALPHAS=(0.001 0.005 0.01 0.02 0.05 0.1 0.15 0.2)

for ALPHA in "${ALPHAS[@]}"; do
    for MATCH in exact partial; do
        PARTIAL_FLAG=""
        if [ "$MATCH" = "partial" ]; then
            PARTIAL_FLAG="--partial"
        fi

        JOBNAME="fdr-${MATCH}-${ALPHA}"
        OUTFILE="$OUTDIR/fdr_${MATCH}_${ALPHA}.csv"

        sbatch --job-name="$JOBNAME" \
               --partition=standard \
               --nodes=1 --ntasks=1 --cpus-per-task=4 \
               --mem=32G \
               --time=8:00:00 \
               --output="$LOGDIR/${JOBNAME}_%j.log" \
               --error="$LOGDIR/${JOBNAME}_%j.err" \
               --wrap="
eval \"\$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)\"
conda activate conformal-s
cd $PROJDIR
echo \"Computing FDR: match=$MATCH alpha=$ALPHA trials=$N_TRIALS\"
echo \"Start: \$(date)\"
python scripts/compute_fdr_table.py \\
    --calibration $CALDATA \\
    --output $OUTFILE \\
    --n-trials $N_TRIALS \\
    --n-calib $N_CALIB \\
    --seed $SEED \\
    --alpha-levels $ALPHA \\
    $PARTIAL_FLAG
echo \"Done: \$(date)\"
"
        echo "Submitted $JOBNAME -> $OUTFILE"
    done
done

echo ""
echo "Submitted ${#ALPHAS[@]}x2 = $((${#ALPHAS[@]} * 2)) jobs."
echo "When all complete, run: bash scripts/combine_fdr_results.sh"
