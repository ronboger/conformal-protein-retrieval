#!/bin/bash
# Submit FDR threshold jobs in parallel - one per alpha level

ALPHAS="0.001 0.005 0.01 0.02 0.05 0.1 0.15 0.2"

for alpha in $ALPHAS; do
    # Exact match
    sbatch --job-name="fdr-exact-${alpha}" \
           --partition=standard \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
           --time=08:00:00 \
           --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_exact_${alpha}_%j.log \
           --wrap="
eval \"\$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)\"
conda activate conformal-s
cd /groups/doudna/projects/ronb/conformal-protein-retrieval
python scripts/compute_fdr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/fdr_exact_alpha_${alpha}.csv \
    --n-trials 100 \
    --n-calib 1000 \
    --seed 42 \
    --alpha-levels ${alpha}
"
    
    # Partial match
    sbatch --job-name="fdr-partial-${alpha}" \
           --partition=standard \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
           --time=08:00:00 \
           --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_partial_${alpha}_%j.log \
           --wrap="
eval \"\$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)\"
conda activate conformal-s
cd /groups/doudna/projects/ronb/conformal-protein-retrieval
python scripts/compute_fdr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/fdr_partial_alpha_${alpha}.csv \
    --n-trials 100 \
    --n-calib 1000 \
    --seed 42 \
    --alpha-levels ${alpha} \
    --partial
"
done

echo "Submitted 16 FDR jobs (8 alphas × 2 match types)"
