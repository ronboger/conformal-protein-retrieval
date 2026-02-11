#!/bin/bash
#SBATCH --job-name=fdr-finalize
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_finalize_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_finalize_%j.err

# Runs after all FDR parallel jobs complete.
# Combines results, pulls latest changes, updates Gradio, and redeploys Modal.

set -euo pipefail

PROJDIR=/groups/doudna/projects/ronb/conformal-protein-retrieval
cd "$PROJDIR"

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "=== FDR Finalize ==="
echo "Start: $(date)"
echo ""

# 1. Verify all 16 per-alpha CSVs exist
echo "--- Step 1: Verify per-alpha results ---"
MISSING=0
for ALPHA in 0.001 0.005 0.01 0.02 0.05 0.1 0.15 0.2; do
    for MATCH in exact partial; do
        F="results/fdr_parts/fdr_${MATCH}_${ALPHA}.csv"
        if [ ! -s "$F" ]; then
            echo "MISSING: $F"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: $MISSING result files missing. Aborting."
    exit 1
fi
echo "All 16 per-alpha CSVs present."
echo ""

# 2. Combine into final CSVs
echo "--- Step 2: Combine results ---"
bash scripts/combine_fdr_results.sh
echo ""

# 3. Pull latest changes from remote
echo "--- Step 3: Pull latest changes ---"
git stash --include-untracked || true
git pull --rebase origin gradio
git stash pop || true
echo ""

# 4. Update FDR_ALPHAS in gradio_interface.py if still [0.1]
echo "--- Step 4: Update FDR_ALPHAS ---"
IFILE="protein_conformal/backend/gradio_interface.py"
if grep -q 'FDR_ALPHAS = \[0.1\]' "$IFILE"; then
    sed -i 's/FDR_ALPHAS = \[0.1\]/FDR_ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]/' "$IFILE"
    echo "Updated FDR_ALPHAS to all 8 levels."
else
    echo "FDR_ALPHAS already updated (or changed upstream), skipping."
fi
echo ""

# 5. Copy updated threshold CSVs so Modal image bakes them in
echo "--- Step 5: Verify results files for Modal ---"
ls -l results/fdr_thresholds.csv results/fdr_thresholds_partial.csv
echo ""

# 6. Deploy to Modal
echo "--- Step 6: Deploy to Modal ---"
modal deploy modal_app.py
echo ""

echo "=== Done: $(date) ==="
echo "App should be live at: https://doudna-lab--cpr-gradio-ui.modal.run"
