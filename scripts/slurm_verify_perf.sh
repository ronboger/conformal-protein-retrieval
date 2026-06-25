#!/bin/bash
#SBATCH --job-name=cpr-verify-perf
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/verify_perf_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/verify_perf_%j.err

# Verifies the perf changes don't alter results:
#   1. verify_syn30 (paper Figure 2A headline number, expect 39.6% = 59/149)
#   2. verify_prebuild_equiv (prebuilt index == load_database on real Swiss-Prot)
# Runs the WORKTREE code against the REAL data in the main checkout.

WORKTREE=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781
DATA_DIR=/groups/doudna/projects/ronb/conformal-protein-retrieval/data

cd "$WORKTREE"
export PYTHONPATH="$WORKTREE"
export TMPDIR=/groups/doudna/projects/ronb/tmp

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "Start: $(date)  Node: $(hostname)  Python: $(which python)"
echo "protein_conformal from: $(python -c 'import protein_conformal,os;print(os.path.dirname(protein_conformal.__file__))')"
echo "==================== verify_syn30 (alpha=0.1) ===================="
python scripts/verify_syn30.py --data-dir "$DATA_DIR" --alpha 0.1
SYN30_RC=$?
echo "==================== verify_prebuild_equiv ===================="
python scripts/verify_prebuild_equiv.py "$DATA_DIR"
EQUIV_RC=$?
echo "================================================================"
echo "verify_syn30 rc=$SYN30_RC   verify_prebuild_equiv rc=$EQUIV_RC"
echo "End: $(date)"
exit $(( SYN30_RC + EQUIV_RC ))
