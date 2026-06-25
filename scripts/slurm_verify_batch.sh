#!/bin/bash
#SBATCH --job-name=cpr-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/batch_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/batch_%j.err

# Batched-vs-per-sequence embedding verification on a GPU node. Embeds the 149
# Syn3.0 proteins both ways, checks results are preserved (59/149, same hit set)
# and reports the speedup.

WORKTREE=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE"
export TMPDIR=/groups/doudna/projects/ronb/tmp
export HF_HOME=/groups/doudna/projects/ronb/huggingface_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "============================================================"
python scripts/verify_batch.py
RC=$?
echo "============================================================"
echo "verify_batch rc=$RC"
echo "End: $(date)"
exit $RC
