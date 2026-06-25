#!/bin/bash
#SBATCH --job-name=cpr-resident
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/resident_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/resident_%j.err

# GPU-resident-vs-original embedding verification. Confirms the new GPU-resident
# embed preserves Syn3.0 results (59/149, same hit set) before it ships.

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
python scripts/verify_gpu_resident.py
RC=$?
echo "verify_gpu_resident rc=$RC"
echo "End: $(date)"
exit $RC
