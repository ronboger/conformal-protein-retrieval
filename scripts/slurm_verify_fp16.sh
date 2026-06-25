#!/bin/bash
#SBATCH --job-name=cpr-fp16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/fp16_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781/logs/fp16_%j.err

# fp16-vs-fp32 embedding verification on a GPU node. Embeds the 149 Syn3.0
# proteins both ways and checks the paper annotation (59/149) is preserved.

WORKTREE=/groups/doudna/projects/ronb/conformal-protein-retrieval/.claude/worktrees/sweet-gauss-646781
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE"
export TMPDIR=/groups/doudna/projects/ronb/tmp
export HF_HOME=/groups/doudna/projects/ronb/huggingface_cache
export HF_HUB_OFFLINE=1        # ProtTrans is cached; don't hit the network
export TOKENIZERS_PARALLELISM=false

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "============================================================"
python scripts/verify_fp16.py
RC=$?
echo "============================================================"
echo "verify_fp16 rc=$RC"
echo "End: $(date)"
exit $RC
