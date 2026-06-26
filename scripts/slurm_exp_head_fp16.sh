#!/bin/bash
#SBATCH --job-name=cpr-headfp16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/headfp16_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/headfp16_%j.err

# Tries fp16 on the Protein-Vec head with different SDPA backends; reports
# which run, their ms/seq vs the fp32 head, and cosine vs fp32 (must be ~1.0).

REPO=/groups/doudna/projects/ronb/conformal-protein-retrieval
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR=/groups/doudna/projects/ronb/tmp
export HF_HOME=/groups/doudna/projects/ronb/huggingface_cache
export HF_HUB_OFFLINE=1        # ProtTrans is cached; don't hit the network
export TOKENIZERS_PARALLELISM=false

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "============================================================"
python scripts/exp_head_fp16.py
RC=$?
echo "============================================================"
echo "exp_head_fp16 rc=$RC"
echo "End: $(date)"
exit $RC
