# SLURM Job Scripts

Quick reference for submitting jobs to the cluster.

## Available Jobs

| Script | Purpose | Resources | Usage |
|--------|---------|-----------|-------|
| `slurm_verify.sh` | Verify paper results | 32G RAM, 1hr | `sbatch scripts/slurm_verify.sh [syn30\|fdr\|dali\|all]` |
| `slurm_embed.sh` | Embed FASTA sequences | 64G RAM, GPU, 4hr | `sbatch scripts/slurm_embed.sh input.fasta output.npy` |
| `slurm_calibrate_fdr.sh` | Compute FDR thresholds | 32G RAM, 2hr | `sbatch scripts/slurm_calibrate_fdr.sh` |

## Verification Options

- `syn30` - JCVI Syn3.0 annotation (Paper Figure 2A: 59/149 = 39.6%)
- `fdr` - FDR algorithm verification
- `dali` - DALI prefiltering (Tables 4-6: 82.8% TPR, 31.5% DB reduction)
- `clean` - CLEAN enzyme classification (Tables 1-2: hierarchical loss control)
- `all` - Run all verifications

Note: Full CLEAN verification with precision/recall metrics requires the CLEAN package
from https://github.com/tttianhao/CLEAN. The basic verification uses pre-computed data.

## Quick Commands

```bash
# Check job status
squeue -u $USER

# View job output (use Read tool or cat, avoid tail -f on login node)
cat logs/cpr-verify-JOBID.out

# Cancel a job
scancel JOBID

# Submit verification jobs
sbatch scripts/slurm_verify.sh syn30
sbatch scripts/slurm_verify.sh dali
sbatch scripts/slurm_verify.sh all

# Submit other jobs
sbatch scripts/slurm_embed.sh my_sequences.fasta my_embeddings.npy
sbatch scripts/slurm_calibrate_fdr.sh
```

## Output

All jobs write to `logs/` directory:
- `logs/cpr-JOB-JOBID.out` - stdout
- `logs/cpr-JOB-JOBID.err` - stderr
