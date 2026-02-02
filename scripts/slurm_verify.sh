#!/bin/bash
#SBATCH --job-name=cpr-verify
#SBATCH --output=logs/cpr-verify-%j.out
#SBATCH --error=logs/cpr-verify-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# CPR Verification - Reproduces paper results
# Usage:
#   sbatch scripts/slurm_verify.sh syn30   # Verify Syn3.0 (Figure 2A)
#   sbatch scripts/slurm_verify.sh fdr     # Verify FDR algorithm
#   sbatch scripts/slurm_verify.sh dali    # Verify DALI prefiltering (Tables 4-6)
#   sbatch scripts/slurm_verify.sh clean   # Verify CLEAN enzyme (Tables 1-2)
#   sbatch scripts/slurm_verify.sh all     # Run all verifications

set -e
mkdir -p logs results

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

CHECK="${1:-all}"

echo "========================================"
echo "CPR Verification"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Check: $CHECK"
echo "========================================"
echo ""

run_syn30() {
    echo "--- Syn3.0 Verification (Paper Figure 2A) ---"
    python scripts/verify_syn30.py
    echo ""
}

run_fdr() {
    echo "--- FDR Algorithm Verification ---"
    python scripts/verify_fdr_algorithm.py
    echo ""
}

run_dali() {
    echo "--- DALI Prefiltering Verification (Tables 4-6) ---"
    python scripts/verify_dali.py
    echo ""
}

run_clean() {
    echo "--- CLEAN Enzyme Classification Verification (Tables 1-2) ---"
    python scripts/verify_clean.py
    echo ""
}

case "$CHECK" in
    syn30)
        run_syn30
        ;;
    fdr)
        run_fdr
        ;;
    dali)
        run_dali
        ;;
    clean)
        run_clean
        ;;
    all)
        run_syn30
        run_fdr
        run_dali
        run_clean
        ;;
    *)
        echo "Unknown check: $CHECK"
        echo "Available: syn30, fdr, dali, clean, all"
        exit 1
        ;;
esac

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
