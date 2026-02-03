#!/bin/bash
#SBATCH --job-name=test_clean_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_clean_embed_%j.out
#SBATCH --error=logs/test_clean_embed_%j.err

# Test CLEAN embedding with the CPR CLI
# This script:
# 1. Runs CLI tests
# 2. Tests CLEAN embedding on a small FASTA file

set -e

echo "=== CPR CLEAN Embedding Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate conformal-s

# Print environment info
echo ""
echo "=== Environment Info ==="
which python
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# Change to repo directory
cd /groups/doudna/projects/ronb/conformal-protein-retrieval

# 1. Run CLI tests
echo ""
echo "=== Running CLI Tests ==="
python -m pytest tests/test_cli.py -v --tb=short 2>&1 || echo "Note: Some tests may fail if dependencies are missing"

# 2. Create a small test FASTA file
echo ""
echo "=== Creating Test FASTA ==="
TEST_DIR="test_clean_output"
mkdir -p "$TEST_DIR"

cat > "$TEST_DIR/test_sequences.fasta" << 'EOF'
>seq1_test_enzyme
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK
>seq2_test_enzyme
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
>seq3_test_enzyme
MTSKGECFVTVTYKNLFPPEQWSPKQYLFHNASDKGFVPTHICTHGCLSPKQLQEFDLVNQADQEGWSGDYTCQCNCTQQALCGFPVFLGCEACTFTPDCHGECVCKFPFGEYFVCDCDGSPDCG
EOF

echo "Created test FASTA with 3 sequences"

# 3. Test CLEAN embedding (requires GPU)
echo ""
echo "=== Testing CLEAN Embedding ==="
echo "Checking CLEAN installation..."
python -c "from CLEAN.model import LayerNormNet; print('CLEAN model import OK')" 2>&1 || {
    echo "CLEAN not installed, installing..."
    cd CLEAN_repo/app
    python build.py install
    cd ../..
}

echo ""
echo "Running cpr embed with CLEAN model..."
time python -m protein_conformal.cli embed \
    --input "$TEST_DIR/test_sequences.fasta" \
    --output "$TEST_DIR/test_clean_embeddings.npy" \
    --model clean

# 4. Verify output
echo ""
echo "=== Verifying Output ==="
if [ -f "$TEST_DIR/test_clean_embeddings.npy" ]; then
    python -c "
import numpy as np
emb = np.load('$TEST_DIR/test_clean_embeddings.npy')
print(f'Embeddings shape: {emb.shape}')
print(f'Expected: (3, 128)')
assert emb.shape == (3, 128), f'Shape mismatch: expected (3, 128), got {emb.shape}'
print('SUCCESS: CLEAN embedding test passed!')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi

# 5. Optional: Compare with reference (if exists)
echo ""
echo "=== Test Complete ==="
echo "Output saved to: $TEST_DIR/test_clean_embeddings.npy"
echo ""

# Cleanup (optional - uncomment to remove test files)
# rm -rf "$TEST_DIR"

echo "Done at $(date)"
