#!/bin/bash
#SBATCH --job-name=apptainer-build
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/apptainer_build_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/apptainer_build_%j.err

# IMPORTANT: Use $HOME2 for all caches to avoid disk quota issues
export HOME2=/groups/doudna/projects/ronb
export APPTAINER_CACHEDIR=$HOME2/.apptainer_cache
export APPTAINER_TMPDIR=$HOME2/tmp
export TMPDIR=$HOME2/tmp

# Create directories
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Change to project directory
cd /groups/doudna/projects/ronb/conformal-protein-retrieval

echo "============================================"
echo "Building Apptainer container for CPR"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Cache dir: $APPTAINER_CACHEDIR"
echo "Temp dir: $APPTAINER_TMPDIR"
echo ""

# Build the container
apptainer build --fakeroot cpr.sif apptainer.def

BUILD_STATUS=$?

echo ""
echo "============================================"
echo "Build completed with status: $BUILD_STATUS"
echo "End time: $(date)"
echo "============================================"

if [ $BUILD_STATUS -eq 0 ]; then
    echo "Container built successfully: $(ls -lh cpr.sif)"

    # Test the container
    echo ""
    echo "Testing container..."
    apptainer exec cpr.sif python --version
    apptainer exec cpr.sif python -c "import torch; print(f'PyTorch: {torch.__version__}')"
else
    echo "Build FAILED"
fi

exit $BUILD_STATUS
