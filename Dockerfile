# Conformal Protein Retrieval (CPR)
# Docker image for functional protein mining with conformal guarantees
#
# Build: docker build -t cpr:latest .
# Run:   docker run -p 7860:7860 -v $(pwd)/data:/workspace/data cpr:latest

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Ron Boger <ronboger@berkeley.edu>"
LABEL description="Conformal Protein Retrieval - Functional protein mining with statistical guarantees"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    gradio>=4.0.0 \
    faiss-gpu \
    biopython \
    pytorch-lightning \
    h5py \
    transformers \
    sentencepiece

# Copy source code
COPY protein_conformal/ ./protein_conformal/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Create directories for data and results
RUN mkdir -p data results protein_vec_models

# Environment variables
ENV PYTHONPATH=/workspace
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose Gradio port
EXPOSE 7860

# Default command: run Gradio app
CMD ["python", "-m", "protein_conformal.gradio_app"]
