# 1. Base image: Ubuntu 22.04
FROM ubuntu:22.04

# 2. Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 3. System dependencies
RUN apt-get update && apt-get install -y \
      wget bzip2 ca-certificates git \
      libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
 && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# 5. Create a working dir and copy only environment spec
WORKDIR /workspace
COPY environment.yml /workspace/

# Pre-accept Anaconda channel Terms of Service
RUN conda tos accept \
      --override-channels \
      --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept \
      --override-channels \
      --channel https://repo.anaconda.com/pkgs/r

# Create the env and clean up
RUN conda env create -f environment.yml && \
    conda run -n protein-conformal pip install --force-reinstall gradio>=4.0.0 && \
    conda clean -afy

# 7. Copy the rest of your code
COPY . /workspace/

# 8. Activate env by default
SHELL ["conda", "run", "-n", "protein-conformal", "/bin/bash", "-c"]

# # 9. Expose Gradio port
EXPOSE 7860

# # 10. Default command: start your Gradio app 
# SHELL ["conda", "run", "-n", "protein-conformal", "/bin/bash", "-c"]
# # CMD python -m protein_conformal.gradio_app --share --host 0.0.0.0