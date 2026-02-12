"""
Modal deployment for Conformal Protein Retrieval with GPU-accelerated embedding.

Architecture:
  - Gradio UI runs on CPU (single container, sticky sessions)
  - Protein embedding (ProtTrans + Protein-Vec) runs on GPU (autoscales 0..N)
  - Models and data are cached in a Modal Volume across cold starts

See MODAL_DEPLOY.md for full setup instructions.
"""

import modal

app = modal.App("cpr-gradio")

# Persistent volume — caches HF models, Protein-Vec checkpoints, and lookup data
volume = modal.Volume.from_name("cpr-data", create_if_missing=True)

VOLUME_PATH = "/vol"
HF_CACHE = f"{VOLUME_PATH}/hf_cache"
PVM_DIR = f"{VOLUME_PATH}/protein_vec_models"
VOL_DATA = f"{VOLUME_PATH}/data"

# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0,<4.44.0",
        "sentencepiece",
        "protobuf",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "biopython>=1.81",
        "h5py",
        "faiss-cpu>=1.7.4",
        "pytorch-lightning",
        "scikit-learn>=1.0.0",
        "huggingface_hub>=0.20.0",
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": "/app"})
    .pip_install(
        "fastapi[standard]",
        "gradio>=5.0.0",
        "pydantic==2.10.6",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.0.0",
        "biopython>=1.81",
        "matplotlib>=3.5.0",
        "plotly>=5.9.0",
        "seaborn>=0.12.0",
        "huggingface_hub>=0.20.0",
        "faiss-cpu>=1.7.4",
    )
    # Bake project source and small result files into the image
    .add_local_dir("protein_conformal", remote_path="/app/protein_conformal")
    .add_local_file("results/fdr_thresholds.csv", remote_path="/app/results/fdr_thresholds.csv")
    .add_local_file("results/fnr_thresholds.csv", remote_path="/app/results/fnr_thresholds.csv")
    .add_local_file("results/calibration_probs.csv", remote_path="/app/results/calibration_probs.csv")
    .add_local_file("results/fdr_thresholds_partial.csv", remote_path="/app/results/fdr_thresholds_partial.csv")
    .add_local_file("results/fnr_thresholds_partial.csv", remote_path="/app/results/fnr_thresholds_partial.csv")
    .add_local_file("data/gene_unknown/unknown_aa_seqs.fasta", remote_path="/app/bundled/syn30.fasta")
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HF_DATASET_ID = "LoocasGoose/cpr_data"


def _download_hf_file(repo_path: str, local_dir: str):
    """Download a single file from the HF dataset repo."""
    import os
    from huggingface_hub import hf_hub_download

    target = os.path.join(local_dir, repo_path)
    if os.path.exists(target):
        return target

    os.makedirs(os.path.dirname(target), exist_ok=True)
    hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        filename=repo_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return target


def _ensure_protein_vec_models():
    """Download and extract protein_vec_models from HF dataset into the volume."""
    import os
    import tarfile
    import shutil

    if os.path.isdir(PVM_DIR) and os.path.exists(os.path.join(PVM_DIR, "protein_vec.ckpt")):
        return

    from huggingface_hub import hf_hub_download

    dl_cache = f"{VOLUME_PATH}/.dl_cache"
    os.makedirs(dl_cache, exist_ok=True)

    arc_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        filename="protein_vec_models.tar.gz",
        local_dir=dl_cache,
        local_dir_use_symlinks=False,
    )

    os.makedirs(PVM_DIR, exist_ok=True)
    with tarfile.open(arc_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.join(PVM_DIR, member.name)
            if not os.path.abspath(member_path).startswith(os.path.abspath(PVM_DIR)):
                raise Exception(f"Unsafe tar path: {member.name}")
        tar.extractall(path=PVM_DIR)

    # Flatten single top-level directory if present
    contents = os.listdir(PVM_DIR)
    if len(contents) == 1:
        nested = os.path.join(PVM_DIR, contents[0])
        if os.path.isdir(nested):
            for entry in os.listdir(nested):
                shutil.move(os.path.join(nested, entry), os.path.join(PVM_DIR, entry))
            shutil.rmtree(nested, ignore_errors=True)


def _ensure_lookup_data():
    """Download large lookup data files into the volume."""
    data_files = [
        "data/lookup_embeddings.npy",
        "data/lookup_embeddings_meta_data.tsv",
        "data/lookup/scope_lookup_embeddings.npy",
        "data/lookup/scope_lookup.fasta",
        "data/gene_unknown/unknown_aa_seqs.fasta",
    ]
    for repo_path in data_files:
        try:
            _download_hf_file(repo_path, VOLUME_PATH)
        except Exception as e:
            print(f"Warning: could not download {repo_path}: {e}")


# ---------------------------------------------------------------------------
# GPU Embedding (runs on T4)
# ---------------------------------------------------------------------------

@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
class Embedder:
    """GPU-accelerated protein embedding using ProtTrans + Protein-Vec."""

    @modal.enter()
    def load_models(self):
        import torch
        import sys
        import gc
        import numpy as np
        from transformers import T5EncoderModel, T5Tokenizer

        _ensure_protein_vec_models()

        self.device = torch.device("cuda:0")

        # Load ProtTrans T5
        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            do_lower_case=False,
            cache_dir=HF_CACHE,
        )
        self.model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            cache_dir=HF_CACHE,
        )
        gc.collect()
        self.model = self.model.to(self.device).eval()

        # Load Protein-Vec MOE
        sys.path.insert(0, PVM_DIR)
        from model_protein_moe import trans_basic_block, trans_basic_block_Config

        config = trans_basic_block_Config.from_json(
            f"{PVM_DIR}/protein_vec_params.json"
        )
        self.model_deep = trans_basic_block.load_from_checkpoint(
            f"{PVM_DIR}/protein_vec.ckpt", config=config
        )
        self.model_deep = self.model_deep.to(self.device).eval()

        # Pre-compute masks (all aspects enabled)
        sampled_keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
        all_cols = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
        masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
        self.masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None, :]

        volume.commit()  # Persist cached downloads

    @modal.method()
    def embed(self, sequences: list) -> list:
        """Embed protein sequences on GPU. Returns list of lists (JSON-safe)."""
        import sys
        import numpy as np

        sys.path.insert(0, PVM_DIR)
        from utils_search import featurize_prottrans, embed_vec

        embeddings = []
        for seq in sequences:
            pt = featurize_prottrans([seq], self.model, self.tokenizer, self.device)
            emb = embed_vec(pt, self.model_deep, self.masks, self.device)
            embeddings.append(emb)

        result = np.concatenate(embeddings)
        return result.tolist()


# ---------------------------------------------------------------------------
# Gradio UI (runs on CPU, single container)
# ---------------------------------------------------------------------------

@app.function(
    image=web_image,
    max_containers=1,
    min_containers=0,
    scaledown_window=60 * 20,  # 20 min idle before shutdown
    volumes={VOLUME_PATH: volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Serve the Gradio interface, with embedding offloaded to GPU."""
    import os
    import sys
    import numpy as np
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    # Work from /app where project source is baked into the image
    os.chdir("/app")

    # Download large lookup data into the volume (cached across cold starts)
    _ensure_lookup_data()
    volume.commit()

    # Symlink /app/data -> /vol/data so Gradio finds files at ./data/
    if not os.path.exists("/app/data"):
        os.symlink(f"{VOLUME_PATH}/data", "/app/data")

    # Monkey-patch the embedding function to use the GPU Embedder
    import protein_conformal.backend.gradio_interface as gi

    def gpu_embed(sequences, progress=None):
        """Call Modal GPU function for protein embedding."""
        if progress:
            progress(0.1, desc="Sending sequences to GPU...")
        embedder = Embedder()
        result = embedder.embed.remote(sequences)
        if progress:
            progress(0.9, desc="Embeddings received from GPU!")
        return np.array(result, dtype=np.float32)

    gi.run_embed_protein_vec = gpu_embed

    # Create and serve the Gradio interface
    from protein_conformal.backend.gradio_interface import create_interface

    demo = create_interface()
    demo.queue(max_size=5)

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
