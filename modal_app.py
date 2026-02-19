"""
Modal deployment for Conformal Protein Retrieval with GPU-accelerated embedding.

Architecture:
  - Gradio UI runs on CPU (single container, sticky sessions)
  - Protein embedding (ProtTrans + Protein-Vec) runs on GPU (autoscales 0..N)
  - Models and data are cached in a Modal Volume across cold starts

See MODAL_DEPLOY.md for full setup instructions.
"""

import os

import modal

app = modal.App("cpr-gradio")

# Persistent volume — caches HF models, Protein-Vec checkpoints, and lookup data
volume = modal.Volume.from_name("cpr-data", create_if_missing=True)

VOLUME_PATH = "/vol"
HF_CACHE = f"{VOLUME_PATH}/hf_cache"
PVM_DIR = f"{VOLUME_PATH}/protein_vec_models"
VOL_DATA = f"{VOLUME_PATH}/data"
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "LoocasGoose/cpr_data")
dataset_config_secret = modal.Secret.from_dict({"HF_DATASET_ID": HF_DATASET_ID})

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

# Image for CLEAN embedder (ESM-1b + LayerNormNet)
clean_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "fair-esm",
        "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
    )
    .add_local_file(
        "CLEAN_repo/app/data/pretrained/split100.pth",
        remote_path="/app/bundled/split100.pth",
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
    .add_local_file("results/clean_thresholds.csv", remote_path="/app/results/clean_thresholds.csv")
    .add_local_file("data/clean/ec_centroid_embeddings.npy", remote_path="/app/bundled/clean/ec_centroid_embeddings.npy")
    .add_local_file("data/clean/ec_centroid_metadata.tsv", remote_path="/app/bundled/clean/ec_centroid_metadata.tsv")
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_volume_data():
    """Verify required data files exist on the Modal volume.

    All data is pre-uploaded via `modal volume put cpr-data`.
    This function only checks — it does not download anything.
    """
    import os

    required = [
        "protein_vec_models/protein_vec.ckpt",
        "data/lookup_embeddings.npy",
        "data/lookup_embeddings_meta_data.tsv",
        "data/lookup/scope_lookup_embeddings.npy",
        "data/lookup/scope_lookup.fasta",
    ]
    optional = [
        "data/gene_unknown/unknown_aa_seqs.fasta",
        "data/afdb/afdb_embeddings_protein_vec.npy",
        "data/afdb/afdb_metadata.tsv",
    ]

    missing_required = []
    for path in required:
        full = os.path.join(VOLUME_PATH, path)
        if not os.path.exists(full):
            missing_required.append(path)

    for path in optional:
        full = os.path.join(VOLUME_PATH, path)
        if not os.path.exists(full):
            print(f"Optional data missing: {path}")

    if missing_required:
        print(f"WARNING: Required data missing from volume: {missing_required}")
        print("Upload with: modal volume put cpr-data <local_path> <remote_path>")


# ---------------------------------------------------------------------------
# GPU Embedding (runs on T4)
# ---------------------------------------------------------------------------

@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=600,
    volumes={VOLUME_PATH: volume},
    secrets=[dataset_config_secret],
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

        # Protein-Vec models must be pre-uploaded to the volume
        if not os.path.exists(os.path.join(PVM_DIR, "protein_vec.ckpt")):
            raise RuntimeError(f"Protein-Vec models not found at {PVM_DIR}. Upload with: modal volume put cpr-data")

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
# CLEAN Enzyme Embedding (ESM-1b + LayerNormNet, runs on A10G)
# ---------------------------------------------------------------------------

CLEAN_MODEL_PATH = "/app/bundled/split100.pth"


@app.cls(
    image=clean_gpu_image,
    gpu="A10G",
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
class CLEANEmbedder:
    """GPU-accelerated enzyme embedding using ESM-1b + CLEAN LayerNormNet."""

    @modal.enter()
    def load_models(self):
        import torch
        import torch.nn as nn
        import esm

        self.device = torch.device("cuda:0")

        # Load ESM-1b
        self.esm_model, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.esm_model = self.esm_model.to(self.device).eval()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

        # Build CLEAN LayerNormNet (1280 -> 512 -> 512 -> 128)
        class LayerNormNet(nn.Module):
            def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
                super().__init__()
                self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
                self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
                self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
                self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
                self.dropout = nn.Dropout(p=drop_out)

            def forward(self, x):
                x = self.dropout(self.ln1(self.fc1(x)))
                x = torch.relu(x)
                x = self.dropout(self.ln2(self.fc2(x)))
                x = torch.relu(x)
                x = self.fc3(x)
                return x

        self.clean_model = LayerNormNet(512, 128, self.device, torch.float32)
        checkpoint = torch.load(CLEAN_MODEL_PATH, map_location=self.device)
        self.clean_model.load_state_dict(checkpoint)
        self.clean_model.eval()

    @modal.method()
    def embed(self, sequences: list) -> list:
        """Embed protein sequences using ESM-1b + CLEAN. Returns list of lists."""
        import torch
        import numpy as np

        # Prepare ESM-1b input
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.esm_batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33])
            # Mean representation over sequence length (layer 33)
            token_reps = results["representations"][33]
            # Average over sequence positions (exclude BOS/EOS tokens)
            esm_embeddings = token_reps[:, 1:-1, :].mean(dim=1)  # (N, 1280)

            # Pass through CLEAN LayerNormNet
            clean_embeddings = self.clean_model(esm_embeddings)  # (N, 128)

        return clean_embeddings.cpu().numpy().tolist()


# ---------------------------------------------------------------------------
# Gradio UI (runs on CPU, single container)
# ---------------------------------------------------------------------------

@app.function(
    image=web_image,
    max_containers=1,
    min_containers=0,
    scaledown_window=60 * 20,  # 20 min idle before shutdown
    volumes={VOLUME_PATH: volume},
    secrets=[dataset_config_secret],
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

    # Verify data files exist on the volume (pre-uploaded via `modal volume put`)
    _check_volume_data()

    # Symlink /app/data -> /vol/data so Gradio finds files at ./data/
    if not os.path.exists("/app/data"):
        os.symlink(f"{VOLUME_PATH}/data", "/app/data")

    # Copy bundled CLEAN centroid files into the data dir
    # (baked to /app/bundled/clean/ to avoid blocking the /app/data symlink)
    import shutil
    os.makedirs("/app/data/clean", exist_ok=True)
    for fname in ["ec_centroid_embeddings.npy", "ec_centroid_metadata.tsv"]:
        src = f"/app/bundled/clean/{fname}"
        dst = f"/app/data/clean/{fname}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

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

    def gpu_embed_clean(sequences, progress=None):
        """Call Modal GPU function for CLEAN enzyme embedding."""
        if progress:
            progress(0.1, desc="Sending sequences to GPU (ESM-1b + CLEAN)...")
        embedder = CLEANEmbedder()
        result = embedder.embed.remote(sequences)
        if progress:
            progress(0.9, desc="CLEAN embeddings received from GPU!")
        return np.array(result, dtype=np.float32)

    gi.run_embed_clean = gpu_embed_clean

    # Create and serve the Gradio interface
    from protein_conformal.backend.gradio_interface import create_interface

    demo = create_interface()
    demo.queue(max_size=5)

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
