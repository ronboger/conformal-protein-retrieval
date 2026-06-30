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


def _rewrite_social_preview_html(html: str, base_url: str) -> str:
    """Remove Gradio's default Open Graph/Twitter preview tags and insert ours.

    Gradio emits tags like og:title=Gradio and og:description=Click to try out
    the app before custom head content. Link unfurlers usually read the first
    tags, so on Modal we rewrite the HTML response server-side.
    """
    import re

    # Remove Gradio/default social preview tags, plus duplicate custom tags that
    # Gradio may emit later with blank descriptions.
    patterns = [
        r'<meta\s+(?:property|name)=["\'](?:og:title|og:description|og:image|og:url|twitter:title|twitter:description|twitter:image|twitter:creator)["\'][^>]*>\s*',
        r'<meta\s+[^>]*(?:property|name)=["\'](?:og:title|og:description|og:image|og:url|twitter:title|twitter:description|twitter:image|twitter:creator)["\'][^>]*>\s*',
    ]
    for pattern in patterns:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE)

    base = base_url.rstrip("/")
    image = f"{base}/favicon.ico"
    title = "Conformal Protein Retrieval"
    desc = "Functional protein mining with statistical guarantees."
    tags = f"""
<title>{title}</title>
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:image" content="{image}">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:card" content="summary">
<meta name="twitter:image" content="{image}">
"""
    if "<head>" in html:
        return html.replace("<head>", "<head>" + tags, 1)
    return tags + html


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # NOTE: pinning to the cluster's torch 2.1 + cu121 + cudnn8 stack was tested
        # and did NOT close the Modal-vs-cluster embed gap: modal_diagnose.py on the
        # *exact* cluster stack still measured 80.6 ms/seq (t5 56.3 + head 23.5),
        # vs the cluster A5000's ~35 ms. The gap is GPU compute (A10G has ~half the
        # A5000's fp16 tensor throughput), not the software stack, so torch stays
        # unpinned. Only a bigger GPU (A100) would help. (Pinning torch==2.1.0 also
        # forces numpy<2, else tensor.numpy() raises "Numpy is not available".)
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
        "data/euk/euk_embeddings_protein_vec.npy",
        "data/euk/euk_metadata.tsv",
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
    gpu="A10G",  # ~4x faster than T4 for the per-sequence T5 forwards (Syn3.0 = 149);
                 # scales to zero, so ~cost-neutral per search (faster offsets higher rate).
    # NOTE: cpu= left at the default. modal_diagnose.py showed the A10G container
    # already sees 17 vCPUs unset, and tokenize+h2d are only ~0.9 ms/seq of an 80
    # ms/seq embed — the embed path is GPU-bound, not CPU-starved, so pinning cpu=4
    # bought nothing (the per-seq cost is t5 56 ms + Protein-Vec head 23 ms).
    timeout=600,
    # Snapshotting the ~11 GB fp32 ProtTrans model needs headroom or creation
    # OOM-kills (exit 137) and falls back to a slow full reload on every cold start.
    memory=32768,
    volumes={VOLUME_PATH: volume},
    secrets=[dataset_config_secret],
    enable_memory_snapshot=True,
    # NOTE: experimental GPU memory snapshot (enable_gpu_snapshot) segfaults on
    # restore (exit 139) on this config; removed. The CPU snapshot needs the
    # memory above to create successfully.
)
class Embedder:
    """GPU-accelerated protein embedding using ProtTrans + Protein-Vec.

    Cold start is shrunk with a Modal memory snapshot: the expensive model load
    (reading ~3B-param ProtTrans + Protein-Vec from the volume into CPU memory)
    runs once during snapshot creation; cold restores skip it and only move the
    already-loaded weights onto the GPU.
    """

    @modal.enter(snap=True)
    def load_models(self):
        """Load models during CPU memory-snapshot creation (no GPU attached here).

        Models load on CPU and the snapshot captures that state; ensure_gpu() then
        moves them onto the GPU on each (cold or snapshot-restored) start. The
        device line also works if a GPU happens to be present.
        """
        import torch
        import sys
        import gc
        import numpy as np
        from transformers import T5EncoderModel, T5Tokenizer

        # Protein-Vec models must be pre-uploaded to the volume
        if not os.path.exists(os.path.join(PVM_DIR, "protein_vec.ckpt")):
            raise RuntimeError(f"Protein-Vec models not found at {PVM_DIR}. Upload with: modal volume put cpr-data")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load ProtTrans T5
        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            do_lower_case=False,
            cache_dir=HF_CACHE,
        )
        self.model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            cache_dir=HF_CACHE,
        ).to(self.device).eval()
        gc.collect()

        # Load Protein-Vec MOE
        sys.path.insert(0, PVM_DIR)
        from model_protein_moe import trans_basic_block, trans_basic_block_Config

        config = trans_basic_block_Config.from_json(
            f"{PVM_DIR}/protein_vec_params.json"
        )
        self.model_deep = trans_basic_block.load_from_checkpoint(
            f"{PVM_DIR}/protein_vec.ckpt", config=config, map_location=self.device
        ).to(self.device).eval()

        # Pre-compute masks (all aspects enabled). Kept on CPU, as before.
        sampled_keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
        all_cols = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
        masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
        self.masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None, :]

        volume.commit()  # Persist cached downloads

    @modal.enter(snap=False)
    def ensure_gpu(self):
        """Ensure models are on the GPU after restore (no-op if GPU snapshot kept them there)."""
        import torch

        if self.device.type != "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model = self.model.to(self.device)
            self.model_deep = self.model_deep.to(self.device)

    @modal.method()
    def embed(self, sequences: list, fp16_head: bool = False) -> list:
        """Embed protein sequences on GPU. Returns list of lists (JSON-safe).

        GPU-resident path: the T5 output stays on the GPU through embed_vec, instead
        of the per-sequence GPU->CPU->GPU round-trip inside
        utils_search.featurize_prottrans (which was written for single-protein
        convenience, not throughput). fp16 autocast on the large ProtTrans T5; the
        Protein-Vec head stays fp32 by default (verified bit-identical, Syn3.0 59/149).

        fp16_head=True (EXPERIMENTAL, opt-in via the UI's Advanced Options) also runs
        the head in fp16 -- cuDNN SDPA excluded, since it raises "no execution plan"
        under fp16 on the A10G. ~1.25x on the head, but can flip a borderline match
        across the FDR threshold (Syn3.0 -> 60/149), so it is NOT the default and not
        for paper reproduction.
        """
        import re
        import numpy as np
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend

        from utils_search import embed_vec  # PVM_DIR already on sys.path from load_models

        # SDPA backends for the (opt-in) fp16 head: anything but cuDNN.
        head_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

        embeddings = []
        with torch.no_grad():
            for seq in sequences:
                # Replicate featurize_prottrans tokenization (space-separated residues,
                # rare residues -> X, truncate to 3000), but keep everything on GPU.
                prepped = re.sub(r"[UZOB]", "X", " ".join(seq[:3000]))
                ids = self.tokenizer.batch_encode_plus(
                    [prepped], add_special_tokens=True, padding=True
                )
                input_ids = torch.tensor(ids["input_ids"]).to(self.device)
                attn = torch.tensor(ids["attention_mask"]).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = self.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
                seq_len = int(attn[0].sum())
                pt = hidden[0, : seq_len - 1].float().unsqueeze(0)  # stays on GPU
                if fp16_head:
                    with torch.autocast("cuda", dtype=torch.float16), sdpa_kernel(head_backends):
                        emb = embed_vec(pt, self.model_deep, self.masks, self.device)
                else:
                    emb = embed_vec(pt, self.model_deep, self.masks, self.device)
                embeddings.append(np.asarray(emb, dtype=np.float32))

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

        # ESM-1b supports max 1022 tokens; truncate longer sequences
        MAX_ESM_LEN = 1022
        sequences = [seq[:MAX_ESM_LEN] for seq in sequences]

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
    import logging
    import numpy as np
    from fastapi import FastAPI, Response
    from gradio.routes import mount_gradio_app

    # Surface StageTimer (gradio_interface logs per-stage durations at INFO) in
    # Modal logs so we can see the embed/faiss/packaging breakdown.
    logging.basicConfig(level=logging.INFO, force=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    # Work from /app where project source is baked into the image
    os.chdir("/app")

    # Verify data files exist on the volume (pre-uploaded via `modal volume put`)
    _check_volume_data()

    # Symlink /app/data -> /vol/data so Gradio finds files at ./data/
    try:
        os.symlink(f"{VOLUME_PATH}/data", "/app/data")
    except FileExistsError:
        pass

    # Copy bundled CLEAN centroid files into the data dir
    # (baked to /app/bundled/clean/ to avoid blocking the /app/data symlink)
    import shutil
    os.makedirs("/app/data/clean", exist_ok=True)
    for fname in ["ec_centroid_embeddings.npy", "ec_centroid_metadata.tsv"]:
        src = f"/app/bundled/clean/{fname}"
        dst = f"/app/data/clean/{fname}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # GPU embedding functions are passed explicitly into create_interface()
    # below. Local runs keep the default subprocess/CPU embedder.
    GPU_TIMEOUT = 600  # seconds — headroom for up to MAX_QUERY_SEQUENCES on A10G;
                       # still fails loudly rather than hanging forever (matches cls timeout)

    FANOUT_THRESHOLD = 600  # at/below this, one container (fan-out's per-container
                            # cold start would dominate); above, split across GPUs
    FANOUT_CHUNK = 400      # sequences per container when fanning out

    def gpu_embed(sequences, progress=None, fp16_head=False):
        """Call Modal GPU function for protein embedding.

        Genome-scale inputs fan out across containers: split into chunks, embed in
        parallel, concatenate in order. Small inputs use a single container.

        fp16_head (EXPERIMENTAL, opt-in) runs the Protein-Vec head in fp16 too -- see
        Embedder.embed; can change borderline conformal results, so it's default off.
        """
        if progress:
            progress(0.1, desc="Sending sequences to GPU...")
        embedder = Embedder()
        if len(sequences) <= FANOUT_THRESHOLD:
            chunks = [sequences]
        else:
            chunks = [sequences[i:i + FANOUT_CHUNK] for i in range(0, len(sequences), FANOUT_CHUNK)]
            if progress:
                progress(0.2, desc=f"Embedding {len(sequences)} sequences across {len(chunks)} GPUs...")
        try:
            # spawn() returns immediately, so all chunks run in parallel; get()
            # gathers them in submission (sequence) order.
            futures = [embedder.embed.spawn(c, fp16_head) for c in chunks]
            parts = [np.array(f.get(timeout=GPU_TIMEOUT), dtype=np.float32) for f in futures]
            arr = parts[0] if len(parts) == 1 else np.concatenate(parts)
        except TimeoutError:
            raise TimeoutError(f"Protein-Vec embedding timed out after {GPU_TIMEOUT}s. Try fewer/shorter sequences.")
        if progress:
            progress(0.9, desc="Embeddings received from GPU!")
        return arr

    def gpu_embed_clean(sequences, progress=None):
        """Call Modal GPU function for CLEAN enzyme embedding."""
        if progress:
            progress(0.1, desc="Sending sequences to GPU (ESM-1b + CLEAN)...")
        embedder = CLEANEmbedder()
        future = embedder.embed.spawn(sequences)
        try:
            result = future.get(timeout=GPU_TIMEOUT)
        except TimeoutError:
            raise TimeoutError(f"CLEAN embedding timed out after {GPU_TIMEOUT}s. Try fewer/shorter sequences.")
        if progress:
            progress(0.9, desc="CLEAN embeddings received from GPU!")
        return np.array(result, dtype=np.float32)

    # Create and serve the Gradio interface with explicit GPU-backed embedders.
    from protein_conformal.backend.gradio_interface import create_interface
    from protein_conformal.gradio_app import _CUSTOM_HEAD, _FAVICON

    demo = create_interface(embed_fn=gpu_embed, clean_embed_fn=gpu_embed_clean)
    demo.queue(max_size=10, default_concurrency_limit=5)

    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def social_preview_middleware(request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if request.url.path == "/" and "text/html" in content_type:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            html = body.decode("utf-8", errors="replace")
            html = _rewrite_social_preview_html(html, str(request.base_url))
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return Response(
                content=html,
                status_code=response.status_code,
                headers=headers,
                media_type="text/html",
            )
        return response

    return mount_gradio_app(
        app=fastapi_app,
        blocks=demo,
        path="/",
        footer_links=[],
        favicon_path=_FAVICON,
        head=_CUSTOM_HEAD,
        theme=getattr(demo, "cpr_theme", None),
        css=getattr(demo, "cpr_css", None),
        app_kwargs=dict(openapi_url=None, docs_url=None, redoc_url=None),
    )
