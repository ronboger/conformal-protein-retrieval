# Deploying CPR on Modal (GPU)

This guide covers deploying the Conformal Protein Retrieval Gradio app on [Modal](https://modal.com) with GPU-accelerated protein embedding.

## Why Modal?

The HuggingFace Spaces deployment runs on CPU, making protein embedding (ProtTrans T5 + Protein-Vec) slow. Modal runs the embedding on a T4 GPU serverlessly -- you pay per-second of compute and it scales to zero when idle.

## Architecture

```
Browser
  |
  v
https://{workspace}--cpr-gradio-ui.modal.run
  |
  v
[CPU Container] Gradio UI + FAISS search + threshold lookup
  |  .remote()
  v
[GPU Container(s)] ProtTrans T5 + Protein-Vec embedding (T4)
  |
  v
[Modal Volume] Cached models, embeddings, lookup data
```

- **CPU container**: Serves the Gradio web interface. Runs FAISS search and conformal prediction (fast on CPU). Limited to 1 container for Gradio's sticky session requirement.
- **GPU container(s)**: Runs protein embedding on a T4 GPU. Autoscales from 0 to N based on demand. Models are loaded once at container startup via `@modal.enter()`.
- **Modal Volume** (`cpr-data`): Persistent storage that caches downloaded models (~3 GB for ProtTrans, ~200 MB for Protein-Vec) and lookup data (~1.6 GB) across cold starts.

## Prerequisites

- Python 3.9+
- A [Modal account](https://modal.com) (free tier includes $30/month compute credit)
- This repository cloned locally

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate

```bash
modal setup
```

This opens a browser window for one-time authentication.

### 3. Deploy

**Dev mode** (temporary URL, hot-reloads on file changes):

```bash
modal serve modal_app.py
```

**Production** (permanent URL, runs until you redeploy or stop):

```bash
modal deploy modal_app.py
```

That's it. Modal builds two container images, downloads dependencies, and gives you a public URL.

## What you get

After `modal deploy`, you'll see output like:

```
Created web endpoint cpr-gradio => https://your-workspace--cpr-gradio-ui.modal.run
```

This URL is your permanent Gradio app, running with GPU embedding.

## First request (cold start)

The very first request after deployment triggers:

1. GPU container starts (~30s)
2. Downloads Protein-Vec models from HuggingFace to the volume (~1 min, one-time)
3. Downloads ProtTrans T5 from HuggingFace to the volume (~2 min, one-time)
4. Loads models onto GPU (~15s)

Subsequent requests reuse cached models from the volume, so cold starts are ~45s (container + model loading). Warm requests (within the scaledown window) are near-instant.

## Data

All data is hosted on HuggingFace at [`LoocasGoose/cpr_data`](https://huggingface.co/datasets/LoocasGoose/cpr_data) and downloaded automatically on first use:

| File | Size | Used by |
|------|------|---------|
| `protein_vec_models.tar.gz` | ~200 MB | GPU container (embedding models) |
| `data/lookup_embeddings.npy` | ~1.1 GB | CPU container (FAISS database) |
| `data/lookup_embeddings_meta_data.tsv` | ~535 MB | CPU container (result metadata) |
| `data/lookup/scope_lookup_embeddings.npy` | small | CPU container (SCOPE database) |
| `data/lookup/scope_lookup.fasta` | small | CPU container (SCOPE metadata) |

Small results files (`results/*.csv`) are baked into the web container image at build time.

The HF dataset ID is hardcoded in `modal_app.py` as `HF_DATASET_ID = "LoocasGoose/cpr_data"`. Change it there if you move the data.

## Cost

| Resource | Rate | Notes |
|----------|------|-------|
| T4 GPU | ~$0.59/hr | Only billed when processing requests |
| CPU (web) | ~$0.03/hr | While Gradio UI is alive |
| Volume | $0.63/GB/month | Cached models + data (~5 GB) |
| Free tier | $30/month | Covers light to moderate usage |

The app scales to zero when idle (after 20 min with no requests), so you're not billed for idle time on GPU. The CPU web container also scales down but has a 20-minute grace period.

## Configuration

Key parameters in `modal_app.py`:

```python
# GPU type — change to "A10G" or "A100" for faster inference
gpu="T4"

# How long to keep containers alive after last request (seconds)
scaledown_window=60 * 20  # 20 minutes

# Max concurrent users per web container
@modal.concurrent(max_inputs=100)

# Gradio request queue depth
demo.queue(max_size=5)
```

## Relationship to HuggingFace Spaces

This deployment is independent of the HuggingFace Spaces version:

| | HuggingFace Spaces | Modal |
|---|---|---|
| Entry point | `app.py` | `modal_app.py` |
| Embedding | CPU (subprocess) | T4 GPU (remote call) |
| Data source | Same HF dataset | Same HF dataset |
| Gradio code | `gradio_interface.py` (unchanged) | Same file, embedding monkey-patched |
| Cost | Free | Pay-per-second (~$30/mo free) |

Both deployments use the same `gradio_interface.py` and the same HF dataset. The Modal version monkey-patches `run_embed_protein_vec` to call the GPU function instead of a local subprocess. No code changes are needed in the Gradio interface itself.

## Troubleshooting

**"Module not found" errors in the web container**: The `protein_conformal` package is baked into the image via `.add_local_dir()`. Make sure you run `modal deploy` from the project root directory.

**Slow first request**: Expected. The volume caches everything after the first download, so subsequent cold starts only need to load models (~45s).

**GPU container timeout**: Default is 600s (10 min). If embedding very long sequences or large batches, increase `timeout` in the `@app.cls` decorator.

**Checking logs**: Use the [Modal dashboard](https://modal.com/apps) to view container logs, GPU utilization, and billing.

**Redeploying after code changes**: Just run `modal deploy modal_app.py` again. Modal rebuilds only the layers that changed.
