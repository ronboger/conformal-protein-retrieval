"""
Experiment: can the Protein-Vec *head* run in fp16?

The deployed embed path (modal_app.Embedder.embed) runs the big ProtTrans T5 in
fp16 autocast but casts back to fp32 for the Protein-Vec head (model_deep), because
an earlier attempt at fp16 on the head hit a cuDNN SDPA "no execution plan" error.
modal_diagnose.py later showed the head is ~23.5 ms/seq (~30% of the embed), so it's
worth revisiting.

This script holds the (already-verified) fp16 T5 output fixed and only varies the
HEAD precision/SDPA backend, reporting for each variant:
  - did it run (or which error), ms/seq (warmed), and cosine vs the fp32 head.

A variant is a keeper only if it (a) runs, (b) is faster than fp32, and (c) has
cosine ~1.0 (so Syn3.0 stays 59/149 -- confirmed separately with verify_fp16.py).

Run on a GPU node:  sbatch scripts/slurm_exp_head_fp16.sh
"""
import sys
import time
import contextlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent.parent
MAIN = Path("/groups/doudna/projects/ronb/conformal-protein-retrieval")
PVM = str(MAIN / "protein_vec_models")
DATA = MAIN / "data"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, PVM)

from transformers import T5EncoderModel, T5Tokenizer
from model_protein_moe import trans_basic_block, trans_basic_block_Config
from utils_search import featurize_prottrans
from protein_conformal.util import read_fasta

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    HAVE_SDPA_KERNEL = True
except ImportError:  # older torch
    HAVE_SDPA_KERNEL = False


def main():
    device = torch.device("cuda:0")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}", flush=True)
    print(f"gpu {torch.cuda.get_device_name(0)}  sdpa_kernel_api={HAVE_SDPA_KERNEL}", flush=True)

    print("Loading ProtTrans + Protein-Vec ...", flush=True)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(
        f"{PVM}/protein_vec.ckpt", config=config
    ).to(device).eval()

    sampled_keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    all_cols = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    m = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(m, dtype=torch.bool))[None, :].to(device)

    seqs, _ = read_fasta(str(DATA / "gene_unknown" / "unknown_aa_seqs.fasta"))
    seqs = seqs[:20]
    print(f"Using {len(seqs)} Syn3.0 sequences.", flush=True)

    # Precompute the fp16-autocast T5 per-residue embeddings ONCE (exactly as prod),
    # cast to fp32 -- this is the head's input and is held fixed across variants.
    pts = []
    with torch.no_grad():
        for s in seqs:
            with torch.autocast("cuda", dtype=torch.float16):
                pt = featurize_prottrans([s], model, tok, device)
            pts.append(pt.float())

    def head_forward(pt):
        # Mirrors utils_search.embed_vec but stays on GPU and returns the tensor.
        padding = torch.zeros(pt.shape[0:2], dtype=torch.bool, device=device)
        out_seq = model_deep.make_matrix(pt, padding)
        return model_deep(out_seq, masks)

    def run_variant(ctx_factory):
        """Run head_forward over all pts under a context-manager factory; time + collect."""
        with torch.no_grad():
            for pt in pts[:3]:  # warmup
                with ctx_factory():
                    head_forward(pt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = []
            for pt in pts:
                with ctx_factory():
                    v = head_forward(pt)
                outs.append(v.float().detach().cpu().numpy())
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / len(pts) * 1000.0
        return np.concatenate(outs).astype(np.float32), ms

    @contextlib.contextmanager
    def fp16_autocast():
        with torch.autocast("cuda", dtype=torch.float16):
            yield

    @contextlib.contextmanager
    def fp16_math():
        with torch.autocast("cuda", dtype=torch.float16):
            if HAVE_SDPA_KERNEL:
                with sdpa_kernel(SDPBackend.MATH):
                    yield
            else:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                ):
                    yield

    @contextlib.contextmanager
    def fp16_efficient():
        with torch.autocast("cuda", dtype=torch.float16):
            if HAVE_SDPA_KERNEL:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                    yield
            else:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_mem_efficient=True, enable_math=True
                ):
                    yield

    # Reference: fp32 head, default backends.
    ref, ref_ms = run_variant(contextlib.nullcontext)

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    variants = [
        ("fp16 + plain autocast (no SDPA override)", fp16_autocast),
        ("fp16 + math SDPA", fp16_math),
        ("fp16 + efficient/flash SDPA (cudnn off)", fp16_efficient),
    ]

    print("\n================ HEAD FP16 EXPERIMENT ================", flush=True)
    print(f"{'variant':<44} {'ran':<5} {'ms/seq':>8} {'speedup':>8} {'min_cos':>9}", flush=True)
    print(f"{'fp32 head (reference)':<44} {'yes':<5} {ref_ms:>8.2f} {'1.00x':>8} {'1.000000':>9}", flush=True)

    # efficient/flash variant needs cuDNN SDPA disabled globally to take effect.
    for name, ctx in variants:
        toggle_cudnn = "efficient/flash" in name and hasattr(torch.backends.cuda, "enable_cudnn_sdp")
        if toggle_cudnn:
            torch.backends.cuda.enable_cudnn_sdp(False)
        try:
            out, ms = run_variant(ctx)
            cos = (unit(ref) * unit(out)).sum(1).min()
            print(f"{name:<44} {'yes':<5} {ms:>8.2f} {ref_ms/ms:>7.2f}x {cos:>9.6f}", flush=True)
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            print(f"{name:<44} {'NO':<5} {'-':>8} {'-':>8}   ERROR: {msg}", flush=True)
        finally:
            if toggle_cudnn:
                torch.backends.cuda.enable_cudnn_sdp(True)

    print("=====================================================", flush=True)
    print("Keeper = ran AND speedup > 1.0 AND min_cos ~1.0 (>=0.9999).", flush=True)
    print("If a keeper exists, port its context into modal_app.Embedder.embed and", flush=True)
    print("re-run scripts/slurm_verify_fp16.sh to confirm Syn3.0 stays 59/149.", flush=True)


if __name__ == "__main__":
    main()
