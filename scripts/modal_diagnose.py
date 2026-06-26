"""
Diagnose the Modal-A10G-vs-cluster-A5000 4x embedding gap (Codex leads).

Prints the runtime stack (torch/CUDA/cuDNN versions, vCPU count, SDPA backends)
and a per-stage breakdown (tokenize / H2D / T5 forward / Protein-Vec head) so we
can tell whether the gap is the stack, CPU starvation, or transfers.

    modal run scripts/modal_diagnose.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
from modal_app import gpu_image, volume, VOLUME_PATH, PVM_DIR, HF_CACHE

# Modal 1.x no longer auto-mounts sibling local modules into the container, so the
# top-level `from modal_app import ...` (above) would fail on import inside the
# container. Explicitly add modal_app.py as a local Python source.
app = modal.App("cpr-diagnose", image=gpu_image.add_local_python_source("modal_app"))


@app.function(gpu="A10G", volumes={VOLUME_PATH: volume}, timeout=600)
def diagnose():
    import re
    import time
    import torch
    import numpy as np

    sys.path.insert(0, PVM_DIR)
    from transformers import T5EncoderModel, T5Tokenizer
    import transformers
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import embed_vec

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "os.cpu_count": os.cpu_count(),
        "torch.get_num_threads": torch.get_num_threads(),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }

    device = torch.device("cuda:0")
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, cache_dir=HF_CACHE)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=HF_CACHE).to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM_DIR}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(
        f"{PVM_DIR}/protein_vec.ckpt", config=config, map_location=device
    ).to(device).eval()
    keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    masks = torch.logical_not(torch.tensor([keys[k] in keys for k in range(len(keys))], dtype=torch.bool))[None, :]

    seqs, cur = [], []
    for line in open(f"{VOLUME_PATH}/data/gene_unknown/unknown_aa_seqs.fasta"):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    seqs = seqs[:20]

    def one(s, acc):
        t0 = time.perf_counter()
        prepped = re.sub(r"[UZOB]", "X", " ".join(s[:3000]))
        ids = tok.batch_encode_plus([prepped], add_special_tokens=True, padding=True)
        acc["tokenize"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attn = torch.tensor(ids["attention_mask"]).to(device)
        torch.cuda.synchronize(); acc["h2d"] += time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        torch.cuda.synchronize(); acc["t5"] += time.perf_counter() - t0

        sl = int(attn[0].sum())
        pt = hidden[0, : sl - 1].float().unsqueeze(0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        embed_vec(pt, model_deep, masks, device)
        torch.cuda.synchronize(); acc["head"] += time.perf_counter() - t0

    warm = {"tokenize": 0, "h2d": 0, "t5": 0, "head": 0}
    for s in seqs[:3]:
        one(s, warm)
    acc = {"tokenize": 0, "h2d": 0, "t5": 0, "head": 0}
    for s in seqs:
        one(s, acc)
    n = len(seqs)
    timings = {k: round(v / n * 1000, 1) for k, v in acc.items()}
    timings["total_ms_per_seq"] = round(sum(acc.values()) / n * 1000, 1)
    return info, timings


@app.local_entrypoint()
def main():
    info, timings = diagnose.remote()
    print("\n=== MODAL A10G RUNTIME STACK ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("\n=== PER-STAGE TIMING (ms/seq, 20 Syn3.0) ===")
    for k, v in timings.items():
        print(f"  {k}: {v}")
