"""
Scaling benchmark: how embedding time grows with the number of query proteins,
on A10G (fp16, per-sequence as in production). Cycles the 149 Syn3.0 sequences up
to each target N. Confirms linearity and shows genome-scale feasibility.

    modal run scripts/modal_bench_scaling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
from modal_app import gpu_image, volume, VOLUME_PATH, PVM_DIR, HF_CACHE

app = modal.App("cpr-bench-scaling", image=gpu_image)
SIZES = [150, 1000, 5000]


@app.function(gpu="A10G", volumes={VOLUME_PATH: volume}, timeout=1800)
def bench_sizes():
    import sys as _sys
    import time
    import torch
    import numpy as np

    _sys.path.insert(0, PVM_DIR)
    from transformers import T5EncoderModel, T5Tokenizer
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import featurize_prottrans, embed_vec

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, cache_dir=HF_CACHE)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=HF_CACHE).to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM_DIR}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(
        f"{PVM_DIR}/protein_vec.ckpt", config=config, map_location=device
    ).to(device).eval()

    keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    m = [keys[k] in keys for k in range(len(keys))]
    masks = torch.logical_not(torch.tensor(m, dtype=torch.bool))[None, :]

    base, cur = [], []
    for line in open(f"{VOLUME_PATH}/data/gene_unknown/unknown_aa_seqs.fasta"):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                base.append("".join(cur)); cur = []
        elif line:
            cur.append(line)
    if cur:
        base.append("".join(cur))

    def one(s):
        with torch.autocast("cuda", dtype=torch.float16):
            pt = featurize_prottrans([s], model, tok, device)
        pt = pt.float()
        return embed_vec(pt, model_deep, masks, device)

    for s in base[:3]:  # warm up
        one(s)
    torch.cuda.synchronize()

    results = []
    for n in SIZES:
        seqs = [base[i % len(base)] for i in range(n)]
        t0 = time.time()
        for s in seqs:
            one(s)
        torch.cuda.synchronize()
        dt = time.time() - t0
        results.append((n, dt, dt / n * 1000))
    return gpu_name, results


@app.local_entrypoint()
def main():
    gpu_name, results = bench_sizes.remote()
    print(f"\n=== EMBED SCALING on {gpu_name} (fp16, per-sequence) ===")
    for n, dt, mspn in results:
        print(f"N={n:6d}: {dt:7.1f}s  ({mspn:5.0f} ms/seq)")
    mspn = results[-1][2]
    print("\nExtrapolated full-genome embedding (at largest-N rate):")
    for label, g in [("bacterial ~4,000", 4000), ("yeast ~6,000", 6000), ("human ~20,000", 20000)]:
        print(f"  {label}: ~{g * mspn / 1000 / 60:.1f} min")
