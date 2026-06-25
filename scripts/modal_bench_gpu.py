"""
Benchmark T4 vs A10G on the real embedding workload (149 Syn3.0 proteins, fp16,
per-sequence as in production). Warm timing (excludes cold start / model load).

    modal run scripts/modal_bench_gpu.py
"""
import os
import sys

# import gpu_image/volume/constants from the sibling modal_app.py (same checkout)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
from modal_app import gpu_image, volume, VOLUME_PATH, PVM_DIR, HF_CACHE

app = modal.App("cpr-bench-gpu", image=gpu_image)


def _bench():
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

    # read the 149 Syn3.0 sequences from the volume
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

    def one(s):
        with torch.autocast("cuda", dtype=torch.float16):
            pt = featurize_prottrans([s], model, tok, device)
        pt = pt.float()
        return embed_vec(pt, model_deep, masks, device)

    for s in seqs[:3]:  # warm up kernels/autocast
        one(s)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in seqs:
        one(s)
    torch.cuda.synchronize()
    return gpu_name, len(seqs), time.time() - t0


@app.function(gpu="T4", volumes={VOLUME_PATH: volume}, timeout=900)
def bench_t4():
    return _bench()


@app.function(gpu="A10G", volumes={VOLUME_PATH: volume}, timeout=900)
def bench_a10g():
    return _bench()


@app.local_entrypoint()
def main():
    f1 = bench_t4.spawn()
    f2 = bench_a10g.spawn()
    g1, n1, t1 = f1.get()
    g2, n2, t2 = f2.get()
    print(f"\n=== GPU EMBED BENCH (warm, {n1} Syn3.0 seqs, fp16, per-sequence) ===")
    print(f"{g1:20s}: {t1:6.1f}s  ({t1 / n1 * 1000:5.0f} ms/seq)")
    print(f"{g2:20s}: {t2:6.1f}s  ({t2 / n2 * 1000:5.0f} ms/seq)")
    print(f"speedup A10G vs T4: {t1 / t2:.2f}x")
