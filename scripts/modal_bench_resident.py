"""
Measure original vs GPU-resident embed on a Modal A10G (where the per-sequence
CPU round-trip overhead actually lives). 149 Syn3.0 proteins, fp16, warm.

    modal run scripts/modal_bench_resident.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
from modal_app import gpu_image, volume, VOLUME_PATH, PVM_DIR, HF_CACHE

app = modal.App("cpr-bench-resident", image=gpu_image)


@app.function(gpu="A10G", volumes={VOLUME_PATH: volume}, timeout=900)
def bench():
    import sys as _s
    import re
    import time
    import torch
    import numpy as np

    _s.path.insert(0, PVM_DIR)
    from transformers import T5EncoderModel, T5Tokenizer
    from model_protein_moe import trans_basic_block, trans_basic_block_Config
    from utils_search import featurize_prottrans, embed_vec

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(0)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False, cache_dir=HF_CACHE)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=HF_CACHE).to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM_DIR}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(
        f"{PVM_DIR}/protein_vec.ckpt", config=config, map_location=device
    ).to(device).eval()
    keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    mk = [keys[k] in keys for k in range(len(keys))]
    masks = torch.logical_not(torch.tensor(mk, dtype=torch.bool))[None, :]

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

    def orig(seqs):
        out = []
        for s in seqs:
            with torch.autocast("cuda", dtype=torch.float16):
                pt = featurize_prottrans([s], model, tok, device)
            pt = pt.float()
            out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
        return np.concatenate(out)

    def resident(seqs):
        out = []
        with torch.no_grad():
            for s in seqs:
                prepped = re.sub(r"[UZOB]", "X", " ".join(s[:3000]))
                ids = tok.batch_encode_plus([prepped], add_special_tokens=True, padding=True)
                input_ids = torch.tensor(ids["input_ids"]).to(device)
                attn = torch.tensor(ids["attention_mask"]).to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
                sl = int(attn[0].sum())
                pt = hidden[0, : sl - 1].float().unsqueeze(0)
                out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
        return np.concatenate(out)

    orig(base[:2])  # warm up
    torch.cuda.synchronize(); t0 = time.time(); orig(base); torch.cuda.synchronize(); t_o = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time(); resident(base); torch.cuda.synchronize(); t_r = time.time() - t0
    return gpu, len(base), t_o, t_r


@app.local_entrypoint()
def main():
    gpu, n, t_o, t_r = bench.remote()
    print(f"\n=== original vs GPU-resident ({n} Syn3.0, fp16, warm) on {gpu} ===")
    print(f"original : {t_o:6.1f}s  ({t_o / n * 1000:5.0f} ms/seq)")
    print(f"resident : {t_r:6.1f}s  ({t_r / n * 1000:5.0f} ms/seq)   speedup={t_o / t_r:.2f}x")
