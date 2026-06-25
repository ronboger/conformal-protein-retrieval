"""
Verify the GPU-resident embed path preserves results vs the original
(featurize_prottrans per-sequence) path, on the 149 Syn3.0 proteins (fp16).

Original path does a per-sequence GPU->CPU->GPU round-trip; the new path keeps the
T5 output on the GPU through embed_vec. Checks per-protein cosine, wall-clock, and
the Syn3.0 annotation (expect 59/149, identical hit set).

Run on a GPU node (scripts/slurm_verify_gpu_resident.sh).
"""
import re
import sys
import time
import tempfile
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
from utils_search import featurize_prottrans, embed_vec
from protein_conformal.util import read_fasta
from verify_syn30 import verify_syn30


def embed_original(seqs, model, tok, model_deep, masks, device):
    out = []
    for s in seqs:
        with torch.autocast("cuda", dtype=torch.float16):
            pt = featurize_prottrans([s], model, tok, device)
        pt = pt.float()
        out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def embed_resident(seqs, model, tok, model_deep, masks, device):
    out = []
    with torch.no_grad():
        for seq in seqs:
            prepped = re.sub(r"[UZOB]", "X", " ".join(seq[:3000]))
            ids = tok.batch_encode_plus([prepped], add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids["input_ids"]).to(device)
            attn = torch.tensor(ids["attention_mask"]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
            seq_len = int(attn[0].sum())
            pt = hidden[0, : seq_len - 1].float().unsqueeze(0)
            out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def main():
    device = torch.device("cuda:0")
    print("Loading ProtTrans + Protein-Vec ...", flush=True)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(f"{PVM}/protein_vec.ckpt", config=config).to(device).eval()

    keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    m = [keys[k] in keys for k in range(len(keys))]
    masks = torch.logical_not(torch.tensor(m, dtype=torch.bool))[None, :].to(device)

    seqs, _ = read_fasta(str(DATA / "gene_unknown" / "unknown_aa_seqs.fasta"))
    print(f"Embedding {len(seqs)} Syn3.0 sequences: original vs GPU-resident ...", flush=True)

    embed_original(seqs[:2], model, tok, model_deep, masks, device)  # warm up
    t0 = time.time(); emb_o = embed_original(seqs, model, tok, model_deep, masks, device); t_o = time.time() - t0
    t0 = time.time(); emb_r = embed_resident(seqs, model, tok, model_deep, masks, device); t_r = time.time() - t0

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    cos = (unit(emb_o) * unit(emb_r)).sum(1)

    tmp = Path(tempfile.mkdtemp(dir="/groups/doudna/projects/ronb/tmp"))
    p_o, p_r = tmp / "orig.npy", tmp / "res.npy"
    np.save(p_o, emb_o); np.save(p_r, emb_r)

    common = dict(
        query_fasta_path=DATA / "gene_unknown" / "unknown_aa_seqs.fasta",
        lookup_embeddings_path=DATA / "lookup_embeddings.npy",
        lookup_metadata_path=DATA / "lookup_embeddings_meta_data.tsv",
        calibration_data_path=DATA / "pfam_new_proteins.npy",
        alpha=0.1, verbose=False,
    )
    ro = verify_syn30(query_embeddings_path=p_o, **common)
    rr = verify_syn30(query_embeddings_path=p_r, **common)

    def hit_set(r):
        return set(np.where(r["results_df"]["is_hit"].values)[0].tolist())

    ho, hr = hit_set(ro), hit_set(rr)

    print("\n============ GPU-RESIDENT VERIFICATION ============")
    print(f"per-protein cosine(original, resident): min={cos.min():.6f} mean={cos.mean():.6f}")
    print(f"embed time  original : {t_o:.1f}s")
    print(f"embed time  resident : {t_r:.1f}s   speedup={t_o / t_r:.2f}x  (cluster GPU; Modal gain may differ)")
    print(f"Syn3.0  original : {ro['n_hits']}/{ro['n_queries']} ({ro['hit_rate']:.1%})")
    print(f"Syn3.0  resident : {rr['n_hits']}/{rr['n_queries']} ({rr['hit_rate']:.1%})")
    print(f"hit-set identical: {ho == hr}  symmetric_diff={sorted(ho ^ hr)}")
    verdict = "GPU-RESIDENT PRESERVES results" if (ro["n_hits"] == rr["n_hits"] and ho == hr) else "CHANGES results -- review"
    print(f"VERDICT: {verdict}")
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
