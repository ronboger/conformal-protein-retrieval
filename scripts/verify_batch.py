"""
Verify batched embedding preserves results (and is faster) vs per-sequence.

The web app embeds N query sequences by calling featurize_prottrans([seq]) once
per sequence -> N separate ProtTrans T5 forward passes (the Syn3.0 slowness).
featurize_prottrans already runs T5 on a padded batch but keeps only features[0];
this batches the expensive T5 forward and keeps ALL per-sequence features, then
runs the cheap per-sequence embed_vec (Protein-Vec head) unchanged.

Compares, on the 149 Syn3.0 proteins (fp16, as in production):
  - per-protein cosine(per-seq, batched)
  - wall-clock embedding time + speedup
  - Syn3.0 annotation count (expect 59/149) and exact hit set

Run on a GPU node (scripts/slurm_verify_fp16.sh pattern).
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

BATCH_SIZE = 8


def embed_per_seq(seqs, model, tok, model_deep, masks, device):
    """Current production path: one fp16 T5 forward per sequence."""
    out = []
    for s in seqs:
        with torch.autocast("cuda", dtype=torch.float16):
            pt = featurize_prottrans([s], model, tok, device)
        pt = pt.float()
        out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def embed_batched(seqs, model, tok, model_deep, masks, device, batch_size=BATCH_SIZE):
    """Batch the T5 forward (keep all per-seq features); embed_vec per sequence."""
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        prepped = [" ".join(s[:3000]) for s in chunk]
        prepped = [re.sub(r"[UZOB]", "X", s) for s in prepped]
        ids = tok.batch_encode_plus(prepped, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float().cpu().numpy()
        for j in range(len(hidden)):
            seq_len = int((attention_mask[j] == 1).sum())
            feat = hidden[j][:seq_len - 1]  # strip EOS, matching featurize_prottrans
            pt = torch.unsqueeze(torch.tensor(feat), 0).to(device)
            out.append(np.asarray(embed_vec(pt, model_deep, masks, device), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


def main():
    device = torch.device("cuda:0")
    print("Loading ProtTrans + Protein-Vec ...", flush=True)
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device).eval()
    config = trans_basic_block_Config.from_json(f"{PVM}/protein_vec_params.json")
    model_deep = trans_basic_block.load_from_checkpoint(f"{PVM}/protein_vec.ckpt", config=config).to(device).eval()

    sampled_keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    all_cols = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    m = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(m, dtype=torch.bool))[None, :].to(device)

    seqs, _ = read_fasta(str(DATA / "gene_unknown" / "unknown_aa_seqs.fasta"))
    print(f"Embedding {len(seqs)} Syn3.0 sequences: per-seq vs batched (B={BATCH_SIZE}) ...", flush=True)

    # warm up (kernels/autocast) so timing is fair
    embed_per_seq(seqs[:2], model, tok, model_deep, masks, device)

    t0 = time.time(); emb_seq = embed_per_seq(seqs, model, tok, model_deep, masks, device); t_seq = time.time() - t0
    t0 = time.time(); emb_bat = embed_batched(seqs, model, tok, model_deep, masks, device); t_bat = time.time() - t0

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    cos = (unit(emb_seq) * unit(emb_bat)).sum(1)

    tmp = Path(tempfile.mkdtemp(dir="/groups/doudna/projects/ronb/tmp"))
    p_seq, p_bat = tmp / "seq.npy", tmp / "bat.npy"
    np.save(p_seq, emb_seq)
    np.save(p_bat, emb_bat)

    common = dict(
        query_fasta_path=DATA / "gene_unknown" / "unknown_aa_seqs.fasta",
        lookup_embeddings_path=DATA / "lookup_embeddings.npy",
        lookup_metadata_path=DATA / "lookup_embeddings_meta_data.tsv",
        calibration_data_path=DATA / "pfam_new_proteins.npy",
        alpha=0.1, verbose=False,
    )
    rs = verify_syn30(query_embeddings_path=p_seq, **common)
    rb = verify_syn30(query_embeddings_path=p_bat, **common)

    def hit_set(r):
        return set(np.where(r["results_df"]["is_hit"].values)[0].tolist())

    hs, hb = hit_set(rs), hit_set(rb)

    print("\n================ BATCH VERIFICATION ================")
    print(f"per-protein cosine(per-seq, batched): min={cos.min():.6f} mean={cos.mean():.6f}")
    print(f"embed time  per-seq : {t_seq:.1f}s")
    print(f"embed time  batched : {t_bat:.1f}s   speedup={t_seq / t_bat:.2f}x")
    print(f"Syn3.0  per-seq : {rs['n_hits']}/{rs['n_queries']} ({rs['hit_rate']:.1%})")
    print(f"Syn3.0  batched : {rb['n_hits']}/{rb['n_queries']} ({rb['hit_rate']:.1%})")
    print(f"hit-set identical: {hs == hb}  symmetric_diff={sorted(hs ^ hb)}")
    verdict = "BATCHING PRESERVES results" if (rs["n_hits"] == rb["n_hits"] and hs == hb) else "BATCHING CHANGES results -- review"
    print(f"VERDICT: {verdict}")
    print("====================================================", flush=True)


if __name__ == "__main__":
    main()
