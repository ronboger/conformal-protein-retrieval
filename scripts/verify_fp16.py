"""
Verify fp16 (torch.autocast) embedding preserves search results vs fp32.

Embeds the 149 JCVI Syn3.0 proteins with the real ProtTrans + Protein-Vec
pipeline in fp32 and in fp16 (autocast), then runs the canonical verify_syn30
annotation (FDR alpha=0.1) on both and compares:
  - per-protein cosine(fp32, fp16)
  - Syn3.0 annotation count (expected 59/149) and the exact hit set
  - per-query top-1 Pfam annotation agreement

Run on a GPU node (see scripts/slurm_verify_fp16.sh). The fp16 path mirrors how it
would run in modal_app.Embedder.embed: fp32 weights + autocast, output cast to fp32.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent.parent
MAIN = Path("/groups/doudna/projects/ronb/conformal-protein-retrieval")
PVM = str(MAIN / "protein_vec_models")
DATA = MAIN / "data"

sys.path.insert(0, str(REPO))          # worktree protein_conformal
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, PVM)                # utils_search, model_protein_moe

from transformers import T5EncoderModel, T5Tokenizer
from model_protein_moe import trans_basic_block, trans_basic_block_Config
from utils_search import featurize_prottrans, embed_vec
from protein_conformal.util import read_fasta
from verify_syn30 import verify_syn30


def main():
    device = torch.device("cuda:0")
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

    seqs, meta = read_fasta(str(DATA / "gene_unknown" / "unknown_aa_seqs.fasta"))
    print(f"Embedding {len(seqs)} Syn3.0 sequences (fp32 and fp16) ...", flush=True)

    def embed_all(fp16):
        out = []
        for s in seqs:
            if fp16:
                # Autocast only the large ProtTrans T5 (where ~all the cost is);
                # keep the small Protein-Vec head in fp32. Its fp16 attention hits
                # a cuDNN SDPA "no execution plan" error and it's cheap anyway.
                with torch.autocast("cuda", dtype=torch.float16):
                    pt = featurize_prottrans([s], model, tok, device)
                pt = pt.float()
                e = embed_vec(pt, model_deep, masks, device)
            else:
                pt = featurize_prottrans([s], model, tok, device)
                e = embed_vec(pt, model_deep, masks, device)
            out.append(np.asarray(e, dtype=np.float32))
        return np.concatenate(out).astype(np.float32)

    emb32 = embed_all(False)
    emb16 = embed_all(True)

    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    cos = (unit(emb32) * unit(emb16)).sum(1)
    print(f"\nPer-protein cosine fp32 vs fp16: min={cos.min():.6f} mean={cos.mean():.6f}", flush=True)

    tmp = Path(tempfile.mkdtemp(dir="/groups/doudna/projects/ronb/tmp"))
    p32, p16 = tmp / "emb32.npy", tmp / "emb16.npy"
    np.save(p32, emb32)
    np.save(p16, emb16)

    common = dict(
        query_fasta_path=DATA / "gene_unknown" / "unknown_aa_seqs.fasta",
        lookup_embeddings_path=DATA / "lookup_embeddings.npy",
        lookup_metadata_path=DATA / "lookup_embeddings_meta_data.tsv",
        calibration_data_path=DATA / "pfam_new_proteins.npy",
        alpha=0.1,
        verbose=False,
    )
    print("\nRunning verify_syn30 annotation (precomputed / my-fp32 / my-fp16) ...", flush=True)
    ref = verify_syn30(query_embeddings_path=DATA / "gene_unknown" / "unknown_aa_seqs.npy", **common)
    r32 = verify_syn30(query_embeddings_path=p32, **common)
    r16 = verify_syn30(query_embeddings_path=p16, **common)

    def hit_set(r):
        return set(np.where(r["results_df"]["is_hit"].values)[0].tolist())

    h_ref, h32, h16 = hit_set(ref), hit_set(r32), hit_set(r16)
    same_pfam = (
        r32["results_df"]["pfam_annotation"].values == r16["results_df"]["pfam_annotation"].values
    ).mean()

    print("\n================ FP16 VERIFICATION ================")
    print(f"precomputed file : {ref['n_hits']}/{ref['n_queries']} ({ref['hit_rate']:.1%})  [reference, expect 59/149]")
    print(f"my fp32 re-embed : {r32['n_hits']}/{r32['n_queries']} ({r32['hit_rate']:.1%})")
    print(f"my fp16 autocast : {r16['n_hits']}/{r16['n_queries']} ({r16['hit_rate']:.1%})")
    print(f"hit-set fp32 vs fp16: identical={h32 == h16}  symmetric_diff={sorted(h32 ^ h16)}")
    print(f"hit-set ref  vs fp32: identical={h_ref == h32}  symmetric_diff={sorted(h_ref ^ h32)}")
    print(f"per-query Pfam annotation identical (fp32 vs fp16): {same_pfam:.1%}")
    verdict = "fp16 PRESERVES results" if (r16["n_hits"] == r32["n_hits"] and h32 == h16) else "fp16 CHANGES results -- review"
    print(f"VERDICT: {verdict}")
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
