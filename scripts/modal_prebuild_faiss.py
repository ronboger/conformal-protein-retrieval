"""
One-off: build prebuilt FAISS sidecar indexes on the cpr-data Modal volume.

    modal run scripts/modal_prebuild_faiss.py

Builds <name>.faissindex next to each cosine Protein-Vec lookup .npy on
/vol/data so production cold starts read the index instead of doing
np.load + normalize + index.add. Idempotent (skips DBs whose sidecar already
exists). Does NOT touch CLEAN (IndexFlatL2, separate code path).

The build below mirrors protein_conformal.util.build_index exactly (copy ->
normalize_L2 -> IndexFlatIP -> add); keep them in sync. It is inlined so this
one-off doesn't need the package baked into the image.
"""
import modal

volume = modal.Volume.from_name("cpr-data")
VOLUME_PATH = "/vol"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "faiss-cpu>=1.7.4"
)

app = modal.App("cpr-prebuild-faiss", image=image)

# Mirror gradio_interface DEFAULT_*_EMBEDDING (relative to /vol). Cosine DBs only.
EMBEDDING_FILES = [
    "data/lookup_embeddings.npy",                 # Swiss-Prot (540K)
    "data/lookup/scope_lookup_embeddings.npy",    # SCOPe
    "data/afdb/afdb_embeddings_protein_vec.npy",  # AFDB
    "data/euk/euk_embeddings_protein_vec.npy",    # Euk (74K)
]


def _lookup_index_path(p: str) -> str:
    return p[:-4] + ".faissindex" if p.endswith(".npy") else p + ".faissindex"


@app.function(volumes={VOLUME_PATH: volume}, timeout=3600, memory=32768)
def prebuild():
    import os
    import numpy as np
    import faiss

    built = 0
    for rel in EMBEDDING_FILES:
        npy = os.path.join(VOLUME_PATH, rel)
        if not os.path.exists(npy):
            print(f"SKIP (missing): {npy}")
            continue
        idx_path = _lookup_index_path(npy)
        if os.path.exists(idx_path):
            print(f"SKIP (exists): {idx_path}")
            continue
        print(f"Loading {npy} ...")
        emb = np.ascontiguousarray(np.load(npy, allow_pickle=True), dtype=np.float32).copy()
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, idx_path)
        print(f"  wrote {idx_path}  ({index.ntotal} x {emb.shape[1]})")
        built += 1

    if built:
        volume.commit()
    print(f"Done: {built} new sidecar index(es) committed.")


@app.local_entrypoint()
def main():
    prebuild.remote()
