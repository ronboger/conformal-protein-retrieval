"""
Prebuild FAISS indexes for the Protein-Vec lookup databases.

Building the exact IndexFlatIP from a raw .npy on every container cold start is a
major part of the site's startup latency: it loads the full (~1 GB) embedding
array, normalizes it, and adds every vector. This script does that work once,
offline, and writes a serialized index next to each .npy (see
util.lookup_index_path). At runtime get_lookup_resources reads the prebuilt index
with faiss.read_index and never touches the .npy.

The indexes are exact (IndexFlatIP, cosine via L2-normalized vectors), so search
results are identical to building on the fly -- no recall loss, conformal
guarantees unaffected.

CLEAN is intentionally excluded: it uses a different index type (IndexFlatL2) on a
separate code path and is tiny (5242 centroids), so prebuilding it buys nothing.

Usage:
    python scripts/prebuild_faiss.py                 # build the known default DBs
    python scripts/prebuild_faiss.py a.npy b.npy     # build specific .npy files
"""
import os
import sys

import numpy as np

# Ensure we import the protein_conformal that lives next to this script (the same
# checkout), not whatever editable install happens to be on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_conformal.util import load_or_build_index, lookup_index_path

# Mirror the DEFAULT_*_EMBEDDING constants in
# protein_conformal/backend/gradio_interface.py (the cosine Protein-Vec DBs that
# flow through get_lookup_resources).
DEFAULT_EMBEDDING_PATHS = [
    "./data/lookup_embeddings.npy",                  # Swiss-Prot (540K)
    "./data/lookup/scope_lookup_embeddings.npy",     # SCOPe
    "./data/afdb/afdb_embeddings_protein_vec.npy",   # AFDB
    "./data/euk/euk_embeddings_protein_vec.npy",     # Euk (74K)
]


def prebuild(embedding_path: str) -> bool:
    """Build and persist the sidecar index for one .npy. Returns True on success."""
    if not os.path.exists(embedding_path):
        print(f"SKIP (missing): {embedding_path}")
        return False

    index_path = lookup_index_path(embedding_path)
    if os.path.exists(index_path):
        print(f"SKIP (exists): {index_path}")
        return True

    print(f"Loading {embedding_path} ...")
    embeddings = np.load(embedding_path, allow_pickle=True).astype(np.float32, copy=False)
    print(f"  building index for {embeddings.shape[0]} x {embeddings.shape[1]} embeddings ...")
    load_or_build_index(embeddings, index_path)
    print(f"  wrote {index_path}")
    return True


def main(argv: list) -> int:
    paths = argv[1:] if len(argv) > 1 else DEFAULT_EMBEDDING_PATHS
    built = sum(prebuild(p) for p in paths)
    print(f"\nDone: {built}/{len(paths)} indexes available.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
