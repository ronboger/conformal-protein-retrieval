"""
Verify the prebuilt-FAISS-index path returns search results IDENTICAL to the
original load_database path, on the REAL Swiss-Prot lookup embeddings.

This is the targeted check that the cold-start optimization (build_index +
prebuilt sidecar + load_lookup_index) does not change search results, so the
conformal guarantees / paper numbers are unaffected.

Usage:
    python scripts/verify_prebuild_equiv.py [DATA_DIR]
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Use the protein_conformal that lives next to this script (worktree code).
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
from protein_conformal.util import (
    load_database, query, build_index, load_lookup_index, lookup_index_path,
)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data"
    lookup_path = data_dir / "lookup_embeddings.npy"
    query_path = data_dir / "gene_unknown" / "unknown_aa_seqs.npy"
    k = 1000

    print(f"Loading lookup {lookup_path} ...")
    emb = np.load(lookup_path).astype(np.float32)
    queries = np.load(query_path).astype(np.float32)
    print(f"  lookup {emb.shape}, queries {queries.shape}, k={k}")

    # OLD path (what the paper verification uses).
    old_index = load_database(emb.copy())
    D_old, I_old = query(old_index, queries.copy(), k)

    # NEW path 1: build_index (non-mutating).
    new_index = build_index(emb)
    D_new, I_new = query(new_index, queries.copy(), k)

    # NEW path 2: write a sidecar then read it back via load_lookup_index (the
    # exact path get_lookup_resources uses in production). The .npy is absent so
    # this also proves the np.load is skipped.
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as td:
        npy_path = os.path.join(td, "lookup_embeddings.npy")  # never created
        faiss.write_index(build_index(emb), lookup_index_path(npy_path))
        side_index, n = load_lookup_index(npy_path)
        D_side, I_side = query(side_index, queries.copy(), k)

    # Assertions: neighbors bit-identical, scores within float tolerance.
    assert np.array_equal(I_new, I_old), "build_index neighbors differ from load_database"
    assert np.allclose(D_new, D_old, atol=1e-6), "build_index scores differ from load_database"
    assert np.array_equal(I_side, I_old), "sidecar-read neighbors differ from load_database"
    assert np.allclose(D_side, D_old, atol=1e-6), "sidecar-read scores differ from load_database"
    assert n == emb.shape[0], f"num_embeddings mismatch: {n} != {emb.shape[0]}"

    print(f"EQUIVALENCE OK: prebuilt index == load_database "
          f"(identical neighbors for all {queries.shape[0]} queries x k={k}; "
          f"max |dD|={np.max(np.abs(D_new - D_old)):.2e})")


if __name__ == "__main__":
    main()
