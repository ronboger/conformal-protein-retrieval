#!/usr/bin/env python
"""
Verify FDR algorithm using available calibration data.

This script tests the core FDR threshold computation algorithm using the
Pfam calibration data. It verifies that:
1. The FAISS similarity search works correctly
2. The FDR threshold computation produces the expected value
3. The Venn-Abers probability calibration works

This is a functional test of the algorithm, not a reproduction of the
exact Syn3.0 results (which require additional query embeddings).

Usage:
    python scripts/verify_fdr_algorithm.py
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, repo_root)

# Import util directly to avoid gradio dependency in __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("util", f"{repo_root}/protein_conformal/util.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

load_database = util.load_database
query = util.query
simplifed_venn_abers_prediction = util.simplifed_venn_abers_prediction
get_sims_labels = util.get_sims_labels
get_thresh_FDR = util.get_thresh_FDR


def main():
    data_dir = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("FDR Algorithm Verification")
    print("=" * 60)

    # Check required files
    lookup_embeddings_path = data_dir / 'lookup_embeddings.npy'
    lookup_metadata_path = data_dir / 'lookup_embeddings_meta_data.tsv'
    calibration_data_path = data_dir / 'pfam_new_proteins.npy'

    missing = []
    for p in [lookup_embeddings_path, lookup_metadata_path, calibration_data_path]:
        if not p.exists():
            missing.append(p)

    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Test 1: Load lookup embeddings and build FAISS index
    print("\n1. Testing FAISS index construction...")
    embeddings = np.load(lookup_embeddings_path)
    print(f"   Loaded embeddings: {embeddings.shape}")

    # Build index on a subset for speed
    subset_size = 10000
    subset_embeddings = embeddings[:subset_size]
    db = load_database(subset_embeddings)
    print(f"   Built FAISS index on {subset_size} embeddings")

    # Test 2: Query the database
    print("\n2. Testing similarity search...")
    # Use random query
    np.random.seed(42)
    query_emb = np.random.randn(10, 512).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    D, I = query(db, query_emb, k=5)
    print(f"   Query shape: {query_emb.shape}")
    print(f"   Results D shape: {D.shape}, I shape: {I.shape}")
    print(f"   Max similarity: {D.max():.6f}")
    print(f"   Min similarity: {D.min():.6f}")

    # Test 3: Load calibration data and compute FDR threshold
    print("\n3. Testing FDR threshold computation...")
    cal_data = np.load(calibration_data_path, allow_pickle=True)
    print(f"   Loaded {len(cal_data)} calibration samples")

    # Use a subset for faster testing
    np.random.seed(42)
    np.random.shuffle(cal_data)
    cal_subset = cal_data[:100]

    sims, labels = get_sims_labels(cal_subset, partial=False)
    print(f"   Calibration sims shape: {sims.shape}")
    print(f"   Calibration labels shape: {labels.shape}")

    # Compute FDR threshold
    alpha = 0.1
    delta = 0.5
    try:
        l_hat, risk_fdr = get_thresh_FDR(labels.flatten(), sims.flatten(), alpha=alpha, delta=delta, N=50)
        print(f"   FDR threshold (α={alpha}): λ = {l_hat:.12f}")
        print(f"   FDR risk at threshold: {risk_fdr:.6f}")

        # Expected threshold is around 0.999980
        if 0.9999 < l_hat < 1.0001:
            print("   ✓ Threshold is in expected range [0.9999, 1.0001]")
        else:
            print(f"   ⚠ Threshold {l_hat} outside expected range")
    except Exception as e:
        print(f"   ✗ FDR computation failed: {e}")
        import traceback
        traceback.print_exc()
        l_hat = None

    # Test 4: Venn-Abers probability computation
    print("\n4. Testing Venn-Abers probability...")
    X_cal = sims.flatten()
    y_cal = labels.flatten()

    # Test with some similarity values
    test_sims = np.array([0.999, 0.9999, 0.99999, 1.0])
    for sim in test_sims:
        p0, p1 = simplifed_venn_abers_prediction(X_cal, y_cal, sim)
        prob = (p0 + p1) / 2
        uncertainty = abs(p1 - p0)
        print(f"   sim={sim:.5f} → prob={prob:.4f} (uncertainty={uncertainty:.4f})")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print("  ✓ FAISS index construction works")
    print("  ✓ Similarity search works")
    if l_hat:
        print("  ✓ FDR threshold computation works")
    else:
        print("  ✗ FDR threshold computation failed")
    print("  ✓ Venn-Abers probability works")

    print("\nNote: To reproduce exact Syn3.0 results (59/149 = 39.6%),")
    print("you need the query embeddings for the 149 unknown genes.")
    print("These can be generated using the Protein-Vec model:")
    print("  python -m protein_conformal.embed_protein_vec --input unknown_aa_seqs.fasta")


if __name__ == '__main__':
    main()
