#!/usr/bin/env python
"""
Verify JCVI Syn3.0 annotation results (Paper Figure 2A).

This script reproduces the key result from the paper: 39.6% (59/149) of genes
with unknown function in JCVI Syn3.0 minimal genome received confident
functional annotations at FDR α=0.1.

Required data files (see docs/INSTALLATION.md for download instructions):
- data/gene_unknown/unknown_aa_seqs.npy: Protein-Vec embeddings of 149 unknown genes
- data/gene_unknown/unknown_aa_seqs.fasta: FASTA sequences (for metadata)
- data/lookup_embeddings.npy: UniProt lookup embeddings (from Zenodo)
- data/lookup_embeddings_meta_data.tsv: UniProt metadata with Pfam annotations
- data/pfam_new_proteins.npy: Calibration data for Venn-Abers (from Zenodo)

Expected output:
- 59 hits out of 149 queries (39.6%) at FDR threshold λ ≈ 0.999980

Usage:
    python scripts/verify_syn30.py
    python scripts/verify_syn30.py --alpha 0.1 --output results/syn30_hits.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_conformal.util import (
    read_fasta,
    load_database,
    query,
    simplifed_venn_abers_prediction,
    get_sims_labels,
)


def load_fdr_threshold(fdr_file: Path = None, alpha: float = 0.1) -> float:
    """
    Load pre-computed FDR threshold or use hardcoded value from paper.

    The FDR threshold is computed using Learn-Then-Test (LTT) calibration.
    For α=0.1, the mean threshold across calibration runs is 0.999980225003127.
    """
    if fdr_file and fdr_file.exists():
        fdr_data = np.load(fdr_file, allow_pickle=True).item()
        return np.mean(fdr_data['lhats'])

    # Hardcoded value from paper/notebook for α=0.1
    # This is the average threshold from 100 calibration trials
    if alpha == 0.1:
        return 0.999980225003127
    else:
        raise ValueError(
            f"No pre-computed threshold for alpha={alpha}. "
            "Please provide an FDR file or use alpha=0.1."
        )


def verify_syn30(
    query_embeddings_path: Path,
    query_fasta_path: Path,
    lookup_embeddings_path: Path,
    lookup_metadata_path: Path,
    calibration_data_path: Path,
    fdr_threshold_path: Path = None,
    alpha: float = 0.1,
    output_csv: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Run the JCVI Syn3.0 verification experiment.

    Returns dict with:
        - n_queries: Total number of query proteins
        - n_hits: Number of proteins with confident hits
        - hit_rate: Fraction of proteins with hits
        - threshold: FDR threshold used
        - hits_df: DataFrame with detailed hit information
    """

    if verbose:
        print("=" * 60)
        print("JCVI Syn3.0 Annotation Verification")
        print("=" * 60)

    # Load query embeddings (149 unknown genes)
    if verbose:
        print(f"\nLoading query embeddings from {query_embeddings_path}...")
    query_embeddings = np.load(query_embeddings_path)
    n_queries = query_embeddings.shape[0]
    if verbose:
        print(f"  Loaded {n_queries} query embeddings, shape: {query_embeddings.shape}")

    # Load query FASTA for metadata
    if verbose:
        print(f"\nLoading query FASTA from {query_fasta_path}...")
    query_fastas, query_metadata = read_fasta(str(query_fasta_path))
    if verbose:
        print(f"  Loaded {len(query_fastas)} sequences")

    # Load lookup database (UniProt with Pfam annotations)
    if verbose:
        print(f"\nLoading lookup embeddings from {lookup_embeddings_path}...")
    embeddings = np.load(lookup_embeddings_path)
    if verbose:
        print(f"  Loaded {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")

    if verbose:
        print(f"\nLoading lookup metadata from {lookup_metadata_path}...")
    lookup_proteins_meta = pd.read_csv(lookup_metadata_path, sep="\t")
    if verbose:
        print(f"  Loaded metadata for {len(lookup_proteins_meta)} proteins")

    # Filter to proteins with Pfam annotations
    column = 'Pfam'
    col_lookup = lookup_proteins_meta[~lookup_proteins_meta[column].isnull()]
    col_lookup_embeddings = embeddings[col_lookup.index]
    col_meta_data = col_lookup[column].values
    if verbose:
        print(f"  {len(col_lookup)} proteins have Pfam annotations")

    # Build FAISS index
    if verbose:
        print("\nBuilding FAISS index...")
    lookup_database = load_database(col_lookup_embeddings)

    # Query for nearest neighbors
    if verbose:
        print("Querying for nearest neighbors (k=1)...")
    k = 1
    D, I = query(lookup_database, query_embeddings, k)
    D_max = np.max(D, axis=1)

    # Load FDR threshold
    l_hat = load_fdr_threshold(fdr_threshold_path, alpha)
    if verbose:
        print(f"\nFDR threshold (α={alpha}): λ = {l_hat:.12f}")

    # Count hits
    hits_mask = D_max > l_hat
    n_hits = hits_mask.sum()
    hit_rate = n_hits / n_queries

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RESULTS")
        print(f"{'=' * 60}")
        print(f"Total queries:     {n_queries}")
        print(f"Confident hits:    {n_hits}")
        print(f"Hit rate:          {hit_rate:.1%} (expected: 39.6%)")
        print(f"{'=' * 60}")

    # Compute Venn-Abers probabilities for hits
    if verbose and calibration_data_path.exists():
        print("\nComputing Venn-Abers probabilities...")
        data = np.load(calibration_data_path, allow_pickle=True)
        n_calib = 100
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(data)
        cal_data = data[:n_calib]
        X_cal, y_cal = get_sims_labels(cal_data, partial=False)
        X_cal = X_cal.flatten()
        y_cal = y_cal.flatten()

        p_s = []
        for d in D:
            p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, d)
            p_s.append((p_0 + p_1) / 2)  # Point estimate
        p_s = np.array(p_s)

        print(f"  Mean probability for hits: {np.mean(p_s[hits_mask]):.3f}")
    else:
        p_s = np.full(n_queries, np.nan)

    # Build results DataFrame
    results_data = {
        'query_name': query_metadata,
        'query_sequence': query_fastas,
        'similarity': D_max,
        'probability': p_s,
        'is_hit': hits_mask,
    }

    # Add Pfam annotations for hits
    filtered_I = I[hits_mask, 0]
    pfam_annotations = np.array([''] * n_queries, dtype=object)
    pfam_annotations[hits_mask] = col_meta_data[filtered_I]
    results_data['pfam_annotation'] = pfam_annotations

    results_df = pd.DataFrame(results_data)
    hits_df = results_df[results_df['is_hit']].copy()

    if output_csv:
        if verbose:
            print(f"\nSaving results to {output_csv}...")
        hits_df.to_csv(output_csv, index=False)

    return {
        'n_queries': n_queries,
        'n_hits': n_hits,
        'hit_rate': hit_rate,
        'threshold': l_hat,
        'hits_df': hits_df,
        'results_df': results_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify JCVI Syn3.0 annotation results (Paper Figure 2A)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Base data directory'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='FDR level (default: 0.1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV file for hit results'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    # Define file paths
    query_embeddings_path = data_dir / 'gene_unknown' / 'unknown_aa_seqs.npy'
    query_fasta_path = data_dir / 'gene_unknown' / 'unknown_aa_seqs.fasta'
    lookup_embeddings_path = data_dir / 'lookup_embeddings.npy'
    lookup_metadata_path = data_dir / 'lookup_embeddings_meta_data.tsv'
    calibration_data_path = data_dir / 'pfam_new_proteins.npy'

    # Check for missing files
    missing_files = []
    for path in [query_embeddings_path, query_fasta_path,
                 lookup_embeddings_path, lookup_metadata_path]:
        if not path.exists():
            missing_files.append(path)

    if missing_files:
        print("ERROR: Missing required data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nSee docs/INSTALLATION.md for download instructions.")
        print("\nQuick fix for Syn3.0 data:")
        print("  The unknown_aa_seqs.npy and .fasta files contain the 149 genes")
        print("  from JCVI Syn3.0 with unknown function. These need to be")
        print("  generated using the Protein-Vec embedding model.")
        sys.exit(1)

    # Run verification
    results = verify_syn30(
        query_embeddings_path=query_embeddings_path,
        query_fasta_path=query_fasta_path,
        lookup_embeddings_path=lookup_embeddings_path,
        lookup_metadata_path=lookup_metadata_path,
        calibration_data_path=calibration_data_path,
        alpha=args.alpha,
        output_csv=args.output,
        verbose=not args.quiet,
    )

    # Verify expected result
    expected_hits = 59
    expected_rate = 0.396

    if results['n_hits'] == expected_hits:
        print(f"\n✓ VERIFICATION PASSED: {results['n_hits']} hits matches expected {expected_hits}")
    else:
        print(f"\n✗ VERIFICATION FAILED: Got {results['n_hits']} hits, expected {expected_hits}")
        print("  This may be due to different calibration data or random seed.")

    return results


if __name__ == '__main__':
    main()
