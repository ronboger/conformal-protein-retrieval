#!/usr/bin/env python3
"""Convert AFDB metadata from .npy to .tsv format.

Reads accession IDs from afdb_metadata.npy and sequences from AFDB_sequences.fasta,
then writes a TSV compatible with the Gradio interface (Entry + Sequence columns).

Uses only numpy and stdlib (no pandas/biopython required).
"""
import numpy as np
import csv
import sys
import os

BACKUP_DIR = "/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data"
OUTPUT_DIR = "/groups/doudna/projects/ronb/conformal-protein-retrieval/data/afdb"


def parse_fasta_simple(fasta_path):
    """Parse FASTA file into dict of {accession_id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                current_seq.append(line)
    # Don't forget the last record
    if current_id is not None:
        seqs[current_id] = "".join(current_seq)
    return seqs


def main():
    # Load the metadata .npy to get accession IDs in the exact same order as embeddings
    meta_path = os.path.join(BACKUP_DIR, "afdb_metadata.npy")
    meta_ids = np.load(meta_path, allow_pickle=True)
    print(f"Metadata .npy: {len(meta_ids)} accession IDs")
    print(f"First 3: {list(meta_ids[:3])}")

    # Load embeddings to verify count (mmap to avoid loading all into RAM)
    emb_path = os.path.join(BACKUP_DIR, "afdb_embeddings_protein_vec.npy")
    emb = np.load(emb_path, mmap_mode="r")
    print(f"Embeddings shape: {emb.shape}")

    assert len(meta_ids) == emb.shape[0], (
        f"Mismatch: {len(meta_ids)} IDs vs {emb.shape[0]} embeddings"
    )

    # Parse FASTA to get sequences keyed by accession ID
    fasta_path = os.path.join(BACKUP_DIR, "AFDB_sequences.fasta")
    print(f"Parsing FASTA file: {fasta_path}")
    fasta_seqs = parse_fasta_simple(fasta_path)
    print(f"FASTA records parsed: {len(fasta_seqs)}")

    # Write TSV directly (avoids needing pandas)
    output_path = os.path.join(OUTPUT_DIR, "afdb_metadata.tsv")
    missing = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["", "Entry", "Sequence"])  # Header with index column
        for i, acc_id in enumerate(meta_ids):
            seq = fasta_seqs.get(acc_id, "")
            if not seq:
                missing += 1
            writer.writerow([i, acc_id, seq])

    print(f"Missing sequences: {missing}")
    print(f"Saved to {output_path}")
    print(f"Row count matches embeddings: {len(meta_ids) == emb.shape[0]}")

    # Quick verification
    with open(output_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print(f"\nHeader: {header}")
        for i, row in enumerate(reader):
            if i < 3:
                print(f"Row {i}: Entry={row[1]}, Seq={row[2][:50]}...")
            else:
                break


if __name__ == "__main__":
    main()
