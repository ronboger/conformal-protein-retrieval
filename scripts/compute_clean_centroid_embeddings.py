#!/usr/bin/env python
"""
Compute CLEAN EC centroid embeddings from precomputed protein embeddings.

Loads the precomputed split100 protein embeddings (100.pt) and computes
EC cluster centroids by averaging all protein embeddings per EC class.

Output:
    data/clean/ec_centroid_embeddings.npy  — (N_ec, 128) float32 array
    data/clean/ec_centroid_metadata.tsv    — EC number for each row

Usage:
    python scripts/compute_clean_centroid_embeddings.py

Requirements:
    - CLEAN_repo/app/data/pretrained/100.pt (precomputed protein embeddings)
    - CLEAN_repo/app/data/split100.csv (protein → EC mapping)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch


CLEAN_REPO = Path("CLEAN_repo/app")
EMB_PATH = CLEAN_REPO / "data/pretrained/100.pt"
CSV_PATH = CLEAN_REPO / "data/split100.csv"
OUT_DIR = Path("data/clean")


def get_ec_id_dict(csv_path: str):
    """Parse split100.csv to get protein→EC and EC→protein mappings.

    Matches the ordering logic in CLEAN_repo/app/src/CLEAN/utils.py.
    """
    id_ec = {}
    ec_id = {}
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, rows in enumerate(reader):
            if i > 0:
                id_ec[rows[0]] = rows[1].split(";")
                for ec in rows[1].split(";"):
                    if ec not in ec_id:
                        ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id


def main():
    if not EMB_PATH.exists():
        print(f"ERROR: {EMB_PATH} not found. Run CLEAN training first.")
        sys.exit(1)
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    print(f"Loading protein embeddings from {EMB_PATH}...")
    emb_train = torch.load(EMB_PATH, map_location="cpu")
    print(f"  Shape: {emb_train.shape}")  # Expected: [241025, 128]

    print(f"Loading EC mapping from {CSV_PATH}...")
    _, ec_id_dict = get_ec_id_dict(str(CSV_PATH))
    ec_list = list(ec_id_dict.keys())
    print(f"  Found {len(ec_list)} unique EC numbers")

    # Compute cluster centers (same logic as CLEAN distance_map.py:get_cluster_center)
    print("Computing EC cluster centroids...")
    centroids = []
    id_counter = 0
    for ec in ec_list:
        n_proteins = len(ec_id_dict[ec])
        emb_cluster = emb_train[id_counter : id_counter + n_proteins]
        centroid = emb_cluster.mean(dim=0)
        centroids.append(centroid.detach().numpy())
        id_counter += n_proteins

    centroids = np.array(centroids, dtype=np.float32)
    print(f"  Centroid matrix shape: {centroids.shape}")  # Expected: (5242, 128)

    # Verify total protein count matches embedding size
    total_counted = sum(len(ec_id_dict[ec]) for ec in ec_list)
    if total_counted != emb_train.shape[0]:
        print(f"  WARNING: CSV protein count ({total_counted}) != embedding rows ({emb_train.shape[0]})")
        print(f"  This may indicate multi-EC proteins or ordering mismatches.")
    else:
        print(f"  Verified: {total_counted} proteins match embedding rows")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npy_path = OUT_DIR / "ec_centroid_embeddings.npy"
    np.save(npy_path, centroids)
    print(f"  Saved embeddings: {npy_path} ({centroids.nbytes / 1024:.1f} KB)")

    tsv_path = OUT_DIR / "ec_centroid_metadata.tsv"
    with open(tsv_path, "w") as f:
        f.write("centroid_idx\tEC_number\tn_proteins\n")
        for i, ec in enumerate(ec_list):
            f.write(f"{i}\t{ec}\t{len(ec_id_dict[ec])}\n")
    print(f"  Saved metadata: {tsv_path}")

    # Verify against calibration data if available
    cal_path = Path("data/clean/clean_new_v_ec_cluster.npy")
    if cal_path.exists():
        cal_data = np.load(cal_path, allow_pickle=True)
        cal_ec_list = cal_data[0]["EC_centroids"]
        n_cal = len(cal_ec_list)
        n_computed = len(ec_list)
        if n_cal == n_computed:
            # Check that EC ordering matches
            mismatches = sum(1 for a, b in zip(cal_ec_list, ec_list) if a != b)
            if mismatches == 0:
                print(f"  Verified: EC ordering matches calibration data ({n_cal} centroids)")
            else:
                print(f"  WARNING: {mismatches}/{n_cal} EC numbers differ from calibration data")
        else:
            print(f"  Note: calibration has {n_cal} centroids, computed {n_computed}")

    print("Done!")


if __name__ == "__main__":
    main()
