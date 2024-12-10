import numpy as np
import pandas as pd
import argparse
from protein_conformal.util import *


def main(args):
    query_embeddings = np.load(args.query_embedding, allow_pickle=True)
    lookup_embeddings = np.load(args.lookup_embedding, allow_pickle=True)
    query_fasta = read_fasta(args.query_fasta)
    if args.lookup_fasta.endswith(".csv"):
        print("Loading lookup sequences and metadata from csv")
        lookup_df = pd.read_csv(args.lookup_fasta, sep="\t")
        # extract sequences in column "Sequence", and metadata in columns "Pfam" and "Protein names"
        lookup_seqs = lookup_df["Sequence"].values
        lookup_meta = lookup_df[["Pfam", "Protein names"]].values
    else:
        lookup_fasta = read_fasta(args.lookup_fasta)
        lookup_seqs, lookup_meta = lookup_fasta
    print("Loaded data")
    # Extract sequences and metadata
    query_seqs, query_meta = query_fasta

    lookup_database = load_database(lookup_embeddings)
    print("Loaded database")
    k = args.k
    D, I = query(lookup_database, query_embeddings, k)

    # Create DataFrame to store results
    results = []
    for i, (indices, distances) in enumerate(zip(I, D)):
        for idx, distance in zip(indices, distances):
            result = {
                "query_seq": query_seqs[i],
                "query_meta": query_meta[i],
                "lookup_seq": lookup_seqs[idx],
                "lookup_meta": lookup_meta[idx],
                "D_score": distance,
            }
            results.append(result)
    results = pd.DataFrame(results)
    if args.save_inter:
        results.to_csv("inter_" + args.output, index=False)

    # filter results based off of conformal guarantees
    if args.fdr and args.fnr:
        raise ValueError("Cannot control both FDR and FNR")
    if args.fdr:
        if args.fdr_lambda:
            lhat = args.fdr_lambda
        else:
            # TODO: compute FDR as per pfam example
            # lhat, fdr_cal = get_thresh_FDR(
            #     y_cal, X_cal, args.alpha, args.delta, N=100
            # )
            lhat = 0.1
        results = results[results["D_score"] >= lhat] # cosine similarity
    elif args.fnr:
        if args.fnr_lambda:
            lhat = args.fnr_lambda
        else:
            pass
        results = results[results["D_score"] >= lhat]

    results.to_csv(args.output, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process data with conformal guarantees"
    )
    parser.add_argument("--fnr", action='store_true', default=False, help="FNR risk control")
    parser.add_argument("--fdr", action='store_true', default=False, help="FPR risk control")
    parser.add_argument(
        "--fdr_lambda",
        type=float,
        default=0.999980225003127,
        help="FDR lambda hat value if precomputed",
    )
    parser.add_argument(
        "--fnr_lambda",
        type=float,
        # default=0.999980225003127,
        help="FNR lambda hat value if precomputed",
    )
    parser.add_argument(
        "--k", type=int, default=1000, help="maximal number of neighbors with FAISS"
    )
    parser.add_argument(
        "--save_inter", action='store_true', help="save intermediate results"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Alpha value for the algorithm"
    )

    parser.add_argument(
        "--num_trials", type=int, default=100, help="Number of trials to run"
    )
    parser.add_argument(
        "--n_calib", type=int, default=1000, help="Number of calibration data points"
    )
    parser.add_argument(
        "--delta", type=float, default=0.5, help="Delta value for the algorithm"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output file for the results",
    )
    parser.add_argument(
        "--add_date", type=bool, default=True, help="Add date to output file name"
    )
    parser.add_argument(
        "--query_embedding", type=str, default="", help="Query file with the embeddings"
    )
    parser.add_argument(
        "--query_fasta", type=str, default="", help="Input file for the query sequences and metadata"
    )  # TODO: add an option to grab more metadata than just from the fasta file
    parser.add_argument(
	"--lookup_embedding", type=str, default="", help="Lookup embeddings file"
    )
    parser.add_argument(
        "--lookup_fasta", type=str, default="", help="Input file for the lookup sequences and metadata"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
