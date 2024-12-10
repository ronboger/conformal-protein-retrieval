import numpy as np
import pandas as pd
import argparse

from protein_conformal.util import *
import tqdm

def main(args):
    sim2prob = pd.DataFrame(columns=["similarity", "prob_exact_p0", "prob_exact_p1", "prob_partial_p0", "prob_partial_p1"])

    # Get a probability for each hit based on the distance using Venn-Abers / isotonic regression
    # load calibration data
    data = np.load(
        args.cal_data,
        allow_pickle=True,
    )
    print("loading calibration data")
    n_calib = args.n_calib
    np.random.shuffle(data)
    cal_data = data[:n_calib]
    X_cal, y_cal = get_sims_labels(cal_data, partial=False)
    X_cal = X_cal.flatten()
    y_cal = y_cal.flatten()

    print("Get probability distribution for a grid of similarity bins")
    min_sim, max_sim = X_cal.min(), X_cal.max()
    # create bins for similarity scores to get SVA probabilities
    bins = np.linspace(min_sim, max_sim, args.n_bins)
    for d in tqdm.tqdm(bins, desc="Calculating probabilities"):
        p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, d)
        sim2prob = sim2prob.append(
            {
                "similarity": d,
                "prob_exact_p0": p_0,
                "prob_exact_p1": p_1,
            },
            ignore_index=True,
        )

    if args.partial:
        # TODO: this stage may not be necessary, but we noticed sometimes that shuffling the data would mess up the original file
        data = np.load(
            args.cal_data,
            allow_pickle=True,
        )
        print("loading calibration data")
        np.random.shuffle(data)
        cal_data = data[:n_calib]
        # partial = True
        X_cal, y_cal = get_sims_labels(cal_data, partial=True)
        X_cal = X_cal.flatten()
        y_cal = y_cal.flatten()

        print("getting partial probabilities")
        for i, d in tqdm.tqdm(enumerate(bins), desc="Calculating partial probabilities", total=len(bins)):
            p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, d)
            # add to column "prob_partial" for the row with the appropriate similarity score bin, i
            sim2prob.loc[i, "prob_partial_p0"] = p_0
            sim2prob.loc[i, "prob_partial_p1"] = p_1
    print("saving df new probabilities")
    sim2prob.to_csv(
        args.output,
        index=False,
    )


    # save the probabilities for the exact hits, then calculate the probabilities for partial hits
    print("saving df exact probabilities")
    sim2prob.to_csv(
        args.output,
        index=False,
    )

def parse_args():
    parser = argparse.ArgumentParser("Get probabilities for similarity score distribution of a given dataset, so its precomputed")
  
    parser.add_argument(
        "--cal_data",
        type=str,
        default="/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/pfam_new_proteins.npy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/groups/doudna/projects/ronb/conformal_backup/results_with_probs.csv",
        help="Output file for the dataframe mapping similarities to probabilities",
    )
    parser.add_argument(
        "--partial",
        type=bool,
        default=False,
        help="Generate probability of partial hits given similarity score bins",
    )
    parser.add_argument(
        "--n_bins", type=int, default=100, help="Number of bins to use for the similarity scores"
    )
    # parser.add_argument(
    #     "--alpha", type=float, default=0.1, help="Alpha value for the algorithm"
    # )
    # parser.add_argument(
    #     "--num_trials", type=int, default=100, help="Number of trials to run"
    # )
    parser.add_argument(
        "--n_calib", type=int, default=100, help="Number of calibration data points"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
