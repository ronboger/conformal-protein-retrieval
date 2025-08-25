import numpy as np
import pandas as pd
import argparse

from protein_conformal.util import *


def main(args):
    df = pd.read_csv(args.input)
    df_probs = pd.read_csv(args.precomputed_path)

    if args.precomputed:
        # NOTE: if probabilities are precomputed, we can just load them and save them. This will involve the following steps:
        # 1. Load the precomputed probabilities
        # 2. For each hit, see which similarity score bin it corresponds to
        # 3. Assign the probability to the hit based on the similarity score bin

        # df = pd.read_csv(args.precomputed_path)
        # if args.partial:
        #     df["prob_partial"] = df["prob_exact"]
        # else:
        #     df["prob_exact"] = df["prob_exact"]
        # df.to_csv(args.output, index=False)
        # return
        prob_exact_lst, prob_partial_lst = [], []

        for d in df["D_score"]:
            # Check if there are any rows where similarity <= d
            if len(df_probs[df_probs["similarity"] <= d]) > 0:
                lower_bin = df_probs[df_probs["similarity"] <= d].iloc[-1]
            else:
                # If d is smaller than all similarities, use the smallest bin
                lower_bin = df_probs.iloc[0]
                
            # Check if there are any rows where similarity >= d
            if len(df_probs[df_probs["similarity"] >= d]) > 0:
                upper_bin = df_probs[df_probs["similarity"] >= d].iloc[0]
            else:
                # If d is larger than all similarities, use the largest bin
                upper_bin = df_probs.iloc[-1]

            # Get probabilities for lower bin, upper bin (columns "prob_exact_p0", "prob_exact_p1")
            p_0_lower = lower_bin["prob_exact_p0"]
            p_1_lower = lower_bin["prob_exact_p1"]
            p_0_upper = upper_bin["prob_exact_p0"]
            p_1_upper = upper_bin["prob_exact_p1"]

            # Interpolate probabilities
            prob_exact = np.mean([
                min(p_0_lower, p_1_lower, p_0_upper, p_1_upper),
                max(p_0_lower, p_1_lower, p_0_upper, p_1_upper)
            ])
            prob_exact_lst.append(prob_exact)

            if args.partial:
                p_0_lower = lower_bin["prob_partial_p0"]
                p_1_lower = lower_bin["prob_partial_p1"]
                p_0_upper = upper_bin["prob_partial_p0"]
                p_1_upper = upper_bin["prob_partial_p1"]

                prob_partial = np.mean([
                    min(p_0_lower, p_1_lower, p_0_upper, p_1_upper),
                    max(p_0_lower, p_1_lower, p_0_upper, p_1_upper)
                ])
                prob_partial_lst.append(prob_partial)
        df["prob_exact"] = prob_exact_lst
        if args.partial:
            df["prob_partial"] = prob_partial_lst
    else:
        # Get a probability for each hit based on the distance using Venn-Abers / isotonic regression

        # Load calibration data
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

        print("getting exact probabilities")
        p_s = []
        for d in df["D_score"]:
            p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, d)
            p_s.append([p_0, p_1])

        p_s = np.array(p_s)
        abs_p = [np.abs(p[0] - p[1]) for p in p_s]
        df["prob_exact"] = np.mean(p_s, axis=1)

        if args.partial:
            # TODO: this stage may not be necessary, but we noticed sometimes that shuffling the data would mess up the original file
            data = np.load(
                args.cal_data,
                allow_pickle=True,
            )
            print("loading calibration data")
            np.random.shuffle(data)
            cal_data = data[:n_calib]
            X_cal, y_cal = get_sims_labels(cal_data, partial=True)
            X_cal = X_cal.flatten()
            y_cal = y_cal.flatten()

            print("getting partial probabilities")
            p_s = []
            for d in df["D_score"]:
                p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, d)
                p_s.append([p_0, p_1])

            p_s = np.array(p_s)
            abs_p = [np.abs(p[0] - p[1]) for p in p_s]
            df["prob_partial"] = np.mean(p_s, axis=1)

    print("saving df new probabilities")
    df.to_csv(
        args.output,
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser("Get probabilities for similarity scores")
    parser.add_argument(
        "--precomputed",
        action='store_true', 
        default=False,
        help="Use precomputed probabilities on similarity scores",
    )
    parser.add_argument(
        "--precomputed_path",
        type=str,
        default="",
        help="Path to precomputed probabilities. This will have probabilities for both partial and exact hits.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/groups/doudna/projects/ronb/conformal_backup/results_no_probs.csv",
        help="Input tabular data with similarity scores and metadata.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/groups/doudna/projects/ronb/conformal_backup/results_with_probs.csv",
        help="Output file for the results",
    )
    parser.add_argument(
        "--partial",
        action='store_true', 
        default=False,
        help="Return probability of partial hits given similarity scores",
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
    parser.add_argument(
        "--cal_data", type=str, default="/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/pfam_new_proteins.npy", help="Path to calibration data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
