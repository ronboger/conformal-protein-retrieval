import numpy as np
import pandas as pd
import argparse

from protein_conformal.util import *


def main(args):
    df = pd.read_csv(args.input)

    if args.precomputed:
        # TODO: if probabilities are precomputed, we can just load them and save them. This will involve the following steps:
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
        pass
    else:
        # Get a probability for each hit based on the distance using Venn-Abers / isotonic regression

        # load calibration data
        data = np.load(
            "/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/pfam_new_proteins.npy",
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
                "/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/pfam_new_proteins.npy",
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
        type=bool,
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
        type=bool,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
