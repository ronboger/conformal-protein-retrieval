import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from protein_conformal.util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the algorithm')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--n_calib', type=int, default=1000, help='Number of calibration data points')
    args = parser.parse_args()
    alpha = args.alpha
    num_trials = args.num_trials
    n_calib = args.n_calib

    # Load the data
    data = np.load('/data/ron/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy', allow_pickle=True)

if __name__ == "__main__":
    # add code for command line arguments
    main()