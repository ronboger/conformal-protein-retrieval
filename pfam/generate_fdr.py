import datetime
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from protein_conformal.util import *

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the algorithm')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--n_calib', type=int, default=1000, help='Number of calibration data points')
    parser.add_argument('--delta', type=float, default=0.5, help='Delta value for the algorithm')
    parser.add_argument('--output', type=str, default='/data/ron/protein-conformal/data/pfam_fdr.npy', help='Output file for the results')
    parser.add_argument('--add_date', type=bool, default=True, help='Add date to output file name')
    args = parser.parse_args()
    alpha = args.alpha
    num_trials = args.num_trials
    n_calib = args.n_calib
    delta = args.delta
    # Load the data
    # data = np.load('/data/ron/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy', allow_pickle=True)
    data = np.load('/data/ron/protein-conformal/data/pfam_new_proteins.npy', allow_pickle=True)

    risks = []
    tprs = []
    lhats = []
    fdr_cals = []
    # alpha = 0.1
    # num_trials = 100
    # n_calib = 1000
    for trial in tqdm(range(num_trials)):
        np.random.shuffle(data)
        cal_data = data[:n_calib]
        test_data = data[n_calib:]
        X_cal, y_cal = get_sims_labels(cal_data, partial=False)
        X_test, y_test_exact = get_sims_labels(test_data, partial=False)
        # sims, labels = get_sims_labels(cal_data, partial=False)
        lhat, fdr_cal = get_thresh_FDR(y_cal, X_cal, alpha, delta, N=100)
        lhats.append(lhat)
        fdr_cals.append(fdr_cal)
        # print(X_test.shape)
        # print(y_test_exact.shape)
        risks.append(risk(X_test, y_test_exact, lhat))
        tprs.append(calculate_true_positives(X_test, y_test_exact, lhat))
    
    print("Risk: ", np.mean(risks))
    print("TPR: ", np.mean(tprs))
    print("Lhat: ", np.mean(lhats))
    print("FDR Cal: ", np.mean(fdr_cals))

    output_file = args.output + ('_' + str(datetime.datetime.now().date()) if args.add_date else '')

    np.save(output_file, 
            {'risks': risks,
             'tprs': tprs, 
             'lhats': lhats,
             'fdr_cals': fdr_cals})
    
if __name__ == "__main__":
    # add code for command line arguments
    main()