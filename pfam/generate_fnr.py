import numpy as np
import pandas as pd
import argparse
import datetime
import os
from tqdm import tqdm
from protein_conformal.util import *

def main():
    parser = argparse.ArgumentParser(description='Generate FNR thresholds for different alpha values')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the algorithm')
    parser.add_argument('--partial', action='store_true', help='Partial hits')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--n_calib', type=int, default=1000, help='Number of calibration data points')
    parser.add_argument('--output', type=str, default='/data/ron/protein-conformal/data/pfam_fnr.npy', help='Output file for the results')
    parser.add_argument('--add_date', action='store_true', help='Add date to output file name')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the pfam data file')
    args = parser.parse_args()
    alpha = args.alpha
    num_trials = args.num_trials
    n_calib = args.n_calib
    partial = args.partial
    
    # Determine data path - use relative path if not specified
    if args.data_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_path = os.path.join(project_root, 'data', 'pfam_new_proteins.npy')
    else:
        data_path = args.data_path
    
    print(f"Loading data from: {data_path}")
    
    # Load the data
    data = np.load(data_path, allow_pickle=True)

    fnrs = []
    lhats = []
    tprs = []
    fprs = []
    
    for trial in tqdm(range(num_trials)):
        np.random.shuffle(data)
        cal_data = data[:n_calib]
        test_data = data[n_calib:]
        X_cal, y_cal = get_sims_labels(cal_data, partial=partial)
        X_test, y_test_exact = get_sims_labels(test_data, partial=partial)
        _, y_test_partial = get_sims_labels(test_data, partial=True)
        
        lhat = get_thresh_new(X_cal, y_cal, alpha)
        lhats.append(lhat)
        
        error, fraction_inexact, error_partial, fraction_partial, fpr = validate_lhat_new(X_test, y_test_partial, y_test_exact, lhat)
        fnrs.append(error)
        fprs.append(fpr)
        tprs.append(calculate_true_positives(X_test, y_test_exact, lhat))
    
    print("FNR: ", np.mean(fnrs))
    print("TPR: ", np.mean(tprs))
    print("Lhat: ", np.mean(lhats))
    print("FPR: ", np.mean(fprs))

    output_file = args.output + ('_' + str(datetime.datetime.now().date()) if args.add_date else '') + '.npy'

    np.save(output_file, 
            {'fnrs': fnrs,
             'tprs': tprs, 
             'lhats': lhats,
             'fprs': fprs})
    
if __name__ == "__main__":
    main()