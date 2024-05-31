
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from protein_conformal.util import *
import datetime

def main(args):
    data = np.load(args.input, allow_pickle=True)
    n_calib = args.n_calib  # Number of calibration data points

    np.random.shuffle(data)
    cal_data = data[:n_calib]
    test_data = data[n_calib:3*n_calib]
    X_cal, y_cal = get_sims_labels(cal_data, partial=False)
    X_test, y_test_exact = get_sims_labels(test_data, partial=False)
    # flatten the data
    X_cal = X_cal.flatten()
    y_cal = y_cal.flatten()
    X_test = X_test.flatten()
    y_test_exact = y_test_exact.flatten()

    sva_results = []
    # generate random indices in the test set
    percent_sva_test = args.percent_sva_test / 100
    print(len(X_test) * args.percent_sva_test)
    i_s = np.random.randint(0, len(X_test), int(len(X_test) * args.percent_sva_test))

    print(f'Running SVA on {len(i_s)} samples with {len(X_test)} test samples.')
    for _, i in tqdm(enumerate(i_s)):
        p_0, p_1 = simplifed_venn_abers_prediction(X_cal, y_cal, X_test[i])
        # np.mean(p_0, p_1)
        sva_results.append((np.mean([p_0, p_1]), X_test[i], y_test_exact[i]))
        # print(f'Prediction: {p_1}, Actual: {y_test[i]}')
    
    df_sva = pd.DataFrame(sva_results, columns=['p', 'x', 'y'])
    output_file = args.output + ('_' + str(datetime.datetime.now().date()) if args.add_date else '') + '.csv'
    print(f'Saving results to {output_file}')
    df_sva.to_csv(output_file, index=False)
    # make bins for p
    df_sva['p_bin'] = pd.cut(df_sva['p'], bins=10)
    print(df_sva.groupby('p_bin')['y'].mean())

if __name__ == "__main__":
    # add code for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/ron/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy', help='Input file for the data')
    # '/data/ron/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy'
    parser.add_argument('--percent_sva_test', type=float, default=.1, help='percent of data to use for SVA testing')
    # parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the algorithm')
    # parser.add_argument('--num_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--n_calib', type=int, default=50, help='Number of calibration data points')
    # parser.add_argument('--delta', type=float, default=0.5, help='Delta value for the algorithm')
    parser.add_argument('--output', type=str, default='/data/ron/protein-conformal/data/sva_results', help='Output file for the results')
    parser.add_argument('--add_date', type=bool, default=True, help='Add date to output file name')
    args = parser.parse_args()
    main(args)