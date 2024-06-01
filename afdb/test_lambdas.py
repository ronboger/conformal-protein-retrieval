import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def find_elbow_point(x, y):
    """
    Finds the elbow point in the given data.

    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.

    Returns:
    int: The index of the elbow point.
    """

    # Calculate the slope of the line connecting the first and last points
    line_vector = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vector = line_vector / np.linalg.norm(line_vector)

    # Project each point onto the line
    projection = np.dot(np.vstack((x - x[0], y - y[0])).T, line_vector)
    projection = np.outer(projection, line_vector)
    proj_x = x[0] + projection[:, 0]
    proj_y = y[0] + projection[:, 1]

    # Calculate the distance from each point to the line
    distances = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

    # Find the index of the maximum distance
    elbow_index = np.argmax(distances)

    return elbow_index

def main(args):
    # open the input file
    data = np.load(args.input, allow_pickle=True)['arr_0']

    lam = args.lam

    results = []

    for result in tqdm(data):
        z_scores = np.array(result['Z_score'])
        s_i = np.array(result['S_i'])
        
        # calculate elbow for z_scores
        idx = find_elbow_point(np.array([x for x in range(np.sum([z_scores > 0]))]) , np.sort(z_scores[z_scores>0])[::-1])
        elbow_z = np.sort(z_scores)[::-1][idx]
        if elbow_z < 2:
            elbow_z = 2

        # calculate statistics for relating z_scores to s_i
        total_samples_above_2 = len(z_scores[(z_scores > 0)])
        total_samples_above_elbow = len(z_scores[(z_scores > elbow_z)])
        missed_samples_above_elbow = z_scores[(z_scores > elbow_z) & (s_i < lam)]
        found_samples_above_elbow = z_scores[(z_scores > elbow_z) & (s_i >= lam)]
        missed_samples_above_2 = z_scores[(z_scores > 0) & (s_i < lam)]
        found_samples_above_2 = z_scores[(z_scores > 0) & (s_i >= lam)]
        total_samples_lambda = len(z_scores[(s_i >= lam)])

        false_discoveries_elbow = z_scores[(z_scores < elbow_z) & (s_i > lam)]
        false_discoveries_2 = z_scores[(z_scores == 0) & (s_i > lam)]

        # calculate the statistics
        false_negative_rate_elbow = len(missed_samples_above_elbow) / total_samples_above_elbow
        true_positive_rate_elbow = len(found_samples_above_elbow) / total_samples_above_elbow
        false_negative_rate_2 = len(missed_samples_above_2) / total_samples_above_2
        true_positive_rate_2 = len(found_samples_above_2) / total_samples_above_2
        false_discovery_rate_elbow = len(false_discoveries_elbow) / total_samples_lambda
        false_discovery_rate_2 = len(false_discoveries_2) / total_samples_lambda
        fraction_of_samples_above_lambda = total_samples_lambda / len(z_scores)
        fraction_of_z_0 = len(z_scores[z_scores == 0]) / len(z_scores)

        results.append({
            'FNR_elbow': false_negative_rate_elbow,
            'TPR_elbow': true_positive_rate_elbow,
            'FNR_2': false_negative_rate_2,
            'TPR_2': true_positive_rate_2,
            'FDR_elbow': false_discovery_rate_elbow,
            'FDR_2': false_discovery_rate_2,
            'frac_samples_above_lambda': fraction_of_samples_above_lambda,
            'frac_z_0': fraction_of_z_0,
            'elbow_z': elbow_z
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'Saved results to {args.output} with lambda {lam}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.99974871, help='lambda for FNR')
    parser.add_argument('--input', type=str, default='/data/ron/protein-conformal/data/dali_results_protein_vec.npz', help='Input file for the data')
    parser.add_argument('--output', type=str, default='/data/ron/protein-conformal/data/dali_results_protein_vec_lam.csv', help='Output file for the results')
    args = parser.parse_args()
    main(args)