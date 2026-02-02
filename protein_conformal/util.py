from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy.stats import binom, norm
from Bio.Align import PairwiseAligner
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch


def calculate_pppl(model, tokenizer, sequence):
    token_ids = tokenizer.encode(sequence, return_tensors="pt")
    input_length = token_ids.size(1)
    log_likelihood = 0.0

    for i in range(input_length):
        # Create a copy of the token IDs
        masked_token_ids = token_ids.clone()
        # Mask a token that we will try to predict back
        masked_token_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            output = model(masked_token_ids)
            logit_prob = torch.nn.functional.log_softmax(output.logits, dim=-1)

        log_likelihood += logit_prob[0, i, token_ids[0, i]]

    # Calculate the average log likelihood per token
    avg_log_likelihood = log_likelihood / input_length

    # Compute and return the pseudo-perplexity
    pppl = torch.exp(-avg_log_likelihood)
    return pppl.item()


def read_fasta(fasta_file):
    """Read a FASTA file and return a list of sequences and metadata"""
    sequences = []
    metadata = []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                metadata.append(line)
                sequences.append("")
            else:
                if "*" in line:
                    print("removing * from amino acid sequence. TODO, what are these?")
                    line = line.replace("*", "")
                sequences[-1] += line
    return sequences, metadata


# Define the sequence identity function using Bio.Align.PairwiseAligner
def seq_identity(seq1, seq2):
    """
    Calculate the sequence identity between two sequences using pairwise alignment.

    Parameters:
    seq1 (str): First sequence
    seq2 (str): Second sequence

    Returns:
    float: Sequence identity percentage
    """
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    seq1_aligned, seq2_aligned = best_alignment.aligned

    # Calculate identity
    matches = sum(a == b for a, b in zip(seq1, seq2) if a == b)
    length = max(len(seq1), len(seq2))
    return matches / length * 100


def get_sims_labels(data, partial=False):
    """
    Get the similarities and labels from the given data.

    Parameters:
    - data: A list of query data.
    - partial: A boolean indicating whether to use partial hits or exact hits.
    exact: Pfam1001 == Pfam1001
    partial: Pfam1001 in [Pfam1001,Pfam1002], where [Pfam1001,Pfam1002] is the set of Pfam domains in a protein.

    Returns:
    - sims: A list of similarity scores.
    - labels: A list of labels.
    """

    sims = np.stack([query["S_i"] for query in data], axis=0)
    # TODO: may want to just return both partial and exact labels
    if partial:
        labels = np.stack(
            [
                (
                    np.any(query["partial"], axis=1)
                    if isinstance(query["partial"][0], list)
                    else query["partial"]
                )
                for query in data
            ],
            axis=0,
        )
    else:
        labels = np.stack([query["exact"] for query in data], axis=0)
    # sims = []
    # labels = []
    # for query in data:
    #     similarity = query["S_i"]
    #     sims += similarity.tolist()
    #     if partial:
    #         labels_to_append = np.logical_or.reduce(query["partial"], axis=1).tolist()
    #     else:
    #         labels_to_append = query["exact"]
    #     labels += labels_to_append
    return sims, labels


def get_arbitrary_attribute(data, attribute: str):
    # get an arbitrary attribute from the data
    attributes = []
    for query in data:
        attribute = query[attribute]
        attributes += attribute.tolist()
    return attributes


def get_thresh_new_FDR(X, Y, alpha):
    # conformal risk control

    # all_sim_exact = []
    all_sim_exact = X.flatten()[~Y.flatten()]
    # by default this includes 10% of overall false discoveries, but it is not 10% of all total discoveries

    n = len(all_sim_exact)
    if n > 0:
        lhat = np.quantile(
            all_sim_exact,
            np.maximum(alpha - (1 - alpha) / n, 0),
            method="lower",
        )
    else:
        lhat = 0

    return lhat


# validate lhat
def validate_lhat_new(X, Y_partial, Y_exact, lhat):
    """
    Validates the value of lhat against the given data.

    Args:
        data (list): A list of dictionaries representing the data.
        lhat (float): The threshold value to validate against.

    Returns:
        tuple: A tuple containing the following values:
            - The ratio of missed exact matches to the total number of exact matches.
            - The ratio of identified inexact matches to the total number of identified matches.
            - The ratio of missed partial matches to the total number of partial matches.
            - The ratio of identified partial matches to the total number of identified matches.
    """

    # print('there')
    # X_exact = X[Y_exact].flatten()
    # X_partial = X[Y_partial].flatten()
    # total_missed = (X_exact < lhat).sum() # number of false negatives
    # total_missed_partial = (X_partial < lhat).sum() # number of false negatives for partial hits
    # total_partial_identified = (X_partial >= lhat).sum() # number of true positives for partial hits
    # total_partial = len(X_partial) # total number of partial hits
    # total_exact = len(X_exact) # total number of exact hits
    # total_inexact_identified = (X[~Y_exact] >= lhat).sum() # number of false positives, also includes partial hits
    # total_identified = (X >= lhat).sum() # total number of true positives

    # return total_missed/total_exact, total_inexact_identified/total_identified, total_missed_partial/total_partial, total_partial_identified/total_identified, total_inexact_identified/total_missed

    X_flat = X.flatten()
    X_exact = X_flat[Y_exact.flatten()]
    X_partial = X_flat[Y_partial.flatten()]

    # False negatives
    total_missed = (X_exact < lhat).sum()
    total_missed_partial = (X_partial < lhat).sum()

    # True positives
    total_partial_identified = (X_partial >= lhat).sum()
    total_exact_identified = (X_exact >= lhat).sum()

    # Totals
    total_partial = len(X_partial)
    total_exact = len(X_exact)
    total_identified = (X_flat >= lhat).sum()

    # False positives
    total_inexact_identified = ((X_flat >= lhat) & ~Y_exact.flatten()).sum()

    error = total_missed / total_exact if total_exact > 0 else 0
    fraction_inexact = (
        total_inexact_identified / total_identified if total_identified > 0 else 0
    )
    error_partial = total_missed_partial / total_partial if total_partial > 0 else 0
    fraction_partial = (
        total_partial_identified / total_identified if total_identified > 0 else 0
    )

    # False postive rate
    total_negative = len(X_flat) - total_exact
    fpr = total_inexact_identified / total_negative if total_negative > 0 else 0

    # false discovery rate
    # total_positive = total_exact + total_inexact_identified
    # fdr = total_inexact_identified / total_positive if total_positive > 0 else 0

    return error, fraction_inexact, error_partial, fraction_partial, fpr


def get_thresh_new(X, Y, alpha):
    # conformal risk control

    # all_sim_exact = []
    all_sim_exact = X.flatten()[Y.flatten()]
    n = len(all_sim_exact)
    if n > 0:
        lhat = np.quantile(
            all_sim_exact,
            np.maximum(alpha - (1 - alpha) / n, 0),
            method="lower",
        )
    else:
        lhat = 0

    return lhat


def get_thresh(data, alpha):
    # conformal risk control
    # TODO: refactor this to just take in X, Y, and alpha

    all_sim_exact = []
    for query in data:
        idx = query["exact"]
        similarity = query["S_i"]
        sims_to_append = similarity[idx]
        all_sim_exact += list(sims_to_append)
        n = len(all_sim_exact)
        if n > 0:
            lhat = np.quantile(
                all_sim_exact,
                np.maximum(alpha - (1 - alpha) / n, 0),
                method="lower",
            )
        else:
            lhat = 0
    return lhat


# Bentkus p value
def bentkus_p_value(r_hat, n, alpha):
    return binom.cdf(np.ceil(n * r_hat), n, alpha / np.e)


# def clt_p_value(r_hat,n,alpha):
# TODO: we may have wanted to do a different implementation of this


def clt_p_value(r_hat, std_hat, n, alpha):
    z = (r_hat - alpha) / (std_hat / np.sqrt(n))
    p_value = norm.cdf(z)
    return p_value


def percentage_of_discoveries(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    return total_discoveries.mean() / len(labels)  # or sims.shape[1]


def risk_1d(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum()
    false_discoveries = ((1 - labels) * (sims >= lam)).sum()
    total_discoveries = np.maximum(total_discoveries, 1)
    fdr = false_discoveries / total_discoveries
    return fdr


def risk(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries / total_discoveries).mean()


def calculate_true_positives(sims, labels, lam):
    # TPR: Number of true matches / number of matches
    total_matches = labels.sum(axis=1)
    true_matches = (labels & (sims >= lam)).sum(axis=1)
    total_matches = np.maximum(total_matches, 1)
    return (true_matches / total_matches).mean()


# def calculate_false_negatives(sims, labels, lam):
#     # FNR: Number of false non-matches / number of non-matches
#     total_non_matches = labels.sum(axis=1)
#     false_non_matches = (labels & (sims < lam)).sum(axis=1)
#     total_non_matches = np.maximum(total_non_matches, 1)
#     return (false_non_matches / total_non_matches).mean()


def calculate_false_negatives(sims, labels, lam):
    # True positives: actual positives (labels == 1) with similarity >= threshold
    true_positives = ((labels == 1) & (sims >= lam)).sum()

    # False negatives: actual positives (labels == 1) with similarity < threshold
    false_negatives = ((labels == 1) & (sims < lam)).sum()

    # Total actual positives
    total_positives = (labels == 1).sum()

    # False negative rate
    if total_positives == 0:
        print("No actual positives")
        return 0
    return false_negatives / total_positives


def risk_no_empties(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    idx = total_discoveries > 0
    total_discoveries = total_discoveries[idx]
    false_discoveries = false_discoveries[idx]
    return (false_discoveries / total_discoveries).mean()


def std_loss(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries / total_discoveries).std()


def std_loss_1d(sims, labels, lam):
    """Standard deviation of loss for 1D arrays (single sample)."""
    # For 1D arrays, we compute the FDR directly without std across samples
    # Return a small value to avoid division issues in CLT p-value
    return 0.01


def get_thresh_FDR(labels, sims, alpha, delta=0.5, N=5000):
    """
    Calculate the threshold value for controlling the False Discovery Rate (FDR) using Learn then Test (LTT).

    Parameters:
    - labels (numpy.ndarray): The labels of the data points. Can be 1D or 2D.
    - sims (numpy.ndarray): The similarity scores of the data points. Can be 1D or 2D.
    - alpha (float): The significance level for controlling the FDR.
    - delta (float, optional): p-value limit. Defaults to 0.5.
    - N (int, optional): The number of lambda values to consider. Defaults to 5000.

    Returns:
    - lhat (float): The threshold value for controlling the FDR.
    - risk_fdr (float): The FDR risk at the threshold.

    """
    # Detect if inputs are 1D or 2D and use appropriate functions
    is_1d = labels.ndim == 1

    if is_1d:
        risk_fn = risk_1d
        std_fn = std_loss_1d
    else:
        risk_fn = risk
        std_fn = std_loss

    n = len(labels)
    lambdas = np.linspace(sims.min(), sims.max(), N)
    risks = np.array([risk_fn(sims, labels, lam) for lam in lambdas])
    stds = np.array([std_fn(sims, labels, lam) for lam in lambdas])
    eps = 1e-6
    stds = np.maximum(stds, eps)
    pvals = np.array([clt_p_value(r, s, n, alpha) for r, s in zip(risks, stds)])

    below = pvals <= delta
    # Pick the smallest lambda such that all lambda above it have p-value below delta
    pvals_satisfy_condition = np.array([np.all(below[i:]) for i in range(N)])
    lhat = lambdas[np.argmax(pvals_satisfy_condition)]
    risk_fdr = risk_fn(sims, labels, lhat)
    return lhat, risk_fdr


def get_isotone_regression(X, y):
    # Standard isotonic regression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)
    return ir


def validate_lhat(data, lhat):
    """
    Validates the value of lhat against the given data.

    Args:
        data (list): A list of dictionaries representing the data.
        lhat (float): The threshold value to validate against.

    Returns:
        tuple: A tuple containing the following values:
            - The ratio of missed exact matches to the total number of exact matches.
            - The ratio of identified inexact matches to the total number of identified matches.
            - The ratio of missed partial matches to the total number of partial matches.
            - The ratio of identified partial matches to the total number of identified matches.
    """
    total_missed = 0  # exact hits missed
    total_missed_partial = 0  # partial hits missed
    total_exact = 0  # total of exact hits

    # TODO: what is the difference between these?
    total_inexact_identified = 0
    total_identified = 0
    total_partial = 0  # total number of true partial hits
    total_partial_identified = 0  # total number of partial hits >= lhat

    # TODO: there's almost certainly a way to do this without looping through the data
    for query in data:
        idx = query["exact"]
        # if partial has multiple rows, we want to take the logical or of all of them. Otherwise just set it to the single row
        # check if there is one or more rows
        # query['partial'] = np.array(query['partial'])
        if len(np.array(query["partial"]).shape) > 1:
            # TODO: should this be any or logical_or?
            idx_partial = np.logical_or.reduce(query["partial"], axis=1)
        else:
            idx_partial = query["partial"]

        sims = query["S_i"]
        sims_exact = sims[idx]  # exact hits
        sims_partial = sims[idx_partial]  # partial hits
        total_missed += (sims_exact < lhat).sum()  # number of false negatives

        # TODO: are there any divisions by zero here?
        total_missed_partial += (
            sims_partial < lhat
        ).sum()  # number of false negatives for partial hits
        total_partial_identified += (
            sims_partial >= lhat
        ).sum()  # number of true positives for partial hits
        total_partial += len(sims_partial)  # total number of partial hits

        total_exact += len(sims_exact)  # total number of exact hits
        total_inexact_identified += (
            sims[~np.array(idx)] >= lhat
        ).sum()  # number of false positives
        total_identified += (sims >= lhat).sum()  # total number of true positives
    return (
        total_missed / total_exact,
        total_inexact_identified / total_identified,
        total_missed_partial / total_partial,
        total_partial_identified / total_identified,
    )


# Simplified version of Venn Abers prediction
def simplifed_venn_abers_prediction(X_cal, Y_cal, X_test_data_point):
    """
    Perform simplified Venn Abers prediction.

    Args:
        X_cal (numpy.ndarray): The similarity scores of the calibration data.
        Y_cal (numpy.ndarray): The labels of the calibration data.
        X_test_data_point: The test data point to be predicted.

    Returns:
        Tuple: A tuple containing the predicted probabilities for two isotonic regressions.
    """
    # print(len(X_cal))
    # print(len(Y_cal))
    # print(X_cal.shape)
    # print(Y_cal.shape)

    # TODO: do we want this with a scalar or a vector?
    # X_cal.append(X_test_data_point)
    # Y_cal.append(True)
    # print(len(X_cal))
    # print(len(Y_cal))
    X_cal = np.append(X_cal, X_test_data_point)
    Y_cal = np.append(Y_cal, True)

    ir_0 = IsotonicRegression(out_of_bounds="clip")
    ir_1 = IsotonicRegression(out_of_bounds="clip")

    ir_0.fit(X_cal, Y_cal)

    # run the second isotonic regression with the last point as a false label
    Y_cal[-1] = False
    ir_1.fit(X_cal, Y_cal)

    p_0 = ir_0.predict([X_test_data_point])[0]
    p_1 = ir_1.predict([X_test_data_point])[0]

    return p_0, p_1


### FAISS related functions
def load_database(lookup_database):
    """
    lookup_database: NxM matrix of embeddings
    """
    # On the fly FAISS import to prevent installation issues
    import faiss

    # Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return index


def query(index, queries, k=10):
    # On the fly FAISS import to prevent installation issues
    import faiss

    # Search indexed database
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return (D, I)


### functions for hierarchical conformal
def build_scope_tree(list_sccs):
    """
    Build a scope tree from a list of SCCs
    """
    tree = {}
    for sccs in list_sccs:
        sccs = sccs.split(".")
        node = tree
        for s in sccs:
            if s not in node:
                node[s] = {}
            node = node[s]
    return tree


def get_thresh_hierarchical(data, lambdas, alpha):
    # get the worst case loss
    wc_loss = max([np.sum(x["loss"]) for x in data])
    wc_loss = wc_loss / 14777
    loss_thresh = alpha - (wc_loss - alpha) / len(
        data
    )  # normalize by size of calib set
    losses = []
    best_lam = None
    for lam in lambdas:
        per_lam_loss = get_hierarchical_loss(data, lam)
        if per_lam_loss > loss_thresh:
            break
        best_lam = lam
        losses.append(per_lam_loss)
    print("worst case loss: " + str(wc_loss))
    print("Loss threshold: " + str(loss_thresh))
    print("Best lambda: " + str(best_lam))
    print("Loss of best lambda: " + str(losses[-1]))

    return best_lam, losses


def get_hierarchical_loss(data_, lambda_):
    losses = []
    for query in data_:
        thresh_idx = query["Sum_Norm_S_i"] <= lambda_
        if np.sum(thresh_idx) == 0:
            loss = 0
        else:
            loss = np.sum(np.asarray(query["loss"])[thresh_idx]) / np.sum(
                thresh_idx
            )  # NOTE: have fixed denominator, but alpha has to change
        losses.append(loss)  # average over all queries
    return np.mean(losses)


def get_thresh_max_hierarchical(data, lambdas, alpha, sim="cosine"):
    # get the worst case loss
    wc_loss = 4  # in the max case, the max_loss is simply retrieving a protein with different class
    loss_thresh = alpha - (wc_loss - alpha) / len(
        data
    )  # normalize by size of calib set
    losses = []
    best_lam = None
    if sim == "cosine":
        ## reverse lambdas and return list
        lambdas = list(reversed(lambdas))
    for (
        lam
    ) in (
        lambdas
    ):  # start from the largest lambda since we are dealing with raw similarity scores
        per_lam_loss = get_hierarchical_max_loss(data, lam, sim=sim)
        if per_lam_loss > loss_thresh:
            break
        best_lam = lam
        losses.append(per_lam_loss)
    print("worst case loss: " + str(wc_loss))
    print("Loss threshold: " + str(loss_thresh))
    print("Best lambda: " + str(best_lam))
    print("Loss of best lambda: " + str(losses[-1]))

    return best_lam, losses


def get_hierarchical_max_loss(data_, lambda_, sim="cosine"):
    losses = []
    for query in data_:
        if sim == "cosine":
            thresh_idx = query["S_i"] >= lambda_
        else:
            thresh_idx = query["S_i"] <= lambda_
        if np.sum(thresh_idx) == 0:
            loss = 0
        else:
            loss = np.max(np.asarray(query["loss"])[thresh_idx])  # monotonic loss
            # loss = np.sum(np.asarray(query['loss'])[thresh_idx]) / np.sum(thresh_idx) # NOTE: have fixed denominator, but alpha has to change
        losses.append(loss)  # average over all queries
    return np.mean(losses)


def get_scope_dict(true_test_idcs, test_df, lookup_idcs, lookup_df, D, I):
    """
    true_test_idcs: indices of the test set within the scope dataframe

    test_df: dataframe containing the test set (indices are the same as true_test_idcs)

    lookup_idcs: indices of the lookup set within the larger scope dataframe

    lookup_df: dataframe containing the lookup set (indices are the same as lookup_idcs)

    D: distances matrix (400 x 14777 or test x lookup by default)

    I: indices matrix (400 x 14777 or test x lookup by default)

    NOTE: Indices computed by FAISS are not the same as the indices of the dataframe, so
    we use the lookup_idcs list to map FAISS indices in I to the indices of the dataframe
    """

    near_ids = []
    min_sim = np.min(D)
    max_sim = np.max(D)

    for i in range(len(true_test_idcs)):
        test_id = test_df.loc[true_test_idcs[i], "sid"]
        test_sccs = test_df.loc[true_test_idcs[i], "sccs"]
        query_ids = [lookup_df.loc[lookup_idcs[j], "sid"] for j in I[i]]
        exact_loss = [
            scope_hierarchical_loss(test_sccs, lookup_df.loc[lookup_idcs[j], "sccs"])
            for j in I[i]
        ]
        # grab the 2nd element in the tuple belonging to each element of exact_loss as mask_exact
        mask_exact = [x[1] for x in exact_loss]
        loss = [x[0] for x in exact_loss]

        # define mask_partial as 1 for any element of loss that is <=1 (tolerate retrieving homolog with diff family but same superfamily)
        mask_partial = [l <= 1 for l in loss]

        # create a row of size len(lookup_df) where each element is the sum of all entries in S_i until that index
        sum = np.cumsum(D[i])
        norm_sim = (D[i] - min_sim) / (
            max_sim - min_sim
        )  # convert similarities into a probability space (0, 1) based on (min_sim, max_sim)
        # mask_exact = [test_sccs == lookup_df.loc[lookup_idcs[j], 'sccs'] for j in I[i]]

        sum_norm_s_i = np.cumsum(norm_sim)
        near_ids.append(
            {
                "test_id": test_id,
                "query_ids": query_ids,
                #'meta_query': meta_query,
                "loss": loss,
                "exact": mask_exact,
                "partial": mask_partial,
                "S_i": D[i],
                "Sum_i": sum,
                "Norm_S_i": norm_sim,
                "Sum_Norm_S_i": sum_norm_s_i,
                "I_i": I[i],
            }
        )
    return near_ids


"""
def scope_hierarchical_loss(y_sccs, y_hat_sccs, slack = 0):

    #Find the common ancestor of two sets of SCCs (0 if family, 1 if superfamily, 2 if fold, 3 if class)

    # Find the common ancestor of the two sets of SCCs
    y_sccs, y_hat_sccs = y_sccs.split('.'), y_hat_sccs.split('.')
    assert len(y_sccs) == len(y_hat_sccs) == 4

    loss, count = None, 0
    while loss is None:
        if y_sccs[-1] != y_hat_sccs[-1]:
            count += 1
            y_sccs.pop()
            y_hat_sccs.pop()
        else:
            break
    loss = count - slack
    
    return loss == 0
"""


def scope_hierarchical_loss(y_sccs, y_hat_sccs):
    """
    Find the common ancestor of two sets of SCCs (0 if family, 1 if superfamily, 2 if fold, 3 if class)
    """
    y_sccs, y_hat_sccs = y_sccs.split("."), y_hat_sccs.split(".")
    first_non_matching_index = next(
        (i for i, (x, y) in enumerate(zip(y_sccs, y_hat_sccs)) if x != y), len(y_sccs)
    )

    loss = (
        len(y_sccs) - first_non_matching_index
    )  # ex if the first mismatch is at idx 2 (0-indexed), that means that the second last label (superfamily) is wrong, which is a loss of two (wrong family & superfamily)
    exact = True if len(y_sccs) == first_non_matching_index else False

    return loss, exact
