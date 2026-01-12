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

    all_sim_exact = X.flatten()[~Y.flatten()]
    n = len(all_sim_exact)
    if n > 0:
        lhat = np.quantile(
            all_sim_exact,
            np.maximum(alpha - (1 - alpha) / n, 0),
            interpolation="lower",
        )
    else:
        lhat = 0

    return lhat


# validate lhat
def validate_lhat_new(X, Y_partial, Y_exact, lhat):
    X_flat = X.flatten()
    X_exact = X_flat[Y_exact.flatten()]
    X_partial = X_flat[Y_partial.flatten()]

    total_missed = (X_exact < lhat).sum()
    total_missed_partial = (X_partial < lhat).sum()
    total_partial_identified = (X_partial >= lhat).sum()
    total_exact_identified = (X_exact >= lhat).sum()
    total_partial = len(X_partial)
    total_exact = len(X_exact)
    total_identified = (X_flat >= lhat).sum()
    total_inexact_identified = ((X_flat >= lhat) & ~Y_exact.flatten()).sum()

    error = total_missed / total_exact if total_exact > 0 else 0
    fraction_inexact = total_inexact_identified / total_identified if total_identified > 0 else 0
    error_partial = total_missed_partial / total_partial if total_partial > 0 else 0
    fraction_partial = total_partial_identified / total_identified if total_identified > 0 else 0

    total_negative = len(X_flat) - total_exact
    fpr = total_inexact_identified / total_negative if total_negative > 0 else 0

    return error, fraction_inexact, error_partial, fraction_partial, fpr


def get_thresh_new(X, Y, alpha):
    all_sim_exact = X.flatten()[Y.flatten()]
    n = len(all_sim_exact)
    if n > 0:
        lhat = np.quantile(
            all_sim_exact,
            np.maximum(alpha - (1 - alpha) / n, 0),
            interpolation="lower",
        )
    else:
        lhat = 0

    return lhat


def get_thresh(data, alpha):
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
                interpolation="lower",
            )
        else:
            lhat = 0
    return lhat


def bentkus_p_value(r_hat, n, alpha):
    return binom.cdf(np.ceil(n * r_hat), n, alpha / np.e)


def clt_p_value(r_hat, std_hat, n, alpha):
    z = (r_hat - alpha) / (std_hat / np.sqrt(n))
    p_value = norm.cdf(z)
    return p_value


def percentage_of_discoveries(sims, labels, lam):
    total_discoveries = (sims >= lam).sum(axis=1)
    return total_discoveries.mean() / len(labels)


def risk_1d(sims, labels, lam):
    total_discoveries = (sims >= lam).sum()
    false_discoveries = ((1 - labels) * (sims >= lam)).sum()
    total_discoveries = np.maximum(total_discoveries, 1)
    fdr = false_discoveries / total_discoveries
    return fdr


def risk(sims, labels, lam):
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries / total_discoveries).mean()


def calculate_true_positives(sims, labels, lam):
    total_matches = labels.sum(axis=1)
    true_matches = (labels & (sims >= lam)).sum(axis=1)
    total_matches = np.maximum(total_matches, 1)
    return (true_matches / total_matches).mean()


def calculate_false_negatives(sims, labels, lam):
    true_positives = ((labels == 1) & (sims >= lam)).sum()
    false_negatives = ((labels == 1) & (sims < lam)).sum()
    total_positives = (labels == 1).sum()
    if total_positives == 0:
        print("No actual positives")
        return 0
    return false_negatives / total_positives


def risk_no_empties(sims, labels, lam):
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    idx = total_discoveries > 0
    total_discoveries = total_discoveries[idx]
    false_discoveries = false_discoveries[idx]
    return (false_discoveries / total_discoveries).mean()


def std_loss(sims, labels, lam):
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries / total_discoveries).std()


def get_thresh_FDR(labels, sims, alpha, delta=0.5, N=5000):
    n = len(labels)
    lambdas = np.linspace(sims.min(), sims.max(), N)
    risks = np.array([risk(sims, labels, lam) for lam in lambdas])
    stds = np.array([std_loss(sims, labels, lam) for lam in lambdas])
    eps = 1e-6
    stds = np.maximum(stds, eps)
    pvals = np.array([clt_p_value(r, s, n, alpha) for r, s in zip(risks, stds)])
    below = pvals <= delta
    pvals_satisfy_condition = np.array([np.all(below[i:]) for i in range(N)])
    lhat = lambdas[np.argmax(pvals_satisfy_condition)]
    risk_fdr = risk(sims, labels, lhat)
    return lhat, risk_fdr


def get_isotone_regression(X, y):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)
    return ir


def validate_lhat(data, lhat):
    total_missed = 0
    total_missed_partial = 0
    total_exact = 0
    total_inexact_identified = 0
    total_identified = 0
    total_partial = 0
    total_partial_identified = 0
    for query in data:
        idx = query['exact']
        if len(np.array(query['partial']).shape) > 1:
            idx_partial = np.logical_or.reduce(query['partial'], axis=1)
        else:
            idx_partial = query['partial']
        sims = query['S_i']
        sims_exact = sims[idx]
        sims_partial = sims[idx_partial]
        total_missed += (sims_exact < lhat).sum()
        total_missed_partial += (sims_partial < lhat).sum()
        total_partial_identified += (sims_partial >= lhat).sum()
        total_partial += len(sims_partial)
        total_exact += len(sims_exact)
        total_inexact_identified += (sims[~np.array(idx)] >= lhat).sum()
        total_identified += (sims >= lhat).sum()
    return total_missed / total_exact, total_inexact_identified / total_identified, total_missed_partial / total_partial, total_partial_identified / total_identified


def load_database(lookup_database):
    import faiss
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)
    return index


def get_thresh_hierarchical(data, lambdas, alpha):
    wc_loss = max([np.sum(x['loss']) for x in data])
    wc_loss = wc_loss / 14777
    loss_thresh = alpha - (wc_loss - alpha) / len(data)
    losses = []
    best_lam = None
    for lam in lambdas:
        per_lam_loss = get_hierarchical_loss(data, lam)
        if per_lam_loss > loss_thresh:
            break
        best_lam = lam
        losses.append(per_lam_loss)
    return best_lam, losses


def get_hierarchical_loss(data_, lambda_):
    losses = []
    for query in data_:
        thresh_idx = query['Sum_Norm_S_i'] <= lambda_
        if np.sum(thresh_idx) == 0:
            loss = 0
        else:
            loss = np.sum(np.asarray(query['loss'])[thresh_idx]) / np.sum(thresh_idx)
        losses.append(loss)
    return np.mean(losses)


def get_thresh_max_hierarchical(data, lambdas, alpha, sim="cosine"):
    wc_loss = 4
    loss_thresh = alpha - (wc_loss - alpha) / len(data)
    losses = []
    best_lam = None
    if sim == "cosine":
        lambdas = list(reversed(lambdas))
    for lam in lambdas:
        per_lam_loss = get_hierarchical_max_loss(data, lam, sim=sim)
        if per_lam_loss > loss_thresh:
            break
        best_lam = lam
        losses.append(per_lam_loss)
    return best_lam, losses


def get_hierarchical_max_loss(data_, lambda_, sim="cosine"):
    losses = []
    for query in data_:
        if sim == "cosine":
            thresh_idx = query['S_i'] >= lambda_
        else:
            thresh_idx = query['S_i'] <= lambda_
        if np.sum(thresh_idx) == 0:
            loss = 0
        else:
            loss = np.max(np.asarray(query['loss'])[thresh_idx])
        losses.append(loss)
    return np.mean(losses)


def query(index, queries, k=10):
    import faiss
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)
    return (D, I)


def build_scope_tree(list_sccs):
    tree = {}
    for sccs in list_sccs:
        sccs = sccs.split('.')
        node = tree
        for s in sccs:
            if s not in node:
                node[s] = {}
            node = node[s]
    return tree

