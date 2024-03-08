import numpy as np
import matplotlib.pyplot as plt
import pdb
import ipdb
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
from scipy.stats import binom, norm

def get_sims_labels(data, partial=False):
    sims = []
    labels = []
    for query in data:
        similarity = query['S_i']
        sims += similarity.tolist()
        if partial:
            #labels_to_append = np.logical_or.reduce(query['partial'], axis=1).tolist()
            # NOTE: no need to do the above for scope - i already handle the pre-processing
            labels_to_append = query['partial']
        else:
            labels_to_append = query['exact']
        labels += labels_to_append
    return sims, labels

def get_thresh(data, alpha):
    # conformal risk control
    all_sim_exact = []
    for query in data:
        idx = query['exact']
        similarity = query['S_i']
        sims_to_append = similarity[idx]
        all_sim_exact += list(sims_to_append)
        n = len(all_sim_exact)
        if n > 0:
            lhat = np.quantile(all_sim_exact, np.maximum(alpha-(1-alpha)/n, 0), interpolation='lower')
        else:
            lhat = 0
    return lhat

# Bentkus p value
def bentkus_p_value(r_hat,n,alpha):
    return binom.cdf(np.ceil(n*r_hat),n,alpha/np.e)

# def clt_p_value(r_hat,n,alpha):

def clt_p_value(r_hat, std_hat, n, alpha):
    z = (r_hat - alpha) / (std_hat / np.sqrt(n))
    p_value = norm.cdf(z)
    return p_value
    
def percentage_of_discoveries(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    return total_discoveries.mean() / len(labels) # or sims.shape[1]

def risk(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1-labels)*(sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries/total_discoveries).mean()

def calculate_false_negatives(sims, labels, lam):
    # FNR: Number of false non-matches / number of non-matches
    total_non_matches = labels.sum(axis=1)
    false_non_matches = (labels & (sims < lam)).sum(axis=1)
    total_non_matches = np.maximum(total_non_matches, 1)
    return (false_non_matches/total_non_matches).mean()

def risk_no_empties(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1-labels)*(sims >= lam)).sum(axis=1)
    idx = total_discoveries > 0
    total_discoveries = total_discoveries[idx]
    false_discoveries = false_discoveries[idx]
    return (false_discoveries/total_discoveries).mean()

def std_loss(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1-labels)*(sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries/total_discoveries).std()

def get_thresh_FDR(labels, sims, alpha, delta=0.5, N=5000):
    # FDR control with LTT
    # labels = np.stack([query['exact'] for query in data], axis=0)
    # sims = np.stack([query['S_i'] for query in data], axis=0)
    print(f"sims.max: {sims.max()}")
    n = len(labels)
    lambdas = np.linspace(sims.min(),sims.max(),N)
    risks = np.array( [risk(sims, labels, lam) for lam in lambdas] )
    stds = np.array( [std_loss(sims, labels, lam) for lam in lambdas] )
    #pvals = np.array( [bentkus_p_value(r,n,alpha) for r in risks] )
    pvals = np.array( [clt_p_value(r,s,n,alpha) for r, s in zip(risks, stds)] )
    below = pvals <= delta
    # Pick the smallest lambda such that all lambda above it have p-value below delta
    pvals_satisfy_condition = np.array([ np.all(below[i:])for i in range(N) ])
    lhat = lambdas[np.argmax(pvals_satisfy_condition)]
    print(f"lhat: {lhat}")
    print(f"risk: {risk(sims, labels, lhat)}")
    return lhat

def get_isotone_regression(data):
    sims, labels = get_sims_labels(data, partial=True)
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(sims, labels)
    return ir

def scope_hierarchical_loss(y_sccs, y_hat_sccs, slack = 0):
    """
    Find the common ancestor of two sets of SCCs (0 if family, 1 if superfamily, 2 if fold, 3 if class)
    """
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

    for i in range(len(true_test_idcs)):
        test_id = test_df.loc[true_test_idcs[i], 'sid']
        test_sccs = test_df.loc[true_test_idcs[i], 'sccs']
        query_ids = [lookup_df.loc[lookup_idcs[j], 'sid'] for j in I[i]]
        exact_loss = [scope_hierarchical_loss(test_sccs, lookup_df.loc[lookup_idcs[j], 'sccs']) for j in I[i]]
        # grab the 2nd element in the tuple belonging to each element of exact_loss as mask_exact
        mask_exact = [x[1] for x in exact_loss]
        loss = [x[0] for x in exact_loss]
        
        # define mask_partial as 1 for any element of loss that is <=1 (tolerate retrieving homolog with diff family but same superfamily)
        mask_partial = [l <= 1 for l in loss]

        #mask_exact = [test_sccs == lookup_df.loc[lookup_idcs[j], 'sccs'] for j in I[i]]
        near_ids.append({
            'test_id': test_id,
            'query_ids': query_ids,
            #'meta_query': meta_query,
            'loss' : loss,
            'exact': mask_exact,
            'partial': mask_partial,
            'S_i': D[i],
            'I_i': I[i]
        })
    return near_ids

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
        # if partial has multiple rows, we want to take the logical or of all of them. Otherwise just set it to the single row
        # check if there is one or more rows
        # query['partial'] = np.array(query['partial'])
        if len(np.array(query['partial']).shape) > 1:
            idx_partial = np.logical_or.reduce(query['partial'], axis=1)
        else:
            idx_partial = query['partial']
        
        sims = query['S_i']
        sims_exact = sims[idx]
        sims_partial = sims[idx_partial]
        total_missed += (sims_exact < lhat).sum()

        # TODO: are there any divisions by zero here?
        total_missed_partial += (sims_partial < lhat).sum()
        total_partial_identified += (sims_partial >= lhat).sum()
        total_partial += len(sims_partial)

        total_exact += len(sims_exact)
        total_inexact_identified += (sims[~np.array(idx)] >= lhat).sum()
        total_identified += (sims >= lhat).sum()
    return total_missed/total_exact, total_inexact_identified/total_identified, total_missed_partial/total_partial, total_partial_identified/total_identified


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

def build_scope_tree(list_sccs):
    """
    Build a scope tree from a list of SCCs
    """
    tree = {}
    for sccs in list_sccs:
        sccs = sccs.split('.')
        node = tree
        for s in sccs:
            if s not in node:
                node[s] = {}
            node = node[s]
    return tree