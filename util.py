from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy.stats import binom, norm


def get_sims_labels(data, partial=False):
    sims = []
    labels = []
    for query in data:
        similarity = query["S_i"]
        sims += similarity.tolist()
        if partial:
            labels_to_append = np.logical_or.reduce(query["partial"], axis=1).tolist()
        else:
            labels_to_append = query["exact"]
        labels += labels_to_append
    return sims, labels


def get_thresh(data, alpha):
    # conformal risk control
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


# Bentkus p value
def bentkus_p_value(r_hat, n, alpha):
    return binom.cdf(np.ceil(n * r_hat), n, alpha / np.e)


# def clt_p_value(r_hat,n,alpha):


def clt_p_value(r_hat, std_hat, n, alpha):
    z = (r_hat - alpha) / (std_hat / np.sqrt(n))
    p_value = norm.cdf(z)
    return p_value


def percentage_of_discoveries(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    return total_discoveries.mean() / len(labels)  # or sims.shape[1]


def risk(sims, labels, lam):
    # FDR: Number of false matches / number of matches
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return (false_discoveries / total_discoveries).mean()


def calculate_false_negatives(sims, labels, lam):
    # FNR: Number of false non-matches / number of non-matches
    total_non_matches = labels.sum(axis=1)
    false_non_matches = (labels & (sims < lam)).sum(axis=1)
    total_non_matches = np.maximum(total_non_matches, 1)
    return (false_non_matches / total_non_matches).mean()


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


def get_thresh_FDR(labels, sims, alpha, delta=0.5, N=5000):
    # FDR control with LTT
    # labels = np.stack([query['exact'] for query in data], axis=0)
    # sims = np.stack([query['S_i'] for query in data], axis=0)
    print(f"sims.max: {sims.max()}")
    n = len(labels)
    lambdas = np.linspace(sims.min(), sims.max(), N)
    risks = np.array([risk(sims, labels, lam) for lam in lambdas])
    stds = np.array([std_loss(sims, labels, lam) for lam in lambdas])
    # pvals = np.array( [bentkus_p_value(r,n,alpha) for r in risks] )
    pvals = np.array([clt_p_value(r, s, n, alpha) for r, s in zip(risks, stds)])
    below = pvals <= delta
    # Pick the smallest lambda such that all lambda above it have p-value below delta
    pvals_satisfy_condition = np.array([np.all(below[i:]) for i in range(N)])
    lhat = lambdas[np.argmax(pvals_satisfy_condition)]
    print(f"lhat: {lhat}")
    print(f"risk: {risk(sims, labels, lhat)}")
    return lhat


def get_isotone_regression(data):
    sims, labels = get_sims_labels(data, partial=True)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(sims, labels)
    return ir


def simplifed_venn_abers_prediction(calib_data, test_data_point):
    sims, labels = get_sims_labels(calib_data, partial=False)
    print(sims)
    print(labels)
    print(len(sims))
    print(len(labels))

    sims.append(test_data_point)
    labels.append(True)
    print(len(sims))
    print(len(labels))

    ir_0 = IsotonicRegression(out_of_bounds="clip")
    ir_1 = IsotonicRegression(out_of_bounds="clip")

    ir_0.fit(sims, labels)

    labels[-1] = False
    ir_1.fit(sims, labels)

    p_0 = ir_0.predict([test_data_point])[0]
    p_1 = ir_1.predict([test_data_point])[0]

    return p_0, p_1


def validate_lhat(data, lhat):
    total_missed = 0
    total_missed_partial = 0
    total_exact = 0
    total_inexact_identified = 0
    total_identified = 0
    total_partial = 0
    total_partial_identified = 0
    for query in data:
        idx = query["exact"]
        # if partial has multiple rows, we want to take the logical or of all of them. Otherwise just set it to the single row
        # check if there is one or more rows
        # query['partial'] = np.array(query['partial'])
        if len(np.array(query["partial"]).shape) > 1:
            idx_partial = np.logical_or.reduce(query["partial"], axis=1)
        else:
            idx_partial = query["partial"]

        sims = query["S_i"]
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
    return (
        total_missed / total_exact,
        total_inexact_identified / total_identified,
        total_missed_partial / total_partial,
        total_partial_identified / total_identified,
    )
