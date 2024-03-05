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