"""
Pytest fixtures for conformal protein retrieval tests.
"""
import numpy as np
import pytest
import tempfile
import os


@pytest.fixture
def sample_fasta_file():
    """Create a temporary FASTA file for testing."""
    content = """>protein1 | test protein 1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>protein2 | test protein 2
MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK
>protein3 | short sequence
ACDEFGHIKLMNPQRSTVWY
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing FAISS operations."""
    np.random.seed(42)
    # 10 query embeddings, 100 lookup embeddings, 128-dimensional
    query_embeddings = np.random.randn(10, 128).astype(np.float32)
    lookup_embeddings = np.random.randn(100, 128).astype(np.float32)
    return query_embeddings, lookup_embeddings


@pytest.fixture
def scope_like_data():
    """
    Create synthetic data similar to SCOPe experiment structure.

    Based on notebook: 400 queries x 14777 lookup, but we use smaller
    sizes for fast testing: 40 queries x 100 lookup.
    """
    np.random.seed(42)
    n_queries = 40
    n_lookup = 100

    # Similarity scores in realistic range (0.999 to 1.0 for protein-vec)
    sims = np.random.uniform(0.9993, 0.99999, size=(n_queries, n_lookup)).astype(np.float32)

    # Make ~10% exact matches (higher similarity)
    labels = np.random.random((n_queries, n_lookup)) < 0.1

    # Exact matches should have higher similarity
    sims[labels] = np.random.uniform(0.9998, 0.99999, size=labels.sum()).astype(np.float32)

    return sims, labels


@pytest.fixture
def calibration_test_split(scope_like_data):
    """Split data into calibration and test sets (like notebooks do 300/100)."""
    sims, labels = scope_like_data
    n_calib = 30  # 75% for calibration

    indices = np.random.permutation(len(sims))
    cal_idx = indices[:n_calib]
    test_idx = indices[n_calib:]

    return {
        'cal_sims': sims[cal_idx],
        'cal_labels': labels[cal_idx],
        'test_sims': sims[test_idx],
        'test_labels': labels[test_idx],
    }
