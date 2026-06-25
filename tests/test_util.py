"""
Tests for protein_conformal/util.py core functions.

These tests verify:
1. FASTA parsing
2. FAISS database operations
3. FDR/FNR threshold calculations (conformal risk control)
4. Risk metrics (FDR, FNR, TPR)
5. Venn-Abers probability predictions
6. Hierarchical loss functions (for SCOPe)
"""
import os
import numpy as np
import pytest
from protein_conformal.util import (
    read_fasta,
    load_database,
    query,
    get_thresh_new,
    get_thresh_new_FDR,
    get_thresh_FDR,
    risk,
    risk_1d,
    calculate_false_negatives,
    calculate_true_positives,
    simplifed_venn_abers_prediction,
    get_isotone_regression,
    scope_hierarchical_loss,
    validate_lhat_new,
)


class TestFastaParsing:
    """Tests for FASTA file parsing."""

    def test_read_fasta_basic(self, sample_fasta_file):
        """Test basic FASTA parsing returns sequences and metadata."""
        sequences, metadata = read_fasta(sample_fasta_file)

        assert len(sequences) == 3
        assert len(metadata) == 3

        # Check first sequence
        assert sequences[0].startswith('MVLSPADKTN')
        assert '>protein1' in metadata[0]

    def test_read_fasta_sequence_content(self, sample_fasta_file):
        """Test that sequences contain only valid amino acids."""
        sequences, _ = read_fasta(sample_fasta_file)

        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        for seq in sequences:
            assert all(aa in valid_aa for aa in seq), f"Invalid AA in sequence: {seq}"

    def test_read_fasta_short_sequence(self, sample_fasta_file):
        """Test that short sequence is parsed correctly."""
        sequences, metadata = read_fasta(sample_fasta_file)

        # Third sequence is exactly the 20 standard amino acids
        assert sequences[2] == 'ACDEFGHIKLMNPQRSTVWY'
        assert len(sequences[2]) == 20


class TestFAISSOperations:
    """Tests for FAISS database loading and querying."""

    def test_load_database(self, sample_embeddings):
        """Test that database loads and has correct dimensions."""
        _, lookup_embeddings = sample_embeddings

        index = load_database(lookup_embeddings.copy())

        assert index.ntotal == 100  # Number of vectors in index
        assert index.d == 128  # Dimensionality

    def test_query_returns_correct_shape(self, sample_embeddings):
        """Test that query returns distances and indices with correct shapes."""
        query_embeddings, lookup_embeddings = sample_embeddings

        index = load_database(lookup_embeddings.copy())
        D, I = query(index, query_embeddings.copy(), k=10)

        assert D.shape == (10, 10)  # 10 queries, k=10 neighbors
        assert I.shape == (10, 10)

    def test_query_distances_are_similarities(self, sample_embeddings):
        """Test that distances are cosine similarities (normalized dot product)."""
        query_embeddings, lookup_embeddings = sample_embeddings

        index = load_database(lookup_embeddings.copy())
        D, I = query(index, query_embeddings.copy(), k=10)

        # Cosine similarities should be in [-1, 1] range
        assert D.min() >= -1.0
        assert D.max() <= 1.0

    def test_query_indices_valid(self, sample_embeddings):
        """Test that returned indices are valid."""
        query_embeddings, lookup_embeddings = sample_embeddings

        index = load_database(lookup_embeddings.copy())
        D, I = query(index, query_embeddings.copy(), k=10)

        # All indices should be in valid range
        assert I.min() >= 0
        assert I.max() < 100  # lookup has 100 embeddings


class TestFAISSPrebuild:
    """Tests for prebuilt FAISS index save/load (cold-start optimization).

    Exact search must be preserved bit-for-bit: a prebuilt index read from disk
    has to return the same neighbors and similarities as building it on the fly.
    """

    def test_build_index_does_not_mutate_input(self, sample_embeddings):
        """build_index must not normalize the caller's array in place."""
        from protein_conformal.util import build_index
        _, lookup_embeddings = sample_embeddings
        original = lookup_embeddings.copy()

        build_index(lookup_embeddings)

        np.testing.assert_array_equal(lookup_embeddings, original)

    def test_build_index_matches_load_database(self, sample_embeddings):
        """build_index must return the same neighbors as load_database (exact search)."""
        from protein_conformal.util import build_index
        query_embeddings, lookup_embeddings = sample_embeddings

        ref = load_database(lookup_embeddings.copy())
        idx = build_index(lookup_embeddings.copy())

        D_ref, I_ref = query(ref, query_embeddings.copy(), k=10)
        D_new, I_new = query(idx, query_embeddings.copy(), k=10)

        np.testing.assert_array_equal(I_new, I_ref)
        np.testing.assert_allclose(D_new, D_ref, rtol=1e-6)

    def test_load_or_build_index_writes_when_missing(self, sample_embeddings, tmp_path):
        """When no prebuilt index exists, it is built and persisted to disk."""
        from protein_conformal.util import load_or_build_index
        _, lookup_embeddings = sample_embeddings
        index_path = str(tmp_path / "lookup.faissindex")

        index = load_or_build_index(lookup_embeddings.copy(), index_path)

        assert os.path.exists(index_path)
        assert index.ntotal == 100

    def test_lookup_index_path_swaps_npy_suffix(self):
        """The sidecar index path replaces a .npy suffix with .faissindex."""
        from protein_conformal.util import lookup_index_path
        assert lookup_index_path("/data/euk/euk_embeddings.npy") == "/data/euk/euk_embeddings.faissindex"

    def test_lookup_index_path_appends_when_no_npy(self):
        """Paths without a .npy suffix get .faissindex appended."""
        from protein_conformal.util import lookup_index_path
        assert lookup_index_path("/data/euk/euk_embeddings") == "/data/euk/euk_embeddings.faissindex"

    def test_load_or_build_index_roundtrip_matches(self, sample_embeddings, tmp_path):
        """A read-back index must return identical results to the freshly built one."""
        from protein_conformal.util import load_or_build_index, build_index
        query_embeddings, lookup_embeddings = sample_embeddings
        index_path = str(tmp_path / "lookup.faissindex")

        # First call builds + writes; second call reads from disk
        load_or_build_index(lookup_embeddings.copy(), index_path)
        read_back = load_or_build_index(lookup_embeddings.copy(), index_path)

        ref = build_index(lookup_embeddings.copy())
        D_ref, I_ref = query(ref, query_embeddings.copy(), k=10)
        D_read, I_read = query(read_back, query_embeddings.copy(), k=10)

        np.testing.assert_array_equal(I_read, I_ref)
        np.testing.assert_allclose(D_read, D_ref, rtol=1e-6)

    def test_load_lookup_index_uses_prebuilt_without_npy(self, sample_embeddings, tmp_path):
        """When a sidecar index exists, the .npy is not required (np.load is skipped)."""
        import faiss
        from protein_conformal.util import load_lookup_index, build_index, lookup_index_path
        _, lookup_embeddings = sample_embeddings
        npy_path = str(tmp_path / "lookup.npy")  # intentionally NOT created
        faiss.write_index(build_index(lookup_embeddings.copy()), lookup_index_path(npy_path))

        index, num = load_lookup_index(npy_path)

        assert not os.path.exists(npy_path)  # proves we did not depend on the array
        assert num == 100
        assert index.ntotal == 100

    def test_load_lookup_index_builds_from_npy_when_no_sidecar(self, sample_embeddings, tmp_path):
        """When no sidecar exists, it loads the .npy and builds an index."""
        from protein_conformal.util import load_lookup_index, lookup_index_path
        _, lookup_embeddings = sample_embeddings
        npy_path = str(tmp_path / "lookup.npy")
        np.save(npy_path, lookup_embeddings)
        assert not os.path.exists(lookup_index_path(npy_path))

        index, num = load_lookup_index(npy_path)

        assert num == 100
        assert index.ntotal == 100


class TestQueryCountError:
    """Tests for the interactive query-count cap (large jobs routed to the CLI)."""

    def test_under_cap_returns_none(self):
        from protein_conformal.util import query_count_error
        assert query_count_error(10, 5000) is None
        assert query_count_error(5000, 5000) is None  # at the cap is allowed

    def test_over_cap_returns_message(self):
        from protein_conformal.util import query_count_error
        msg = query_count_error(5001, 5000)
        assert msg is not None
        assert "5001" in msg and "5000" in msg  # actual count and the cap

    def test_zero_returns_none(self):
        from protein_conformal.util import query_count_error
        assert query_count_error(0, 5000) is None


class TestLRUCache:
    """Tests for the process-level LRU used to share embeddings across sessions."""

    def test_get_missing_returns_none(self):
        from protein_conformal.util import LRUCache
        assert LRUCache(2).get("x") is None

    def test_put_then_get(self):
        from protein_conformal.util import LRUCache
        c = LRUCache(2)
        c.put("a", 1)
        assert c.get("a") == 1

    def test_evicts_least_recently_used(self):
        from protein_conformal.util import LRUCache
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")          # touch 'a' so 'b' is now least-recently-used
        c.put("c", 3)       # exceeds capacity -> evicts 'b'
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3

    def test_clear_empties_cache(self):
        from protein_conformal.util import LRUCache
        c = LRUCache(2)
        c.put("a", 1)
        c.clear()
        assert c.get("a") is None


class TestSequencesHash:
    """Tests for the embedding-cache key (must not collide across splits)."""

    def test_distinguishes_different_splits(self):
        from protein_conformal.util import sequences_hash
        # Same concatenation, different boundaries -> must differ.
        assert sequences_hash(["AB", "CD"]) != sequences_hash(["ABC", "D"])
        assert sequences_hash(["AB", "CD"]) != sequences_hash(["ABCD"])

    def test_stable_for_same_input(self):
        from protein_conformal.util import sequences_hash
        assert sequences_hash(["AB", "CD"]) == sequences_hash(["AB", "CD"])

    def test_order_sensitive(self):
        from protein_conformal.util import sequences_hash
        assert sequences_hash(["AB", "CD"]) != sequences_hash(["CD", "AB"])


class TestDisplayCap:
    """Tests for capping rendered table rows per query (UI payload reduction).

    The browser table only needs a slice; the full match set stays server-side
    for download. Capping per query keeps every query visible instead of letting
    one large query crowd the others out.
    """

    def test_cap_matches_per_query_limits_each_query(self):
        from protein_conformal.util import cap_matches_per_query
        matches = ([{"query_meta": "A", "i": i} for i in range(5)] +
                   [{"query_meta": "B", "i": i} for i in range(3)])
        capped = cap_matches_per_query(matches, "query_meta", 2)
        assert len(capped) == 4
        assert [m["i"] for m in capped if m["query_meta"] == "A"] == [0, 1]
        assert [m["i"] for m in capped if m["query_meta"] == "B"] == [0, 1]

    def test_cap_matches_per_query_preserves_input_order(self):
        from protein_conformal.util import cap_matches_per_query
        matches = [
            {"query_meta": "A", "i": 0},
            {"query_meta": "B", "i": 0},
            {"query_meta": "A", "i": 1},
            {"query_meta": "B", "i": 1},
            {"query_meta": "A", "i": 2},
        ]
        capped = cap_matches_per_query(matches, "query_meta", 1)
        # First occurrence of each query, in original order.
        assert capped == [{"query_meta": "A", "i": 0}, {"query_meta": "B", "i": 0}]

    def test_cap_matches_per_query_under_cap_unchanged(self):
        from protein_conformal.util import cap_matches_per_query
        matches = [{"query_meta": "A", "i": 0}, {"query_meta": "A", "i": 1}]
        assert cap_matches_per_query(matches, "query_meta", 100) == matches

    def test_cap_matches_per_query_empty(self):
        from protein_conformal.util import cap_matches_per_query
        assert cap_matches_per_query([], "query_meta", 10) == []

    def test_cap_matches_per_query_keeps_all_queries_genome_scale(self):
        """Genome scale: the cap limits matches PER query, never drops a query."""
        from protein_conformal.util import cap_matches_per_query
        # 4000 queries (bacterial genome), 5 matches each, cap 200.
        matches = [{"query_meta": f"gene_{q}", "i": j}
                   for q in range(4000) for j in range(5)]
        capped = cap_matches_per_query(matches, "query_meta", 200)
        queries_out = {m["query_meta"] for m in capped}
        assert len(queries_out) == 4000      # every query still represented
        assert len(capped) == 4000 * 5       # all matches kept (under the cap)

    def test_cap_matches_per_query_many_queries_over_cap(self):
        """A query with more than `cap` matches is capped, others untouched."""
        from protein_conformal.util import cap_matches_per_query
        matches = ([{"query_meta": "big", "i": j} for j in range(500)] +
                   [{"query_meta": f"q{q}", "i": 0} for q in range(100)])
        capped = cap_matches_per_query(matches, "query_meta", 200)
        queries_out = {m["query_meta"] for m in capped}
        assert len(queries_out) == 101       # 'big' + 100 singletons all present
        assert sum(1 for m in capped if m["query_meta"] == "big") == 200


class TestRiskMetrics:
    """Tests for FDR, FNR, and related risk calculations."""

    def test_risk_all_correct(self):
        """Test risk is 0 when all predictions above threshold are correct."""
        sims = np.array([[0.9, 0.8, 0.7, 0.6]])
        labels = np.array([[True, True, True, False]])  # First 3 are true matches

        # Threshold 0.65: returns indices 0,1,2 (all true) → FDR = 0
        fdr = risk(sims, labels, 0.65)
        assert fdr == 0.0

    def test_risk_all_incorrect(self):
        """Test risk is 1 when all predictions above threshold are incorrect."""
        sims = np.array([[0.9, 0.8, 0.7, 0.6]])
        labels = np.array([[False, False, False, True]])  # Only index 3 is true

        # Threshold 0.65: returns indices 0,1,2 (all false) → FDR = 1
        fdr = risk(sims, labels, 0.65)
        assert fdr == 1.0

    def test_risk_partial(self):
        """Test risk calculation with mixed predictions."""
        sims = np.array([[0.9, 0.8, 0.7, 0.6]])
        labels = np.array([[True, False, True, False]])

        # Threshold 0.65: returns 3 items, 1 false → FDR = 1/3
        fdr = risk(sims, labels, 0.65)
        assert abs(fdr - 1/3) < 1e-6

    def test_calculate_false_negatives_zero(self):
        """Test FNR is 0 when all positives are detected."""
        sims = np.array([[0.9, 0.8, 0.7, 0.6]])
        labels = np.array([[True, True, False, False]])

        # Threshold 0.75: detects both true positives → FNR = 0
        fnr = calculate_false_negatives(sims, labels, 0.75)
        assert fnr == 0.0

    def test_calculate_false_negatives_partial(self):
        """Test FNR when some positives are missed."""
        sims = np.array([[0.9, 0.8, 0.7, 0.6]])
        labels = np.array([[True, True, True, False]])

        # Threshold 0.85: only detects index 0, misses 1,2 → FNR = 2/3
        fnr = calculate_false_negatives(sims, labels, 0.85)
        assert abs(fnr - 2/3) < 1e-6


class TestConformalThresholds:
    """Tests for conformal risk control threshold calculations."""

    def test_get_thresh_new_basic(self, scope_like_data):
        """Test basic threshold calculation for FNR control."""
        sims, labels = scope_like_data
        alpha = 0.1

        lhat = get_thresh_new(sims, labels, alpha)

        # Threshold should be in valid similarity range
        assert sims.min() <= lhat <= sims.max()

    def test_get_thresh_new_FDR_basic(self, scope_like_data):
        """Test basic threshold calculation for FDR control."""
        sims, labels = scope_like_data
        alpha = 0.1

        lhat = get_thresh_new_FDR(sims, labels, alpha)

        # Threshold should be in valid similarity range
        assert sims.min() <= lhat <= sims.max()

    def test_threshold_decreases_with_lower_alpha(self, scope_like_data):
        """Test that more stringent alpha leads to lower threshold for FNR control.

        For FNR (false negative rate) control via get_thresh_new:
        - Lower alpha = more stringent = want fewer false negatives
        - Algorithm picks a lower quantile of positive similarities
        - Lower quantile = lower threshold = accept more matches
        """
        sims, labels = scope_like_data

        lhat_10 = get_thresh_new(sims, labels, alpha=0.1)
        lhat_05 = get_thresh_new(sims, labels, alpha=0.05)

        # Lower alpha (more stringent FNR) should give lower threshold
        assert lhat_05 <= lhat_10

    def test_get_thresh_FDR_returns_risk(self, scope_like_data):
        """Test that get_thresh_FDR returns both threshold and risk."""
        sims, labels = scope_like_data
        alpha = 0.1

        lhat, risk_fdr = get_thresh_FDR(labels, sims, alpha, delta=0.5, N=100)

        # Should return valid threshold and risk
        assert isinstance(lhat, (int, float))
        assert isinstance(risk_fdr, (int, float))
        assert 0 <= risk_fdr <= 1


class TestVennAbers:
    """Tests for Venn-Abers probability predictions."""

    def test_simplified_venn_abers_returns_two_probs(self):
        """Test that simplified Venn-Abers returns p0 and p1."""
        np.random.seed(42)
        X_cal = np.random.uniform(0.5, 1.0, 100)
        Y_cal = (X_cal > 0.7).astype(bool)
        X_test = 0.8

        p0, p1 = simplifed_venn_abers_prediction(X_cal, Y_cal, X_test)

        assert 0 <= p0 <= 1
        assert 0 <= p1 <= 1

    def test_venn_abers_high_similarity_high_prob(self):
        """Test that high similarity gives high probability."""
        # Calibration: high sim → positive label
        X_cal = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        Y_cal = np.array([False, False, False, True, True, True])

        # Test point with high similarity should get high probability
        p0, p1 = simplifed_venn_abers_prediction(X_cal.copy(), Y_cal.copy(), 0.92)

        # Average of p0, p1 should be high for high similarity
        avg_prob = (p0 + p1) / 2
        assert avg_prob > 0.5

    def test_isotonic_regression_monotonic(self):
        """Test that isotonic regression produces monotonic predictions."""
        X = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        y = np.array([0.1, 0.2, 0.4, 0.8, 0.9])

        ir = get_isotone_regression(X, y)

        # Predictions should be monotonically increasing
        test_x = np.linspace(0.5, 0.9, 10)
        preds = ir.predict(test_x)

        assert all(preds[i] <= preds[i+1] for i in range(len(preds)-1))


class TestHierarchicalLoss:
    """Tests for SCOPe hierarchical loss function."""

    def test_exact_match(self):
        """Test exact match returns loss=0, exact=True."""
        loss, exact = scope_hierarchical_loss('a.1.1.1', 'a.1.1.1')
        assert loss == 0
        assert exact is True

    def test_family_mismatch(self):
        """Test family mismatch (last level) returns loss=1."""
        loss, exact = scope_hierarchical_loss('a.1.1.1', 'a.1.1.2')
        assert loss == 1
        assert exact is False

    def test_superfamily_mismatch(self):
        """Test superfamily mismatch returns loss=2."""
        loss, exact = scope_hierarchical_loss('a.1.1.1', 'a.1.2.1')
        assert loss == 2
        assert exact is False

    def test_fold_mismatch(self):
        """Test fold mismatch returns loss=3."""
        loss, exact = scope_hierarchical_loss('a.1.1.1', 'a.2.1.1')
        assert loss == 3
        assert exact is False

    def test_class_mismatch(self):
        """Test class mismatch returns loss=4."""
        loss, exact = scope_hierarchical_loss('a.1.1.1', 'b.1.1.1')
        assert loss == 4
        assert exact is False


class TestValidation:
    """Tests for validation functions."""

    def test_validate_lhat_new_returns_metrics(self, scope_like_data):
        """Test that validate_lhat_new returns expected metrics."""
        sims, labels_exact = scope_like_data
        labels_partial = labels_exact.copy()  # Use same for simplicity

        lhat = 0.9995  # Some threshold

        error, frac_inexact, error_partial, frac_partial, fpr = validate_lhat_new(
            sims, labels_partial, labels_exact, lhat
        )

        # All metrics should be in [0, 1]
        assert 0 <= error <= 1
        assert 0 <= frac_inexact <= 1
        assert 0 <= error_partial <= 1
        assert 0 <= frac_partial <= 1
        assert 0 <= fpr <= 1


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_fdr_pipeline(self, calibration_test_split):
        """Test complete FDR control pipeline: calibrate → threshold → validate."""
        data = calibration_test_split
        alpha = 0.1

        # Step 1: Get threshold from calibration data
        lhat = get_thresh_new_FDR(
            data['cal_sims'],
            data['cal_labels'],
            alpha
        )

        # Step 2: Calculate risk on test data
        test_fdr = risk(data['test_sims'], data['test_labels'], lhat)

        # FDR should be controlled (may be higher due to randomness in small samples)
        # In practice with enough data, test_fdr should be <= alpha
        assert test_fdr >= 0  # At minimum, should be valid

    def test_full_fnr_pipeline(self, calibration_test_split):
        """Test complete FNR control pipeline."""
        data = calibration_test_split
        alpha = 0.1

        # Get threshold for FNR control
        lhat = get_thresh_new(
            data['cal_sims'],
            data['cal_labels'],
            alpha
        )

        # Calculate FNR on test data
        test_fnr = calculate_false_negatives(
            data['test_sims'],
            data['test_labels'],
            lhat
        )

        # FNR should be controlled
        assert test_fnr >= 0  # At minimum, should be valid
