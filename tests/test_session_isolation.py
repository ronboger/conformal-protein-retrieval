"""
Tests for per-user session isolation in the Gradio backend.

These verify that the search entry points thread a per-user ``session`` dict
through instead of writing a shared module-global ``CURRENT_SESSION``. With the
old global, two concurrent users clobbered each other's results (one user could
receive another's matches). The contract here:

  * ``process_input``        -> (summary_json, df, session)
  * ``process_clean_input``  -> (summary_json, df, session)

and the returned ``session`` carries only that call's own results, even when two
calls run concurrently.

Note: gradio is not installed in the test environment, so we mock it before import.
"""
import json
import sys
import threading

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Mock gradio before importing gradio_interface
_gr_mock = MagicMock()
_gr_mock.Progress = MagicMock(return_value=MagicMock())
sys.modules.setdefault("gradio", _gr_mock)


@pytest.fixture(autouse=True)
def _reset_embedding_cache():
    """Clear the process-level embedding cache around each test.

    Several tests reuse the same query sequence with a barrier inside the embed
    mock; a leftover cache entry would skip embed and deadlock the barrier.
    """
    import protein_conformal.backend.gradio_interface as gi
    gi.EMBEDDING_CACHE.clear()
    yield
    gi.EMBEDDING_CACHE.clear()


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_clean.py — kept local to avoid touching conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_centroid_dir(tmp_path):
    """Create temporary CLEAN centroid data files (embeddings + metadata)."""
    n_centroids = 10
    dim = 128

    np.random.seed(99)
    embeddings = np.random.randn(n_centroids, dim).astype(np.float32)
    emb_path = str(tmp_path / "ec_centroid_embeddings.npy")
    np.save(emb_path, embeddings)

    meta_path = str(tmp_path / "ec_centroid_metadata.tsv")
    with open(meta_path, "w") as f:
        f.write("EC_number\tn_proteins\n")
        for i in range(n_centroids):
            f.write(f"1.2.3.{i}\t{(i + 1) * 10}\n")

    return {"emb_path": emb_path, "meta_path": meta_path, "dim": dim, "embeddings": embeddings}


@pytest.fixture
def clean_thresholds_file(tmp_path):
    """Create a temporary CLEAN thresholds CSV (large thresholds => results pass)."""
    thresh_path = str(tmp_path / "clean_thresholds.csv")
    with open(thresh_path, "w") as f:
        f.write("alpha,threshold_mean,threshold_std,test_loss_mean,n_valid_trials\n")
        f.write("0.5,6.0,0.1,0.5,20\n")
        f.write("1.0,7.0,0.05,0.9,20\n")
        f.write("1.5,8.0,0.08,1.5,20\n")
    return thresh_path


def _reset_clean_cache(gi):
    gi.CLEAN_RESOURCE_CACHE.clear()
    gi._cached_results_dataframe.cache_clear()


# ---------------------------------------------------------------------------
# Regression guard: the module-global must be gone
# ---------------------------------------------------------------------------

def test_no_module_global_session():
    """The shared-mutable CURRENT_SESSION global must be removed entirely."""
    import protein_conformal.backend.gradio_interface as gi
    assert not hasattr(gi, "CURRENT_SESSION"), (
        "CURRENT_SESSION global still present — per-user session must flow through "
        "gr.State()/return values, not a module global."
    )


# ---------------------------------------------------------------------------
# CLEAN path
# ---------------------------------------------------------------------------

class TestCleanSessionThreading:
    def _run(self, gi, fasta, alpha, clean_centroid_dir, clean_thresholds_file,
             session=None, embed=None):
        # Default: return centroid[0] for every query so each lands on EC 1.2.3.0
        # at L2 distance ~0 (passes the threshold and yields deterministic matches).
        embed = embed or (lambda seqs, progress=None:
                          np.repeat(clean_centroid_dir["embeddings"][:1], len(seqs), axis=0).astype(np.float32))
        with patch.object(gi, "DEFAULT_CLEAN_CENTROID_EMBEDDING", clean_centroid_dir["emb_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_CENTROID_METADATA", clean_centroid_dir["meta_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_THRESHOLDS", clean_thresholds_file), \
             patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p), \
             patch.object(gi, "ensure_local_results_file", side_effect=lambda p: p), \
             patch.object(gi, "run_embed_clean", side_effect=embed):
            return gi.process_clean_input(fasta, None, alpha, session=session,
                                          progress=MagicMock())

    def test_returns_session_with_own_results(self, clean_centroid_dir, clean_thresholds_file):
        """process_clean_input returns a 3-tuple whose session holds this call's matches."""
        import protein_conformal.backend.gradio_interface as gi
        _reset_clean_cache(gi)

        fasta = ">queryA_MARK enzyme\nMVLSPADKTNVKAAWGKVGA\n"
        summary_str, df, session = self._run(
            gi, fasta, 1.0, clean_centroid_dir, clean_thresholds_file, session={})

        assert json.loads(summary_str).get("status") == "success"
        assert isinstance(session, dict)
        matches = session["results"]["matches"]
        assert matches, "expected non-empty matches"
        assert all("queryA_MARK" in m["query_meta"] for m in matches)
        assert session["parameters"]["alpha"] == 1.0
        _reset_clean_cache(gi)

    def test_sessions_isolated_under_concurrency(self, clean_centroid_dir, clean_thresholds_file):
        """Two concurrent CLEAN calls must not see each other's results."""
        import protein_conformal.backend.gradio_interface as gi
        _reset_clean_cache(gi)

        barrier = threading.Barrier(2)

        def embed(seqs, progress=None):
            barrier.wait(timeout=15)  # force the two calls to interleave
            return np.repeat(clean_centroid_dir["embeddings"][:1], len(seqs), axis=0).astype(np.float32)

        out = {}

        def worker(tag, alpha):
            fasta = f">query{tag}_MARK enzyme\nMVLSPADKTNVKAAWGKVGA\n"
            out[tag] = self._run(gi, fasta, alpha, clean_centroid_dir,
                                  clean_thresholds_file, session={}, embed=embed)

        with patch.object(gi, "DEFAULT_CLEAN_CENTROID_EMBEDDING", clean_centroid_dir["emb_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_CENTROID_METADATA", clean_centroid_dir["meta_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_THRESHOLDS", clean_thresholds_file), \
             patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p), \
             patch.object(gi, "ensure_local_results_file", side_effect=lambda p: p):
            ta = threading.Thread(target=worker, args=("A", 1.0))
            tb = threading.Thread(target=worker, args=("B", 1.5))
            ta.start(); tb.start()
            ta.join(timeout=30); tb.join(timeout=30)

        _, _, session_a = out["A"]
        _, _, session_b = out["B"]
        matches_a = session_a["results"]["matches"]
        matches_b = session_b["results"]["matches"]
        assert session_a is not session_b
        assert matches_a and matches_b, "expected non-empty matches in both sessions"
        assert all("queryA_MARK" in m["query_meta"] for m in matches_a)
        assert all("queryB_MARK" in m["query_meta"] for m in matches_b)
        assert session_a["parameters"]["alpha"] == 1.0
        assert session_b["parameters"]["alpha"] == 1.5
        _reset_clean_cache(gi)


# ---------------------------------------------------------------------------
# Protein-Vec path
# ---------------------------------------------------------------------------

def _mock_load_results_dataframe(path, required_columns=None):
    """Stand in for FDR/FNR threshold and calibration CSV loads."""
    cols = set(required_columns or [])
    if {"alpha", "lambda_threshold"} <= cols:
        return pd.DataFrame({
            "alpha": [0.1], "lambda_threshold": [0.0],
            "exact_fdr": [0.05], "partial_fdr": [0.05],
            "exact_fnr": [0.05], "partial_fnr": [0.05],
        })
    if "similarity" in cols:  # calibration table
        return pd.DataFrame({
            "similarity": [0.0, 1.0],
            "prob_exact_p0": [0.0, 1.0],
            "prob_exact_p1": [0.0, 1.0],
        })
    return pd.DataFrame()


def _mock_run_search(embeddings, query_seqs, query_meta, lookup_db=None,
                     metadata_db=None, threshold=0.0, k=1000, progress=None):
    """Echo each query's metadata into a result row (D_score above threshold)."""
    rows = []
    for i, qm in enumerate(query_meta):
        rows.append({
            "query_seq": query_seqs[i], "query_meta": qm,
            "lookup_seq": "AAAA", "D_score": 0.9,
            "lookup_entry": "E0", "lookup_pfam": "PF0",
            "lookup_protein_names": "name",
        })
    return pd.DataFrame(rows)


class TestProteinSearchSessionThreading:
    def _call(self, gi, fasta, session):
        return gi.process_input(
            "", fasta, None, "fasta_format", "FDR", 0.1, 10,
            True, None, gi.DEFAULT_LOOKUP_EMBEDDING, gi.DEFAULT_LOOKUP_METADATA,
            None, None, "Exact", 0.5, session=session, progress=MagicMock(),
        )

    def test_returns_session_with_own_results(self):
        import protein_conformal.backend.gradio_interface as gi

        def embed(seqs, progress=None):
            return np.ones((len(seqs), 512), dtype=np.float32)

        with patch.object(gi, "run_embed_protein_vec", side_effect=embed), \
             patch.object(gi, "run_search", side_effect=_mock_run_search), \
             patch.object(gi, "load_results_dataframe", side_effect=_mock_load_results_dataframe):
            fasta = ">queryA_MARK protein\nMVLSPADKTNVKAAWGKVGA\n"
            summary_str, df, session = self._call(gi, fasta, {})

        assert json.loads(summary_str).get("status") == "success"
        matches = session["results"]["matches"]
        assert matches
        assert all("queryA_MARK" in m["query_meta"] for m in matches)

    def test_summary_reports_search_time(self):
        """The result summary includes how long the search took."""
        import protein_conformal.backend.gradio_interface as gi

        def embed(seqs, progress=None):
            return np.ones((len(seqs), 512), dtype=np.float32)

        with patch.object(gi, "run_embed_protein_vec", side_effect=embed), \
             patch.object(gi, "run_search", side_effect=_mock_run_search), \
             patch.object(gi, "load_results_dataframe", side_effect=_mock_load_results_dataframe):
            fasta = ">queryA_MARK protein\nMVLSPADKTNVKAAWGKVGA\n"
            summary_str, _, _ = self._call(gi, fasta, {})

        summary = json.loads(summary_str)
        assert "search_time_seconds" in summary
        assert isinstance(summary["search_time_seconds"], (int, float))
        assert summary["search_time_seconds"] >= 0

    def test_embedding_cache_skips_repeat_embed(self):
        """A second fresh-session search of the same sequence reuses the embedding."""
        import protein_conformal.backend.gradio_interface as gi

        calls = {"n": 0}

        def embed(seqs, progress=None):
            calls["n"] += 1
            return np.ones((len(seqs), 512), dtype=np.float32)

        with patch.object(gi, "run_embed_protein_vec", side_effect=embed), \
             patch.object(gi, "run_search", side_effect=_mock_run_search), \
             patch.object(gi, "load_results_dataframe", side_effect=_mock_load_results_dataframe):
            fasta = ">queryA_MARK protein\nMVLSPADKTNVKAAWGKVGA\n"
            self._call(gi, fasta, {})   # fresh session -> embeds + caches
            self._call(gi, fasta, {})   # fresh session, same seq -> cache hit

        assert calls["n"] == 1

    def test_sessions_isolated_under_concurrency(self):
        import protein_conformal.backend.gradio_interface as gi

        barrier = threading.Barrier(2)

        def embed(seqs, progress=None):
            barrier.wait(timeout=15)
            return np.ones((len(seqs), 512), dtype=np.float32)

        out = {}

        def worker(tag):
            fasta = f">query{tag}_MARK protein\nMVLSPADKTNVKAAWGKVGA\n"
            out[tag] = self._call(gi, fasta, {})

        with patch.object(gi, "run_embed_protein_vec", side_effect=embed), \
             patch.object(gi, "run_search", side_effect=_mock_run_search), \
             patch.object(gi, "load_results_dataframe", side_effect=_mock_load_results_dataframe):
            ta = threading.Thread(target=worker, args=("A",))
            tb = threading.Thread(target=worker, args=("B",))
            ta.start(); tb.start()
            ta.join(timeout=30); tb.join(timeout=30)

        _, _, session_a = out["A"]
        _, _, session_b = out["B"]
        matches_a = session_a["results"]["matches"]
        matches_b = session_b["results"]["matches"]
        assert session_a is not session_b
        assert matches_a and matches_b, "expected non-empty matches in both sessions"
        assert all("queryA_MARK" in m["query_meta"] for m in matches_a)
        assert all("queryB_MARK" in m["query_meta"] for m in matches_b)
