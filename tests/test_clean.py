"""
Tests for CLEAN enzyme classification backend functions.

These tests verify:
1. get_clean_resources() — loads centroid embeddings/metadata, builds FAISS L2 index, caches
2. process_clean_input() — validates input, embeds sequences, searches index, returns results

Note: gradio is not installed in the test environment, so we mock it before import.
"""
import json
import sys
import numpy as np
import pandas as pd
import pytest
import os
from unittest.mock import patch, MagicMock

# Mock gradio before importing gradio_interface
_gr_mock = MagicMock()
_gr_mock.Progress = MagicMock(return_value=MagicMock())
sys.modules.setdefault("gradio", _gr_mock)


# ---------------------------------------------------------------------------
# Fixtures
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

    return {
        "emb_path": emb_path,
        "meta_path": meta_path,
        "n_centroids": n_centroids,
        "dim": dim,
        "embeddings": embeddings,
    }


@pytest.fixture
def clean_thresholds_file(tmp_path):
    """Create a temporary CLEAN thresholds CSV."""
    thresh_path = str(tmp_path / "clean_thresholds.csv")
    with open(thresh_path, "w") as f:
        f.write("alpha,threshold_mean,threshold_std,test_loss_mean,n_valid_trials\n")
        f.write("0.5,6.0,0.1,0.5,20\n")
        f.write("1.0,7.0,0.05,0.9,20\n")
        f.write("1.5,8.0,0.08,1.5,20\n")
        # Use large thresholds so test queries always produce results
    return thresh_path


SIMPLE_FASTA = """>seq1 test enzyme
MVLSPADKTNVKAAWGKVGA
>seq2 another enzyme
ACDEFGHIKLMNPQRSTVWY
"""


# ---------------------------------------------------------------------------
# TestGetCleanResources
# ---------------------------------------------------------------------------

class TestGetCleanResources:
    """Tests for get_clean_resources() loading and caching."""

    def _reset_cache(self):
        import protein_conformal.backend.gradio_interface as gi
        gi.CLEAN_RESOURCE_CACHE.clear()

    def test_loads_embeddings_and_builds_index(self, clean_centroid_dir):
        """get_clean_resources returns embeddings, ec_numbers, and FAISS index."""
        import protein_conformal.backend.gradio_interface as gi

        self._reset_cache()

        with patch.object(gi, "DEFAULT_CLEAN_CENTROID_EMBEDDING", clean_centroid_dir["emb_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_CENTROID_METADATA", clean_centroid_dir["meta_path"]), \
             patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p):

            result = gi.get_clean_resources()

        assert result["centroids"].shape == (clean_centroid_dir["n_centroids"], clean_centroid_dir["dim"])
        assert result["centroids"].dtype == np.float32
        assert len(result["ec_numbers"]) == clean_centroid_dir["n_centroids"]
        assert result["ec_numbers"][0] == "1.2.3.0"
        assert result["index"].ntotal == clean_centroid_dir["n_centroids"]
        assert result["num_centroids"] == clean_centroid_dir["n_centroids"]

        self._reset_cache()

    def test_caches_on_second_call(self, clean_centroid_dir):
        """Second call returns same cached object without reloading."""
        import protein_conformal.backend.gradio_interface as gi

        self._reset_cache()

        with patch.object(gi, "DEFAULT_CLEAN_CENTROID_EMBEDDING", clean_centroid_dir["emb_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_CENTROID_METADATA", clean_centroid_dir["meta_path"]), \
             patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p):

            result1 = gi.get_clean_resources()
            result2 = gi.get_clean_resources()

        assert result2 is gi.CLEAN_RESOURCE_CACHE
        np.testing.assert_array_equal(result1["centroids"], result2["centroids"])

        self._reset_cache()


# ---------------------------------------------------------------------------
# TestProcessCleanInput
# ---------------------------------------------------------------------------

class TestProcessCleanInput:
    """Tests for process_clean_input() end-to-end pipeline."""

    def _reset_cache(self):
        import protein_conformal.backend.gradio_interface as gi
        gi.CLEAN_RESOURCE_CACHE.clear()

    def test_rejects_empty_input(self):
        """Empty FASTA input returns error JSON and empty DataFrame."""
        import protein_conformal.backend.gradio_interface as gi

        mock_progress = MagicMock()
        summary_str, df = gi.process_clean_input("", None, 1.0, progress=mock_progress)
        parsed = json.loads(summary_str)

        assert "error" in parsed
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_rejects_whitespace_only_input(self):
        """Whitespace-only FASTA text returns error."""
        import protein_conformal.backend.gradio_interface as gi

        mock_progress = MagicMock()
        summary_str, df = gi.process_clean_input("   \n  \t  ", None, 1.0, progress=mock_progress)
        parsed = json.loads(summary_str)

        assert "error" in parsed
        assert df.empty

    def test_output_dataframe_has_expected_columns(self, clean_centroid_dir, clean_thresholds_file):
        """Results DataFrame has Query, Predicted EC, and L2 Distance columns."""
        import protein_conformal.backend.gradio_interface as gi

        self._reset_cache()

        dim = clean_centroid_dir["dim"]
        np.random.seed(42)
        fake_embeddings = np.random.randn(2, dim).astype(np.float32)

        mock_progress = MagicMock()

        def mock_run_embed_clean(sequences, progress=None):
            return fake_embeddings

        with patch.object(gi, "DEFAULT_CLEAN_CENTROID_EMBEDDING", clean_centroid_dir["emb_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_CENTROID_METADATA", clean_centroid_dir["meta_path"]), \
             patch.object(gi, "DEFAULT_CLEAN_THRESHOLDS", clean_thresholds_file), \
             patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p), \
             patch.object(gi, "ensure_local_results_file", side_effect=lambda p: p), \
             patch.object(gi, "run_embed_clean", side_effect=mock_run_embed_clean):

            # Clear LRU cache so our temp thresholds file is used
            gi._cached_results_dataframe.cache_clear()

            summary_str, df = gi.process_clean_input(
                SIMPLE_FASTA, None, 1.0, progress=mock_progress
            )

        parsed = json.loads(summary_str)
        assert parsed.get("status") == "success"
        assert parsed.get("mode") == "Enzyme Classification (CLEAN)"

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        expected_cols = {"Query", "Predicted EC", "L2 Distance"}
        assert expected_cols.issubset(set(df.columns)), f"Missing columns. Got: {list(df.columns)}"

        self._reset_cache()
        gi._cached_results_dataframe.cache_clear()
