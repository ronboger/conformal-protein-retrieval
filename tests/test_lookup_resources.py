"""
Tests for get_lookup_resources() — the protein-search lookup loader.

Covers the cold-start fast path: when a prebuilt FAISS sidecar index exists, the
large .npy embedding file is never loaded.

Note: gradio is not installed in the test environment, so we mock it before import
(same pattern as test_clean.py).
"""
import os
import sys
import threading
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Mock gradio before importing gradio_interface
_gr_mock = MagicMock()
_gr_mock.Progress = MagicMock(return_value=MagicMock())
sys.modules.setdefault("gradio", _gr_mock)


@pytest.fixture
def lookup_dir(tmp_path):
    """Create temporary lookup embeddings (.npy) + metadata (.tsv)."""
    n, dim = 20, 32
    np.random.seed(7)
    embeddings = np.random.randn(n, dim).astype(np.float32)
    emb_path = str(tmp_path / "lookup_embeddings.npy")
    np.save(emb_path, embeddings)

    meta_path = str(tmp_path / "lookup_meta.tsv")
    with open(meta_path, "w") as f:
        f.write("Entry\tProtein names\tSequence\n")
        for i in range(n):
            f.write(f"P{i:05d}\tprotein {i}\tMVLSPADKTNVKAAW\n")

    return {"emb_path": emb_path, "meta_path": meta_path, "n": n, "dim": dim}


def _reset_cache(gi):
    gi.LOOKUP_RESOURCE_CACHE.clear()


class TestGetLookupResources:
    """Integration tests for the protein-search lookup loader."""

    def test_builds_from_npy_when_no_prebuilt_index(self, lookup_dir):
        """With no sidecar, it loads the .npy and assembles the resource dict."""
        import protein_conformal.backend.gradio_interface as gi
        _reset_cache(gi)

        with patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p):
            res = gi.get_lookup_resources(lookup_dir["emb_path"], lookup_dir["meta_path"])

        assert res["index"].ntotal == lookup_dir["n"]
        assert res["num_embeddings"] == lookup_dir["n"]
        assert len(res["lookup_seqs"]) == lookup_dir["n"]
        _reset_cache(gi)

    def test_uses_prebuilt_index_without_npy(self, lookup_dir):
        """With a prebuilt sidecar present, the .npy is not needed (np.load skipped)."""
        import faiss
        import protein_conformal.backend.gradio_interface as gi
        from protein_conformal.util import build_index, lookup_index_path
        _reset_cache(gi)

        # Prebuild sidecar, then delete the .npy to prove it is never loaded.
        emb = np.load(lookup_dir["emb_path"])
        faiss.write_index(build_index(emb), lookup_index_path(lookup_dir["emb_path"]))
        os.remove(lookup_dir["emb_path"])

        with patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p):
            res = gi.get_lookup_resources(lookup_dir["emb_path"], lookup_dir["meta_path"])

        assert not os.path.exists(lookup_dir["emb_path"])
        assert res["index"].ntotal == lookup_dir["n"]
        assert res["num_embeddings"] == lookup_dir["n"]
        _reset_cache(gi)


class TestSingleFlight:
    """Concurrent cache-misses on the same DB must build the index only once."""

    def test_concurrent_misses_build_once(self, lookup_dir):
        import protein_conformal.backend.gradio_interface as gi
        _reset_cache(gi)

        calls = {"n": 0}
        count_lock = threading.Lock()
        sentinel_index = object()

        def slow_load(embedding_path):
            with count_lock:
                calls["n"] += 1
            time.sleep(0.3)  # hold the build so the second thread reaches the lock
            return sentinel_index, lookup_dir["n"]

        results = {}
        start = threading.Barrier(2)

        def worker(tag):
            start.wait(timeout=5)
            results[tag] = gi.get_lookup_resources(lookup_dir["emb_path"], lookup_dir["meta_path"])

        with patch.object(gi, "ensure_local_data_file", side_effect=lambda p: p), \
             patch.object(gi, "load_lookup_index", side_effect=slow_load):
            ta = threading.Thread(target=worker, args=("A",))
            tb = threading.Thread(target=worker, args=("B",))
            ta.start(); tb.start()
            ta.join(timeout=10); tb.join(timeout=10)

        assert calls["n"] == 1  # singleflight: one build despite two concurrent misses
        assert results["A"]["index"] is results["B"]["index"]  # both got the same built resource
        _reset_cache(gi)
