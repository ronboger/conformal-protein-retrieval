"""Smoke tests for the Gradio web UI.

These tests intentionally do not launch the server or run a real protein search.
They cover the lightweight UI helpers and verify the Blocks app can be built.
"""

import json

import pytest


def test_summary_html_formats_success_and_error():
    pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import _format_summary_html

    success = {
        "status": "success",
        "matches_found": 42,
        "risk_control": {"type": "FDR", "alpha_used": 0.1},
        "search_config": {"database": "Swiss-Prot (540K)", "match_type": "exact", "max_k": 1000},
    }
    html = _format_summary_html(json.dumps(success), elapsed_s=3.2)
    assert "42 matches" in html
    assert "FDR α=0.1" in html
    assert "Swiss-Prot" in html
    assert "3.2s" in html
    assert "style=" in html

    err = _format_summary_html(json.dumps({"error": "boom"}))
    assert "Search failed" in err
    assert "boom" in err


def test_export_html_escapes_plain_text_status():
    pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import _format_export_html

    html = _format_export_html("✅ Exported 5 records to <bad>.csv")
    assert "Exported 5 records" in html
    assert "&lt;bad&gt;.csv" in html
    assert "#2d2440" in html


def test_setup_status_html_renders_readiness_rows():
    pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import _setup_status_html

    html = _setup_status_html()
    assert "Local readiness" in html
    assert "Swiss-Prot embeddings" in html
    assert "Prebuilt FAISS index" in html
    assert "Protein-Vec model files" in html
    assert "Python dependencies" in html


def test_create_interface_smoke_builds_blocks_with_theme_and_css_attrs():
    gr = pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import create_interface

    demo = create_interface()
    assert hasattr(demo, "blocks")
    assert hasattr(demo, "cpr_theme")
    assert hasattr(demo, "cpr_css")
    assert "#fasta-input" in demo.cpr_css

    # Public web UI should not expose local CPU/MPS/CUDA device selection;
    # Modal deployments use the GPU monkey-patched embedder automatically.
    labels = [getattr(block, "label", None) for block in demo.blocks.values()]
    assert "Embedding Device" not in labels


def test_create_interface_accepts_injected_embedders():
    pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import create_interface

    def fake_embed(seqs, progress=None, fp16_head=False):
        raise AssertionError("not called during UI construction")

    def fake_clean_embed(seqs, progress=None):
        raise AssertionError("not called during UI construction")

    demo = create_interface(embed_fn=fake_embed, clean_embed_fn=fake_clean_embed)
    assert hasattr(demo, "cpr_theme")


def test_results_display_df_is_database_aware():
    pytest.importorskip("gradio")
    import pandas as pd
    from protein_conformal.backend.gradio_interface import _format_results_display_df

    raw = pd.DataFrame([
        {
            "query_meta": "q1",
            "lookup_entry": "EUK_001",
            "lookup_protein_names": "hypothetical viral protein",
            "lookup_organism": "Nomburg virome sample",
            "lookup_pfam": "",
            "lookup_meta": "",
            "lookup_seq": "ACDE",
            "prob_exact": "80.0% ± 1.0%",
            "prob_partial": "90.0% ± 2.0%",
        }
    ])

    euk = _format_results_display_df(raw, "Euk (74K)")
    assert list(euk.columns) == ["Query", "Match ID", "Description", "Organism / Source", "Exact Prob", "Partial Prob"]
    assert "Pfam" not in euk.columns

    afdb = _format_results_display_df(raw.assign(lookup_protein_names="", lookup_organism=""), "AFDB (Clustered)")
    assert list(afdb.columns) == ["Query", "AFDB / UniProt Accession", "Exact Prob", "Partial Prob"]

    scope = _format_results_display_df(raw.assign(lookup_meta=">d1abcA SCOPe domain", lookup_entry="d1abcA"), "SCOPE")
    assert list(scope.columns) == ["Query", "SCOPe Domain", "Exact Prob", "Partial Prob"]
