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


def test_create_interface_smoke_builds_blocks_with_theme_css_and_device_option():
    gr = pytest.importorskip("gradio")
    from protein_conformal.backend.gradio_interface import create_interface

    demo = create_interface()
    assert isinstance(demo, gr.Blocks)
    assert hasattr(demo, "cpr_theme")
    assert hasattr(demo, "cpr_css")
    assert "#fasta-input" in demo.cpr_css

    labels = [getattr(block, "label", None) for block in demo.blocks.values()]
    assert "Embedding Device" in labels

    embedding_device_blocks = [
        block for block in demo.blocks.values()
        if getattr(block, "label", None) == "Embedding Device"
    ]
    assert embedding_device_blocks
    choices = [choice[0] if isinstance(choice, (tuple, list)) else choice
               for choice in embedding_device_blocks[0].choices]
    assert "CPU" in choices
    assert "Apple MPS (experimental)" in choices
