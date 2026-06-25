"""
Tests for showing an uploaded FASTA file's content in the text box.

Note: gradio is not installed in the test environment, so we mock it before import
(same pattern as test_clean.py).
"""
import sys
from unittest.mock import MagicMock

_gr_mock = MagicMock()
_gr_mock.Progress = MagicMock(return_value=MagicMock())
sys.modules.setdefault("gradio", _gr_mock)

FASTA = ">queryA test\nMVLSPADKTNVKAAWGKVGA\n"


def test_load_uploaded_fasta_reads_path(tmp_path):
    """Uploading a file (by path) returns its raw text for the text box."""
    import protein_conformal.backend.gradio_interface as gi
    p = tmp_path / "query.fasta"
    p.write_text(FASTA)
    out = gi.load_uploaded_fasta(str(p))
    assert ">queryA" in out
    assert "MVLSPADKTNVKAAWGKVGA" in out


def test_load_uploaded_fasta_reads_filelike(tmp_path):
    """A file-like object (has .read()) is read directly."""
    import io
    import protein_conformal.backend.gradio_interface as gi
    out = gi.load_uploaded_fasta(io.BytesIO(FASTA.encode()))
    assert ">queryA" in out


def test_load_uploaded_fasta_none_returns_empty():
    """No upload -> empty string (clears the box), never an error."""
    import protein_conformal.backend.gradio_interface as gi
    assert gi.load_uploaded_fasta(None) == ""


def test_process_uploaded_file_still_parses(tmp_path):
    """The refactor keeps process_uploaded_file parsing into sequences."""
    import protein_conformal.backend.gradio_interface as gi
    p = tmp_path / "query.fasta"
    p.write_text(FASTA)
    seqs, meta = gi.process_uploaded_file(str(p))
    assert seqs == ["MVLSPADKTNVKAAWGKVGA"]
    assert len(meta) == 1
