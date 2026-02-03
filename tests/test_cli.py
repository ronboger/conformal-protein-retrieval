"""
Tests for CPR CLI (protein_conformal/cli.py).

Tests cover:
- Help text for all commands
- Basic functionality with mock data
- Error handling
"""
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def run_cli(*args):
    """Helper to run CLI commands via subprocess."""
    result = subprocess.run(
        [sys.executable, '-m', 'protein_conformal.cli'] + list(args),
        capture_output=True,
        text=True
    )
    return result


def test_main_help():
    """Test that 'cpr --help' shows all subcommands."""
    result = run_cli('--help')
    assert result.returncode == 0
    assert 'embed' in result.stdout
    assert 'search' in result.stdout
    assert 'verify' in result.stdout
    assert 'prob' in result.stdout
    assert 'calibrate' in result.stdout
    assert 'Conformal Protein Retrieval' in result.stdout


def test_main_no_command():
    """Test that running cpr with no command shows help."""
    result = run_cli()
    assert result.returncode == 1
    # Should show help when no command provided
    assert 'embed' in result.stdout or 'embed' in result.stderr


def test_embed_help():
    """Test that 'cpr embed --help' works and shows expected options."""
    result = run_cli('embed', '--help')
    assert result.returncode == 0
    assert '--input' in result.stdout
    assert '--output' in result.stdout
    assert '--model' in result.stdout
    assert 'protein-vec' in result.stdout
    assert 'clean' in result.stdout
    assert '--cpu' in result.stdout


def test_search_help():
    """Test that 'cpr search --help' works."""
    result = run_cli('search', '--help')
    assert result.returncode == 0
    assert '--query' in result.stdout
    assert '--database' in result.stdout
    assert '--output' in result.stdout
    assert '--k' in result.stdout
    assert '--threshold' in result.stdout
    assert '--database-meta' in result.stdout


def test_verify_help():
    """Test that 'cpr verify --help' works."""
    result = run_cli('verify', '--help')
    assert result.returncode == 0
    assert '--check' in result.stdout
    assert 'syn30' in result.stdout
    assert 'fdr' in result.stdout
    assert 'dali' in result.stdout
    assert 'clean' in result.stdout


def test_prob_help():
    """Test that 'cpr prob --help' works."""
    result = run_cli('prob', '--help')
    assert result.returncode == 0
    assert '--input' in result.stdout
    assert '--calibration' in result.stdout
    assert '--output' in result.stdout
    assert '--score-column' in result.stdout
    assert '--n-calib' in result.stdout
    assert '--seed' in result.stdout


def test_calibrate_help():
    """Test that 'cpr calibrate --help' works."""
    result = run_cli('calibrate', '--help')
    assert result.returncode == 0
    assert '--calibration' in result.stdout
    assert '--output' in result.stdout
    assert '--alpha' in result.stdout
    assert '--n-trials' in result.stdout
    assert '--n-calib' in result.stdout
    assert '--method' in result.stdout
    assert 'ltt' in result.stdout
    assert 'quantile' in result.stdout


def test_embed_missing_args():
    """Test that embed command fails without required args."""
    result = run_cli('embed')
    assert result.returncode != 0
    assert '--input' in result.stderr or 'required' in result.stderr


def test_search_missing_args():
    """Test that search command fails without required args."""
    result = run_cli('search')
    assert result.returncode != 0
    assert '--query' in result.stderr or 'required' in result.stderr


def test_verify_missing_args():
    """Test that verify command fails without required args."""
    result = run_cli('verify')
    assert result.returncode != 0
    assert '--check' in result.stderr or 'required' in result.stderr


def test_verify_invalid_check():
    """Test that verify command fails with invalid check name."""
    result = run_cli('verify', '--check', 'invalid_check_name')
    assert result.returncode != 0


def test_search_with_mock_data(tmp_path):
    """Test search command with small mock embeddings."""
    # Create mock query and database embeddings
    np.random.seed(42)
    query_embeddings = np.random.randn(5, 128).astype(np.float32)
    db_embeddings = np.random.randn(20, 128).astype(np.float32)

    # Normalize to unit vectors (for cosine similarity)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    # Save to temp files
    query_file = tmp_path / "query.npy"
    db_file = tmp_path / "db.npy"
    output_file = tmp_path / "results.csv"

    np.save(query_file, query_embeddings)
    np.save(db_file, db_embeddings)

    # Run search
    result = run_cli(
        'search',
        '--query', str(query_file),
        '--database', str(db_file),
        '--output', str(output_file),
        '--k', '3'
    )

    assert result.returncode == 0
    assert output_file.exists()

    # Verify output
    df = pd.read_csv(output_file)
    assert len(df) == 5 * 3  # 5 queries * 3 neighbors
    assert 'query_idx' in df.columns
    assert 'match_idx' in df.columns
    assert 'similarity' in df.columns

    # Check that similarities are reasonable (cosine similarity range)
    assert df['similarity'].min() >= -1.0
    assert df['similarity'].max() <= 1.0


def test_search_with_threshold(tmp_path):
    """Test search command with similarity threshold."""
    np.random.seed(42)
    query_embeddings = np.random.randn(3, 128).astype(np.float32)
    db_embeddings = np.random.randn(10, 128).astype(np.float32)

    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    query_file = tmp_path / "query.npy"
    db_file = tmp_path / "db.npy"
    output_file = tmp_path / "results.csv"

    np.save(query_file, query_embeddings)
    np.save(db_file, db_embeddings)

    # Run search with high threshold
    result = run_cli(
        'search',
        '--query', str(query_file),
        '--database', str(db_file),
        '--output', str(output_file),
        '--k', '10',
        '--threshold', '0.9'
    )

    assert result.returncode == 0
    assert output_file.exists()

    # With high threshold on random embeddings, file may be empty or have few results
    # Random unit vectors have expected cosine similarity ~0, so 0.9 threshold filters most
    try:
        df = pd.read_csv(output_file)
        # With high threshold, we should have fewer results
        assert len(df) <= 3 * 10  # At most 3 queries * 10 neighbors
        # All results should be above threshold
        if len(df) > 0:
            assert df['similarity'].min() >= 0.9
    except pd.errors.EmptyDataError:
        # Empty file is valid - no results passed threshold
        pass


def test_search_with_metadata(tmp_path):
    """Test search command with database metadata."""
    np.random.seed(42)
    query_embeddings = np.random.randn(2, 128).astype(np.float32)
    db_embeddings = np.random.randn(5, 128).astype(np.float32)

    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    query_file = tmp_path / "query.npy"
    db_file = tmp_path / "db.npy"
    meta_file = tmp_path / "meta.csv"
    output_file = tmp_path / "results.csv"

    np.save(query_file, query_embeddings)
    np.save(db_file, db_embeddings)

    # Create metadata
    meta_df = pd.DataFrame({
        'protein_id': [f'PROT_{i:03d}' for i in range(5)],
        'description': [f'Protein {i}' for i in range(5)],
        'organism': ['E. coli', 'Human', 'Yeast', 'Mouse', 'Rat'],
    })
    meta_df.to_csv(meta_file, index=False)

    # Run search with metadata
    result = run_cli(
        'search',
        '--query', str(query_file),
        '--database', str(db_file),
        '--database-meta', str(meta_file),
        '--output', str(output_file),
        '--k', '3'
    )

    assert result.returncode == 0
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 2 * 3  # 2 queries * 3 neighbors
    # Check that metadata columns were added
    assert 'match_protein_id' in df.columns
    assert 'match_description' in df.columns
    assert 'match_organism' in df.columns


def test_prob_with_mock_data(tmp_path):
    """Test prob command with mock calibration data and scores."""
    np.random.seed(42)

    # Create mock calibration data (format: array of dicts with S_i, exact, partial)
    n_calib = 50
    cal_data = []
    for i in range(n_calib):
        sims = np.random.uniform(0.998, 0.9999, size=10).astype(np.float32)
        exact_labels = (np.random.random(10) < 0.2).astype(bool)
        partial_labels = exact_labels | (np.random.random(10) < 0.1)
        cal_data.append({
            "S_i": sims,
            "exact": exact_labels,
            "partial": partial_labels,
        })

    cal_file = tmp_path / "calibration.npy"
    np.save(cal_file, np.array(cal_data, dtype=object))

    # Create input scores
    scores = np.array([0.9985, 0.9990, 0.9995, 0.9998])
    score_file = tmp_path / "scores.npy"
    np.save(score_file, scores)

    output_file = tmp_path / "probs.csv"

    # Run prob command
    result = run_cli(
        'prob',
        '--input', str(score_file),
        '--calibration', str(cal_file),
        '--output', str(output_file),
        '--n-calib', '50',
        '--seed', '42'
    )

    assert result.returncode == 0
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 4
    assert 'score' in df.columns
    assert 'probability' in df.columns
    assert 'uncertainty' in df.columns

    # Probabilities should be in [0, 1]
    assert df['probability'].min() >= 0.0
    assert df['probability'].max() <= 1.0
    # Uncertainties should be in [0, 1]
    assert df['uncertainty'].min() >= 0.0
    assert df['uncertainty'].max() <= 1.0


def test_prob_with_csv_input(tmp_path):
    """Test prob command with CSV input (e.g., from search results)."""
    np.random.seed(42)

    # Create mock calibration data (format: array of dicts with S_i, exact, partial)
    n_calib = 30
    cal_data = []
    for i in range(n_calib):
        sims = np.random.uniform(0.998, 0.9999, size=5).astype(np.float32)
        exact_labels = (np.random.random(5) < 0.2).astype(bool)
        partial_labels = exact_labels | (np.random.random(5) < 0.1)
        cal_data.append({
            "S_i": sims,
            "exact": exact_labels,
            "partial": partial_labels,
        })

    cal_file = tmp_path / "calibration.npy"
    np.save(cal_file, np.array(cal_data, dtype=object))

    # Create CSV input with similarity scores
    input_df = pd.DataFrame({
        'query_idx': [0, 0, 1, 1],
        'match_idx': [5, 10, 3, 8],
        'similarity': [0.9985, 0.9990, 0.9995, 0.9998],
        'match_protein_id': ['PROT_A', 'PROT_B', 'PROT_C', 'PROT_D'],
    })
    input_file = tmp_path / "input.csv"
    input_df.to_csv(input_file, index=False)

    output_file = tmp_path / "output.csv"

    # Run prob command
    result = run_cli(
        'prob',
        '--input', str(input_file),
        '--calibration', str(cal_file),
        '--output', str(output_file),
        '--score-column', 'similarity',
        '--n-calib', '30'
    )

    assert result.returncode == 0
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 4
    # Original columns should be preserved
    assert 'query_idx' in df.columns
    assert 'match_idx' in df.columns
    assert 'similarity' in df.columns
    assert 'match_protein_id' in df.columns
    # New columns should be added
    assert 'probability' in df.columns
    assert 'uncertainty' in df.columns


def test_calibrate_with_mock_data(tmp_path):
    """Test calibrate command with mock calibration data."""
    np.random.seed(42)

    # Create mock calibration data (format: array of dicts with S_i, exact, partial)
    n_samples = 100
    cal_data = []
    for i in range(n_samples):
        sims = np.random.uniform(0.997, 0.9999, size=10).astype(np.float32)
        # Create labels: higher similarity -> higher chance of being positive
        exact_labels = (sims > 0.9995).astype(bool)
        partial_labels = (sims > 0.999).astype(bool)
        cal_data.append({
            "S_i": sims,
            "exact": exact_labels,
            "partial": partial_labels,
        })

    cal_file = tmp_path / "calibration.npy"
    np.save(cal_file, np.array(cal_data, dtype=object))

    output_file = tmp_path / "thresholds.csv"

    # Run calibrate command (small number of trials for speed)
    result = run_cli(
        'calibrate',
        '--calibration', str(cal_file),
        '--output', str(output_file),
        '--alpha', '0.1',
        '--n-trials', '5',
        '--n-calib', '50',
        '--method', 'quantile',
        '--seed', '42'
    )

    assert result.returncode == 0
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 5  # 5 trials
    assert 'trial' in df.columns
    assert 'alpha' in df.columns
    assert 'fdr_threshold' in df.columns
    assert 'fnr_threshold' in df.columns

    # All alpha values should be 0.1
    assert (df['alpha'] == 0.1).all()
    # Thresholds should be in reasonable range
    assert df['fdr_threshold'].min() > 0.0
    assert df['fdr_threshold'].max() <= 1.0
    assert df['fnr_threshold'].min() > 0.0
    assert df['fnr_threshold'].max() <= 1.0


def test_embed_missing_input_file():
    """Test that embed fails gracefully with missing input file."""
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        output_file = tmp.name

    try:
        result = run_cli(
            'embed',
            '--input', '/nonexistent/file.fasta',
            '--output', output_file
        )
        assert result.returncode != 0
    finally:
        Path(output_file).unlink(missing_ok=True)


def test_search_missing_query_file(tmp_path):
    """Test that search fails gracefully with missing query file."""
    # Create a valid database file
    db_embeddings = np.random.randn(10, 128).astype(np.float32)
    db_file = tmp_path / "db.npy"
    np.save(db_file, db_embeddings)

    output_file = tmp_path / "results.csv"

    result = run_cli(
        'search',
        '--query', '/nonexistent/query.npy',
        '--database', str(db_file),
        '--output', str(output_file)
    )
    assert result.returncode != 0


def test_search_missing_database_file(tmp_path):
    """Test that search fails gracefully with missing database file."""
    # Create a valid query file
    query_embeddings = np.random.randn(5, 128).astype(np.float32)
    query_file = tmp_path / "query.npy"
    np.save(query_file, query_embeddings)

    output_file = tmp_path / "results.csv"

    result = run_cli(
        'search',
        '--query', str(query_file),
        '--database', '/nonexistent/db.npy',
        '--output', str(output_file)
    )
    assert result.returncode != 0


def test_prob_missing_calibration_file(tmp_path):
    """Test that prob fails gracefully with missing calibration file."""
    scores = np.array([0.998, 0.999])
    score_file = tmp_path / "scores.npy"
    np.save(score_file, scores)

    output_file = tmp_path / "probs.csv"

    result = run_cli(
        'prob',
        '--input', str(score_file),
        '--calibration', '/nonexistent/calibration.npy',
        '--output', str(output_file)
    )
    assert result.returncode != 0


def test_calibrate_missing_calibration_file(tmp_path):
    """Test that calibrate fails gracefully with missing calibration file."""
    output_file = tmp_path / "thresholds.csv"

    result = run_cli(
        'calibrate',
        '--calibration', '/nonexistent/calibration.npy',
        '--output', str(output_file),
        '--n-trials', '1'
    )
    assert result.returncode != 0


def test_search_with_k_larger_than_database(tmp_path):
    """Test search when k is larger than database size."""
    np.random.seed(42)
    query_embeddings = np.random.randn(2, 128).astype(np.float32)
    db_embeddings = np.random.randn(3, 128).astype(np.float32)  # Only 3 items

    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    query_file = tmp_path / "query.npy"
    db_file = tmp_path / "db.npy"
    output_file = tmp_path / "results.csv"

    np.save(query_file, query_embeddings)
    np.save(db_file, db_embeddings)

    # Request k=10 but only have 3 items in database
    result = run_cli(
        'search',
        '--query', str(query_file),
        '--database', str(db_file),
        '--output', str(output_file),
        '--k', '10'
    )

    # Should succeed (FAISS will return at most db size)
    assert result.returncode == 0
    assert output_file.exists()

    df = pd.read_csv(output_file)
    # Should have at most 2 * 3 = 6 results (2 queries, 3 db items each)
    assert len(df) <= 6


def test_cli_module_import():
    """Test that CLI module can be imported and has expected functions."""
    from protein_conformal import cli

    assert hasattr(cli, 'main')
    assert hasattr(cli, 'cmd_embed')
    assert hasattr(cli, 'cmd_search')
    assert hasattr(cli, 'cmd_verify')
    assert hasattr(cli, 'cmd_prob')
    assert hasattr(cli, 'cmd_calibrate')
    assert callable(cli.main)
