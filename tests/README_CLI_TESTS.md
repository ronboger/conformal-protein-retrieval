# CLI Test Suite Documentation

## Overview

`test_cli.py` contains comprehensive integration tests for the CPR command-line interface (`protein_conformal/cli.py`).

## Test Categories

### 1. Help Text Tests (7 tests)
Verify that help text is displayed correctly for all commands:
- `test_main_help()` - Main `cpr --help` shows all subcommands
- `test_main_no_command()` - Running `cpr` with no args shows help
- `test_embed_help()` - `cpr embed --help` shows embedding options
- `test_search_help()` - `cpr search --help` shows search options
- `test_verify_help()` - `cpr verify --help` shows verification options
- `test_prob_help()` - `cpr prob --help` shows probability conversion options
- `test_calibrate_help()` - `cpr calibrate --help` shows calibration options

### 2. Missing Arguments Tests (4 tests)
Verify that commands fail gracefully when required arguments are missing:
- `test_embed_missing_args()` - Embed requires --input and --output
- `test_search_missing_args()` - Search requires --query, --database, --output
- `test_verify_missing_args()` - Verify requires --check
- `test_verify_invalid_check()` - Verify rejects invalid check names

### 3. Search Integration Tests (5 tests)
Test the search command with various scenarios using mock data:
- `test_search_with_mock_data()` - Basic search with 5 queries x 20 database
- `test_search_with_threshold()` - Search with similarity threshold filtering
- `test_search_with_metadata()` - Search with database metadata CSV
- `test_search_with_k_larger_than_database()` - Edge case: k > database size
- `test_search_missing_query_file()` - Error handling for missing query file
- `test_search_missing_database_file()` - Error handling for missing database

### 4. Probability Conversion Tests (3 tests)
Test the prob command for converting similarity scores to calibrated probabilities:
- `test_prob_with_mock_data()` - Convert .npy scores using mock calibration
- `test_prob_with_csv_input()` - Convert scores in CSV (e.g., search results)
- `test_prob_missing_calibration_file()` - Error handling for missing calibration

### 5. Calibration Tests (2 tests)
Test the calibrate command for computing FDR/FNR thresholds:
- `test_calibrate_with_mock_data()` - Calibrate thresholds using mock data
- `test_calibrate_missing_calibration_file()` - Error handling for missing data

### 6. File Handling Tests (3 tests)
Test error handling for missing/invalid files:
- `test_embed_missing_input_file()` - Embed fails on missing FASTA
- `test_search_missing_query_file()` - Search fails on missing query
- `test_search_missing_database_file()` - Search fails on missing database

### 7. Module Import Test (1 test)
- `test_cli_module_import()` - Verify CLI module structure and exports

## Running the Tests

### Run all CLI tests:
```bash
pytest tests/test_cli.py -v
```

### Run specific test:
```bash
pytest tests/test_cli.py::test_search_with_mock_data -v
```

### Run with coverage:
```bash
pytest tests/test_cli.py --cov=protein_conformal.cli --cov-report=term-missing
```

## Design Principles

1. **No GPU Required**: All tests use small mock data and can run on CPU
2. **No Large Data Files**: Tests create synthetic data in memory
3. **Fast Execution**: Each test completes in < 1 second
4. **Isolated**: Tests use temporary directories (pytest's `tmp_path` fixture)
5. **Realistic**: Mock data mimics structure of real calibration/embedding data

## Mock Data Structure

### Embeddings (for search tests)
- Shape: (n_samples, 128) float32
- Normalized to unit vectors for cosine similarity
- Small sizes: 2-20 samples for speed

### Calibration Data (for prob/calibrate tests)
- Structure: array of (query_emb, lookup_emb, sims, labels, metadata)
- `sims`: similarity scores in [0.997, 0.9999] (realistic protein range)
- `labels`: binary labels (0/1) for matches
- Size: 30-100 samples for speed

### Metadata (for search tests)
- CSV/TSV with columns: protein_id, description, organism
- Merged with search results using match_idx

## Common Issues

### Import Errors
If tests fail with import errors, ensure the environment has:
- numpy
- pandas
- pytest
- faiss-cpu or faiss-gpu
- scikit-learn

### Path Issues
Tests use `subprocess` to call the CLI, which requires:
- `protein_conformal` package installed or in PYTHONPATH
- Or run from repo root with package in current directory

### Slow Tests
If tests are slow:
- Check n_trials in calibrate tests (should be 5-10 for tests)
- Check calibration data size (should be < 100 samples)
- Verify no GPU initialization happening (use --cpu flag if needed)

## Future Enhancements

- [ ] Add test for `cpr embed` with tiny mock model (requires mocking transformers)
- [ ] Add integration test that chains: embed → search → prob
- [ ] Add test for verify command (requires mock verification data)
- [ ] Add performance benchmarks for large-scale search
- [ ] Add test for search with precomputed probabilities
