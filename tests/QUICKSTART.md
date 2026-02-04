# CLI Test Suite Quickstart

## Prerequisites

Ensure you have the conda environment activated:
```bash
conda activate conformal-s
```

## Running Tests

### Run all CLI tests
```bash
cd /groups/doudna/projects/ronb/conformal-protein-retrieval
pytest tests/test_cli.py -v
```

Expected output:
```
tests/test_cli.py::test_main_help PASSED                            [  4%]
tests/test_cli.py::test_main_no_command PASSED                      [  8%]
tests/test_cli.py::test_embed_help PASSED                           [ 12%]
tests/test_cli.py::test_search_help PASSED                          [ 16%]
...
======================== 24 passed in 2.34s ========================
```

### Run a single test
```bash
pytest tests/test_cli.py::test_search_with_mock_data -v
```

### Run tests with detailed output
```bash
pytest tests/test_cli.py -v -s
```
The `-s` flag shows print statements from the code.

### Run tests and see which code is tested
```bash
pytest tests/test_cli.py --cov=protein_conformal.cli --cov-report=term-missing
```

## What Each Test Does

### Help Tests (fast, no computation)
```bash
# These verify help text is correct
pytest tests/test_cli.py -k "help" -v
```
Tests: `test_*_help` (7 tests)
- Verifies all commands have proper documentation
- Checks that all options are listed
- Confirms command structure is correct

### Search Tests (uses mock data)
```bash
# These test the search functionality
pytest tests/test_cli.py -k "search" -v
```
Tests: `test_search_*` (8 tests)
- Creates small mock embeddings (5x128 and 20x128)
- Tests FAISS similarity search
- Tests threshold filtering
- Tests metadata merging
- Tests edge cases

### Probability Tests (uses mock calibration)
```bash
# These test probability conversion
pytest tests/test_cli.py -k "prob" -v
```
Tests: `test_prob_*` (3 tests)
- Creates mock calibration data
- Tests Venn-Abers probability conversion
- Tests CSV input/output

### Calibration Tests (uses mock data)
```bash
# These test threshold calibration
pytest tests/test_cli.py -k "calibrate" -v
```
Tests: `test_calibrate_*` (2 tests)
- Creates mock similarity/label pairs
- Tests FDR/FNR threshold computation
- Tests multiple calibration trials

## Example Test Walkthrough

Let's look at `test_search_with_mock_data()` in detail:

```python
def test_search_with_mock_data(tmp_path):
    """Test search command with small mock embeddings."""
    # 1. Create mock query embeddings (5 proteins, 128-dim)
    query_embeddings = np.random.randn(5, 128).astype(np.float32)

    # 2. Create mock database embeddings (20 proteins, 128-dim)
    db_embeddings = np.random.randn(20, 128).astype(np.float32)

    # 3. Normalize to unit vectors (for cosine similarity)
    query_embeddings = query_embeddings / np.linalg.norm(...)
    db_embeddings = db_embeddings / np.linalg.norm(...)

    # 4. Save to temporary files
    np.save(tmp_path / "query.npy", query_embeddings)
    np.save(tmp_path / "db.npy", db_embeddings)

    # 5. Run CLI command via subprocess
    subprocess.run([
        sys.executable, '-m', 'protein_conformal.cli',
        'search',
        '--query', str(tmp_path / "query.npy"),
        '--database', str(tmp_path / "db.npy"),
        '--output', str(tmp_path / "results.csv"),
        '--k', '3'
    ])

    # 6. Verify output exists and has correct structure
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 5 * 3  # 5 queries * 3 neighbors
    assert 'similarity' in df.columns
```

## Understanding Test Failures

### Import Errors
```
ModuleNotFoundError: No module named 'faiss'
```
**Solution**: Install dependencies
```bash
conda install -c conda-forge faiss-cpu
```

### File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/...'
```
**Solution**: This shouldn't happen with `tmp_path` fixture. Check that pytest is creating temp directories.

### Assertion Errors
```
AssertionError: assert 8 == 15
```
**Solution**: Check if test expectations match actual behavior. This could indicate:
- Bug in code
- Test expectations wrong
- Random seed not working

### Subprocess Errors
```
subprocess.CalledProcessError: Command returned non-zero exit status 1
```
**Solution**: Run the command manually to see error:
```bash
python -m protein_conformal.cli search --query test.npy --database db.npy ...
```

## Adding Your Own Test

Template for a new CLI test:

```python
def test_my_new_feature(tmp_path):
    """Test description here."""
    # 1. Create test data
    test_data = np.array([1, 2, 3])
    input_file = tmp_path / "input.npy"
    np.save(input_file, test_data)

    # 2. Run CLI command
    result = subprocess.run(
        [sys.executable, '-m', 'protein_conformal.cli',
         'my-command',
         '--input', str(input_file),
         '--output', str(tmp_path / "output.csv")],
        capture_output=True,
        text=True
    )

    # 3. Check return code
    assert result.returncode == 0

    # 4. Verify output
    output_file = tmp_path / "output.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) > 0
    assert 'expected_column' in df.columns
```

## Debugging Tests

### Run test with debugger
```bash
pytest tests/test_cli.py::test_search_with_mock_data --pdb
```
This will drop into Python debugger on failure.

### Show print statements
```bash
pytest tests/test_cli.py::test_search_with_mock_data -s
```
This shows any `print()` statements from the code.

### Show warnings
```bash
pytest tests/test_cli.py -v -W all
```
This shows all Python warnings (deprecation, etc.)

### Keep temporary files
```bash
pytest tests/test_cli.py::test_search_with_mock_data --basetemp=./test_tmp
```
This keeps temp files in `./test_tmp/` for inspection.

## Performance

All 24 CLI tests should complete in **< 30 seconds**:
- Help tests: ~0.1s each (no computation)
- Mock data tests: ~0.5-2s each (small arrays)
- No GPU required
- No large data files

If tests are slow:
1. Check if GPU is being initialized (use `--cpu` flag)
2. Check calibration data size (should be < 100 samples in tests)
3. Check for network calls (shouldn't happen in these tests)

## Next Steps

After CLI tests pass:
1. Run full test suite: `pytest tests/ -v`
2. Run paper verification: `cpr verify --check syn30`
3. Try the CLI on real data: `cpr search --query ... --database ...`
4. Read `TEST_SUMMARY.md` for complete test documentation
