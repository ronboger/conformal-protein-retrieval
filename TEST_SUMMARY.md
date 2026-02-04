# CPR Test Suite Summary

## Test Files

### 1. `tests/test_util.py` - Core Algorithm Tests (27 tests)
Tests for conformal prediction algorithms in `protein_conformal/util.py`:
- FDR threshold calculation (`get_thresh_FDR`, `get_thresh_new_FDR`)
- FNR threshold calculation (`get_thresh_new`)
- Venn-Abers calibration (`simplifed_venn_abers_prediction`)
- SCOPe hierarchical loss (`scope_hierarchical_loss`)
- FAISS database operations (`load_database`, `query`)
- FASTA file parsing (`read_fasta`)

**Status**: ✅ All 27 tests passing

### 2. `tests/test_cli.py` - CLI Integration Tests (24 tests)
Tests for command-line interface in `protein_conformal/cli.py`:

#### Help Text Tests (7 tests)
- Main help and all subcommand help screens
- Verifies all expected options are documented

#### Argument Validation Tests (4 tests)
- Missing required arguments
- Invalid argument values
- Graceful error handling

#### Search Command Tests (5 tests)
- Basic search with mock embeddings
- Threshold filtering
- Metadata merging
- Edge cases (k > database size)
- Missing file handling

#### Probability Conversion Tests (3 tests)
- Converting .npy scores
- Converting CSV scores (from search results)
- Venn-Abers calibration

#### Calibration Tests (2 tests)
- Computing FDR/FNR thresholds
- Multiple calibration trials

#### Error Handling Tests (3 tests)
- Missing input files
- Missing database files
- Missing calibration files

**Status**: ✅ Created and verified (24 tests)

### 3. `tests/conftest.py` - Shared Test Fixtures
Pytest fixtures used across test files:
- `sample_fasta_file` - Temporary FASTA with 3 proteins
- `sample_embeddings` - Random embeddings (10 query, 100 lookup)
- `scope_like_data` - Synthetic SCOPe-like data (40 queries, 100 lookup)
- `calibration_test_split` - Train/test split for calibration

## Test Coverage by CLI Command

| Command | Help Test | Integration Test | Error Handling | Count |
|---------|-----------|------------------|----------------|-------|
| `cpr` (main) | ✅ | ✅ | ✅ | 3 |
| `cpr embed` | ✅ | ⚠️ Mock only | ✅ | 3 |
| `cpr search` | ✅ | ✅ | ✅ | 8 |
| `cpr verify` | ✅ | ⚠️ Subprocess | ✅ | 3 |
| `cpr prob` | ✅ | ✅ | ✅ | 4 |
| `cpr calibrate` | ✅ | ✅ | ✅ | 3 |

**Legend:**
- ✅ Fully tested
- ⚠️ Partial coverage (see notes)
- ❌ Not tested

## Running All Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_cli.py -v
pytest tests/test_util.py -v

# Run with coverage
pytest tests/ --cov=protein_conformal --cov-report=html

# Run specific test
pytest tests/test_cli.py::test_search_with_mock_data -v
```

## Test Requirements

### Environment
- Python 3.8+
- pytest
- numpy
- pandas
- faiss-cpu (or faiss-gpu)
- scikit-learn
- biopython (for FASTA parsing)

### Data Requirements
- **None** - All tests use synthetic/mock data
- Tests create temporary files in pytest's `tmp_path`
- Tests clean up after themselves

### Compute Requirements
- **CPU only** - No GPU required
- **Memory**: < 1 GB (mock data is small)
- **Time**: All 51 tests complete in < 30 seconds

## Coverage Gaps

### Not Yet Tested
1. **Embed command with real models**
   - Would require downloading ProtTrans/CLEAN models (>10 GB)
   - Current test only checks missing file errors
   - **Recommendation**: Add mock model test or skip in CI

2. **Verify command end-to-end**
   - Requires real verification scripts in `scripts/`
   - Current test only checks subprocess call
   - **Recommendation**: Add integration test with small mock data

3. **Multi-model workflows**
   - Testing `--model protein-vec` vs `--model clean`
   - Testing model-specific calibration
   - **Recommendation**: Add when CLEAN integration is complete

4. **Performance tests**
   - Large database search (1M+ proteins)
   - Calibration with 10K+ samples
   - **Recommendation**: Add separate performance test suite

## Paper Verification Tests

Separate verification scripts in `scripts/`:
- `verify_syn30.py` - JCVI Syn3.0 annotation (Figure 2A)
- `verify_fdr_algorithm.py` - FDR threshold calculation
- `verify_dali.py` - DALI prefiltering (Tables 4-6)
- `verify_clean.py` - CLEAN enzyme classification (Tables 1-2)

These can be run via: `cpr verify --check [syn30|fdr|dali|clean]`

## Adding New Tests

### For New CLI Commands
1. Add help test: `test_<command>_help()`
2. Add integration test: `test_<command>_with_mock_data(tmp_path)`
3. Add error handling: `test_<command>_missing_<required_arg>()`

### For New Algorithms
1. Add unit test in `tests/test_util.py`
2. Use fixtures from `tests/conftest.py`
3. Compare against expected values (with tolerance)

### Best Practices
- Use `tmp_path` fixture for file operations
- Set random seeds for reproducibility
- Keep test data small (< 100 samples)
- Test edge cases (empty input, k=0, etc.)
- Test error messages, not just return codes

## CI/CD Integration

Recommended GitHub Actions workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          conda install -c conda-forge faiss-cpu pytest pytest-cov
          pip install -e .
      - name: Run tests
        run: pytest tests/ -v --cov=protein_conformal
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Maintenance

### Before Each Release
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Run paper verification: `cpr verify --check [all]`
- [ ] Check test coverage: `pytest --cov=protein_conformal --cov-report=term-missing`
- [ ] Update test expectations if algorithms change

### When Adding Features
- [ ] Add unit tests for new functions
- [ ] Add CLI tests for new commands
- [ ] Update this summary document
- [ ] Add examples to test README

### When Fixing Bugs
- [ ] Add regression test that fails before fix
- [ ] Verify test passes after fix
- [ ] Add to test_util.py or test_cli.py as appropriate
