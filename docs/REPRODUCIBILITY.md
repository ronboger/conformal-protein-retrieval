# Reproducibility Notes

This document explains expected variability when reproducing results from the paper
"Functional protein mining with conformal guarantees" (Nature Communications 2025).

## FDR Threshold Variability

The FDR-controlling thresholds are computed using Learn-then-Test (LTT) calibration,
which involves random sampling of calibration data. This introduces expected variability:

### Paper Results (α = 0.1)
- **Reported threshold**: λ = 0.9999802250
- **JCVI Syn3.0 hits**: 59/149 (39.6%)

### Reproduction Results
- **Computed threshold**: λ = 0.9999802250 ± ~2e-6 (varies by trial)
- **Observed hits**: 58-60/149 (38.9-40.3%)

### Why Results May Differ by ±1 Hit

The 59th protein in the Syn3.0 dataset has a similarity score extremely close to
the FDR threshold:

| Protein Rank | Similarity Score | vs Threshold (λ = 0.9999802250) |
|--------------|------------------|----------------------------------|
| 58th         | 0.999980390      | +1.65×10⁻⁷ (above threshold)    |
| **59th**     | **0.999980032**  | **-1.93×10⁻⁷ (below threshold)**|
| 60th         | 0.999979556      | -6.69×10⁻⁷ (below threshold)    |

The difference between the 59th protein's score and the threshold is only **0.00002%**.
This means:
- Small variations in the computed threshold (from different calibration samples)
  can flip this protein above or below the threshold
- This is expected behavior for conformal methods - the guarantee is statistical
  (FDR ≤ α on average), not that every run produces identical results

### Recommended Practice

1. **Use the lookup table**: Pre-computed thresholds in `results/fdr_thresholds.csv`
   provide stable, reproducible values averaged over 100 calibration trials.

2. **Report uncertainty**: When reporting results, include the threshold uncertainty
   (e.g., λ = 0.99998 ± 2×10⁻⁶) to indicate expected variability.

3. **Set random seeds**: For exact reproduction, use the same random seed when
   computing thresholds:
   ```python
   np.random.seed(42)
   ```

4. **Use sufficient trials**: The paper uses 100 calibration trials to compute
   stable threshold estimates. Fewer trials increase variability.

## FDR Threshold Lookup Table

Pre-computed thresholds for common alpha levels (see `results/fdr_thresholds.csv`):

| Alpha (α) | Threshold (λ) | Use Case |
|-----------|---------------|----------|
| 0.001     | ~0.99999+     | Very stringent (0.1% FDR) |
| 0.01      | ~0.99999      | Stringent (1% FDR) |
| 0.05      | ~0.99998      | Moderate (5% FDR) |
| **0.10**  | **0.99998**   | **Paper default (10% FDR)** |
| 0.15      | ~0.99997      | Relaxed (15% FDR) |
| 0.20      | ~0.99996      | Discovery-focused (20% FDR) |

Note: Exact values depend on calibration data and are computed by:
```bash
sbatch scripts/slurm_compute_fdr_thresholds.sh
```

## Calibration Data

The correct calibration dataset is `data/pfam_new_proteins.npy` (from Zenodo).

**WARNING**: Do not use `conformal_pfam_with_lookup_dataset.npy` - this dataset
has data leakage (the first 50 samples share the same Pfam family "PF01266;").
See `DEVELOPMENT.md` for details.

## Verification Commands

To verify paper results:

```bash
# Verify JCVI Syn3.0 annotation rate
cpr verify --check syn30

# Verify FDR threshold computation
cpr verify --check fdr

# Verify DALI prefiltering
cpr verify --check dali

# Verify CLEAN enzyme classification
cpr verify --check clean
```

Expected output for `cpr verify --check syn30`:
- Hits: 58-60 out of 149 (38.9-40.3%)
- Threshold: λ ≈ 0.99998

The ±1 hit variability is expected due to the borderline case described above.
