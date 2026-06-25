# Claude Code Guidelines for CPR

## Working Patterns

### Before Writing Code
- **Describe your approach first** and wait for approval before implementing
- **Ask clarifying questions** if requirements are ambiguous - don't assume
- **If a task requires changes to more than 3 files**, stop and break it into smaller tasks first
- Verify current behavior matches expectations before changing anything

### While Writing Code
- Run existing tests before and after changes
- For paper reproduction, verify numbers match before claiming success
- Submit fast/reduced trials first to validate approach, then full runs

### After Writing Code
- **List what could break** and suggest tests to cover edge cases
- Run the test suite to confirm nothing regressed
- Archive (don't delete) old scripts - they may have useful patterns

### Bug Fixing
- **Start by writing a test that reproduces the bug**
- Fix the code until the test passes
- Keep the test to prevent regression

### Learning From Mistakes
- **When corrected, add a new rule to this file** so the mistake never happens again
- Document gotchas and edge cases discovered during debugging

### Session Continuity
- Check `DEVELOPMENT.md` changelog for recent work
- Check running SLURM jobs: `squeue -u ronb`
- Check `results/*.csv` for computed values
- The development log below tracks session-to-session context

---

## Bash Guidelines

### IMPORTANT: Avoid commands that cause output buffering issues
- DO NOT pipe through `head`, `tail`, `less`, or `more` when monitoring
- Use command-specific flags: `git log -n 10` not `git log | head -10`
- For log files, read directly rather than piping through filters

### IMPORTANT: Use $HOME2 for storage, not $HOME
- `$HOME` (/home/ronb) has limited quota - builds will fail
- `$HOME2` (/groups/doudna/projects/ronb/) has 2 PB storage
- Set: `APPTAINER_CACHEDIR=$HOME2/.apptainer_cache`
- Set: `PIP_CACHE_DIR=$HOME2/.pip_cache`

### IMPORTANT: Use SLURM for GPU or heavy CPU tasks
- NEVER run GPU code on login nodes - submit to SLURM
- Partitions: `standard` (CPU), `gpu` (GPU), `memory` (high-mem)
- Always use `eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"` in SLURM
- Example scripts: `scripts/slurm_*.sh`

---

## Project-Specific Guidelines

### Paper Reference
- **Title**: "Functional protein mining with conformal guarantees"
- **Journal**: Nature Communications (2025) 16:85
- **DOI**: https://doi.org/10.1038/s41467-024-55676-y

### Verified Paper Claims ✅
| Claim | Paper Value | Verified Value |
|-------|-------------|----------------|
| Syn3.0 annotation (α=0.1) | 39.6% (59/149) | 39.6% (59/149) |
| FDR threshold (α=0.1) | 0.9999802250 | 0.9999801 |
| DALI TPR | 82.8% | 81.8% |
| DALI DB reduction | 31.5% | 31.5% |
| CLEAN loss ≤ α | 1.0 | 0.97 |

### Core Algorithms (in `protein_conformal/util.py`)
- `get_thresh_FDR()` / `get_thresh_new_FDR()` - FDR threshold
- `get_thresh_new()` - FNR threshold
- `simplifed_venn_abers_prediction()` - Calibrated probabilities
- `scope_hierarchical_loss()` - Hierarchical loss
- `load_database()` / `query()` - FAISS operations

### ⚠️ Data Leakage Warning
**DO NOT USE** `conformal_pfam_with_lookup_dataset.npy` from backup directories.
**USE** `pfam_new_proteins.npy` from Zenodo - produces correct threshold.

---

## Key Files Reference

### CLI
- `protein_conformal/cli.py` - Main CLI (`cpr embed`, `cpr search`, `cpr verify`)

### Threshold Computation
- `scripts/compute_fdr_table.py` - FDR thresholds (use `--partial` for partial match)
- `scripts/compute_fnr_table.py` - FNR thresholds
- `scripts/slurm_compute_fdr_thresholds.sh` - SLURM wrapper
- `scripts/slurm_compute_fnr_thresholds.sh` - SLURM wrapper

### Verification
- `scripts/verify_syn30.py` - JCVI Syn3.0 (Figure 2A)
- `scripts/verify_dali.py` - DALI prefiltering (Tables 4-6)
- `scripts/verify_clean.py` - CLEAN enzyme (Tables 1-2)

### Results
- `results/fdr_thresholds.csv` - FDR thresholds with stats
- `results/fnr_thresholds.csv` - FNR exact match thresholds
- `results/fnr_thresholds_partial.csv` - FNR partial match thresholds
- `results/dali_thresholds.csv` - DALI prefiltering results

### Documentation
- `GETTING_STARTED.md` - User quick-start (most important)
- `DEVELOPMENT.md` - Dev status and changelog
- `DATA.md` - Data file documentation
- `REPO_ORGANIZATION.md` - Paper figures → code mapping

---

## Development Log

### 2026-06-25 - GPU embed latency: memory snapshot + fp16 + volume prebuild

Prior round (cold-start FAISS / UI payload / LRU) only helped a little because a
*fresh* search is dominated by GPU embedding (ProtTrans T5-XL load + forward), which
it didn't touch. User declined warm GPU (cost); pursued cost-free + validation-gated levers.

**Done (all on PREVIEW app, prod untouched):**
- **Modal memory snapshot on Embedder** (modal_app.py): `enable_memory_snapshot=True`;
  split `@modal.enter` into snap=True (load ProtTrans+Protein-Vec to CPU, `map_location='cpu'`)
  + snap=False (move to GPU). Cold restores skip the ~3B-param model load. Cost-free.
- **fp16 autocast scoped to ProtTrans T5** in Embedder.embed (Protein-Vec head stays fp32:
  its fp16 attention hits a cuDNN SDPA "no execution plan" error; `pt.float()` before embed_vec).
  VALIDATED on GPU (scripts/verify_fp16.py, SLURM job 1141782): cosine fp32-vs-fp16 = 1.000000,
  Syn3.0 = 59/149 for both fp32 and fp16 (matches paper), identical hit set, 100% identical Pfam.
- **StageTimer surfaced**: `logging.basicConfig(INFO, force=True)` in `ui()` so per-stage
  durations show in Modal logs.
- **UI Stop toggle** (gradio_interface.py): Search button swaps to a red "⏹ Stop" in the
  same spot only while running (visibility toggle via _searching/_idle), reverting on
  completion. `stop_btn.click(_idle, cancels=[search_event])` cancels the in-flight search
  (UI-level: frees the worker; the GPU spawn may still finish). No extra real estate.
- **Input cap** (gradio_interface.py + util.query_count_error): MAX_QUERY_SEQUENCES=5000;
  over the cap returns an error routing genome-scale jobs to the CLI (`cpr search`, checkpointed).
  Bumped GPU_TIMEOUT 300->600s for headroom at the cap on A10G. 5000 ~ bacterial genome.
- **GPU bench scripts**: scripts/modal_bench_gpu.py (T4 vs A10G, 149 Syn3.0) +
  modal_bench_scaling.py (A10G at N=150/1000/5000). Embedding is linear in N, so genome-scale
  time extrapolates from the per-seq rate. (`modal run`; ephemeral apps, output buffered.)
- **CLI "fast mode"** (cli.py): `--fast` flag on `cpr embed` and `cpr search` -> fp16 via
  `_should_use_fp16(fast, device)` (gated to CUDA) threaded into `_embed_protein_vec` with the
  same validated autocast pattern. So the package (not just the web app) can run fp16.
- **Measured (preview StageTimer):** warm search ~1s (protein_vec_embedding 0.38-0.91s +
  faiss 0.10s); repeat query ~0s (LRU); first/cold query ~32s (all in the embedding stage =
  GPU container spin-up + snapshot restore + CPU->GPU transfer; snapshot confirmed restoring).
- **Syn3.0 (149 queries) slow = embedding, NOT batchable.** StageTimer: protein_vec_embedding
  52s (incl. cold start; ~20s warm on T4), database_search 1.65s, rest ~0. Batching RULED OUT:
  verify_batch.py (GPU job 1141906) shows batched embedding is bit-identical (cosine 1.0, 59/149)
  but SLOWER (0.68x) -- padding short seqs to the batch max wastes more than it saves, and
  embed_vec stays per-seq. featurize_prottrans (utils_search.py:52) already runs T5 on the batch
  but keeps only features[0], so per-seq calls = N full T5 forwards. Real lever: GPU speed --
  same 149 embeds took 5.3s on A5000 vs ~20s T4. **Switched Embedder T4 -> A10G** (~4x faster,
  scales to zero so ~cost-neutral per search). scripts/verify_batch.py + slurm_verify_batch.sh.
- **FAISS sidecars prebuilt on cpr-data volume** (scripts/modal_prebuild_faiss.py via `modal run`):
  lookup 540K, AFDB 2.3M, Euk 74K, SCOPe. Cold-start index build now skipped.

**Verify harness:** scripts/verify_fp16.py + slurm_verify_fp16.sh (partition=gpu; conda
`conformal-s` has the full stack: torch cu124 + transformers + faiss + pytorch_lightning).
ProtTrans cached at /groups/doudna/projects/ronb/huggingface_cache (set HF_HOME + HF_HUB_OFFLINE=1).

**Deploy state:** PREVIEW app `cpr-gradio-preview` (https://doudna-lab--cpr-gradio-preview-ui.modal.run)
via `modal deploy --name`. PROD (`cpr-gradio`) UNTOUCHED. Preview needs CLEAN binaries symlinked
from main checkout (worktree gitignores them). Cleanup when done: `modal app stop cpr-gradio-preview`.

**Cluster gotchas:** `*.modal.run` DNS is flaky from login nodes (input-plane invocation +
web URL fail intermittently); the control plane (deploy/run/logs) works. So the app can't be
driven from the cluster — user tests from their machine. Background monitors get killed between turns.

### 2026-06-24 - Performance: prebuilt FAISS index + UI payload cap

**Context:** Embedding + site felt slow. Got a Codex diagnosis, then implemented
the zero-cost wins (user chose NO warm containers, so no `min_containers` changes —
cold start is shrunk, not eliminated). Plan: `docs/plans/2026-06-24-perf-optimization.md`.

**Completed (TDD, all in worktree `claude/sweet-gauss-646781`):**
- **Prebuilt FAISS index (biggest cold-start win).** New `util.py` functions:
  `build_index` (non-mutating normalized IndexFlatIP), `lookup_index_path` (sidecar
  path: `*.npy` -> `*.faissindex`), `load_or_build_index` (read if present else
  build+write), `load_lookup_index` (prefers sidecar; **skips the ~1 GB np.load**
  when present). Exact search preserved — round-trip test proves read-back results
  are bit-identical, so conformal guarantees unaffected.
- **Offline builder** `scripts/prebuild_faiss.py` — idempotent, builds sidecars for
  the 4 cosine DBs (Swiss-Prot/SCOPe/AFDB/Euk). NOT CLEAN (IndexFlatL2, separate path).
- **Gradio `get_lookup_resources`** now uses `load_lookup_index` (dropped the
  `embeddings.copy()`). Falls back to building from `.npy` if no sidecar (safe to
  deploy before prebuilding).
- **Display cap** `cap_matches_per_query` — table shows top 200/query; full match
  set stays in `session["results"]["matches"]` so **download returns everything**.
  Added a `display_note` in the summary. Constant: `DISPLAY_ROWS_PER_QUERY`.
- **Cache-key fix** `sequences_hash` — replaced collision-prone `"".join(sequences)`
  (`["AB","CD"]` == `["ABC","D"]`) with a `\x00`-separated hash.
- **Process-level embedding LRU** `util.LRUCache` + `gi.EMBEDDING_CACHE` (cap 256,
  keyed `protein_vec:<seqhash>`) — common queries skip the GPU round-trip across
  sessions/users. test_session_isolation has an autouse fixture that clears it
  (a leftover entry would skip the barrier-mock embed and deadlock the concurrency test).
- **Singleflight index build** `gi.LOOKUP_BUILD_LOCKS` + `_get_lookup_build_lock` —
  concurrent cold-start misses on the same DB now build once (per-key lock +
  double-checked cache), instead of each loading ~1 GB and rebuilding.

**Tests:** new — `tests/test_util.py` TestFAISSPrebuild (8), TestDisplayCap (4),
TestSequencesHash (3), TestLRUCache (4); `tests/test_lookup_resources.py` (3, incl.
reading prebuilt index with `.npy` deleted + threaded singleflight build-once);
`tests/test_session_isolation.py` (+1 embedding-cache reuse).

**DEPLOYMENT STEP REQUIRED for the speedup:** run `prebuild_faiss.py` against the
Modal volume data and commit, so the `.faissindex` sidecars exist in prod. Without
them the code still works (builds from `.npy`), just no cold-start win.

**Deferred / needs decision (each blocked for a real reason, not skipped lazily):**
- ANN index (HNSW/IVF) — gated on validating recall vs verified numbers
  (verify_syn30/verify_dali + threshold tables). Recall risk to conformal guarantees.
  Needs a compute-node run against the real DBs.
- Group 1 Modal GPU: embedding batching + pickled-bytes return — real embedding
  speedups but in `modal_app.py`; batching correctness depends on `utils_search`
  padding behavior (not in repo, on the volume) and needs a `modal serve` + GPU run
  to verify. Not safe to ship blind (wrong embeddings would silently corrupt search).
- Skipped as low-value: Protein-Vec length guard (a warning, not a speedup;
  truncating would risk reproducibility); plot-data caching (plotly rebuild is already
  ms-range); self-host fonts (cosmetic, changes typography — a design choice).

**Lessons:** `docs/plans/2026-06-24-worktree-import-lessons.md` (worktree scripts
import the installed package not the worktree; `conda run python -` ignores heredoc stdin).

### 2026-02-19 - Euk Database + Concurrency + CLEAN Fixes

**Completed:**
- Added "Euk (74K)" database from Nomburg et al. ("Birth of novel protein folds in the virome")
- Parsed `euk.fasta` (74,129 sequences) into `data/euk/euk_metadata.tsv` with Entry, Protein names, Organism, Sequence columns
- Moved embeddings to `data/euk/euk_embeddings_protein_vec.npy` (shape 74129x512)
- Added Euk as radio option in Gradio UI (4 touch points in `gradio_interface.py`)
- Added euk files to `_check_volume_data()` optional list in `modal_app.py`
- Uploaded both files to Modal volume `cpr-data`
- Deployed to production — Euk option visible at production URL
- Fixed CLEAN UI text: removed vague "family" term, added precise loss levels (e.g., "loss ≤ 1" = same sub-subclass, 4th digit may differ)
- Added GPU embedding timeouts (300s) using `.spawn()` + `.get(timeout=)` — hangs now fail loudly instead of blocking forever
- Increased Gradio queue concurrency: `default_concurrency_limit=5` so multiple users aren't serialized

**CLEAN lookup database:** 5,242 EC centroids (ESM-1b + LayerNormNet 128-d embeddings), searched via FAISS L2 distance. Thresholds pre-computed from 20 calibration trials per α level. Empirical test losses match targets (α=1.0 → loss=0.93, α=2.0 → loss=1.98, α=3.0 → loss=2.95).

**CLEAN bugs found and fixed:**
1. `threshold_mean` column was renamed to `lambda_threshold` by the CSV alias map (designed for FDR/FNR CSVs). Fixed CLEAN lookup to use `lambda_threshold`.
2. FAISS `IndexFlatL2` returns **squared** L2 distances, but calibration thresholds use regular L2. Added `D = np.sqrt(D)` after FAISS search. Verified: adenylate kinase → EC 2.7.4.3, L2 dist=5.98, threshold=7.18 at α=1.0.

**Verified on dev (`modal serve`) before deploying:**
- CLEAN: adenylate kinase → 1 match (EC 2.7.4.3, correct)
- Protein Search: hemoglobin alpha → 909 matches (top hit P69905 HBA_HUMAN, correct)

**Open issues:**
- **`CURRENT_SESSION` global dict** — Module-level mutable state shared across all users. Causes data races (user A's results overwritten by user B). Proper fix: refactor to `gr.State()` (per-session). This is a ~20-touch-point refactor across `gradio_interface.py`. Concurrency limit increase mitigates the blocking symptom but not the data race.

**Branch:** `gradio`

---

### 2026-02-18 - Deploy Fix + AFDB + CLEAN

**Completed:**
- Fixed baked file path conflict: CLEAN centroid files moved from `/app/data/clean/` to `/app/bundled/clean/` (was creating real dir that blocked volume symlink)
- Fixed Gradio 6 dropdown validation: converted float choices to strings (Gradio 6.5.1 strict type comparison)
- Deployed AFDB + CLEAN modes to production
- All 56 tests passing
- Production URL: `https://doudna-lab--cpr-gradio-ui.modal.run`

**Architecture (3 Modal containers):**
- `Embedder` — ProtTrans T5 + Protein-Vec on T4 GPU
- `CLEANEmbedder` — ESM-1b + LayerNormNet on A10G GPU
- `ui` — Gradio on CPU, calls `.remote()` to GPU containers

**Branch:** `gradio`

---

### 2026-02-11 - Gradio UI/UX Overhaul (Phase A + B)

**Completed:**
- Phase A: Mounted partial threshold CSVs in Modal, clamped FDR threshold to [0,1], renamed "UniProt" → "Swiss-Prot (540K)"
- Phase B: Per-query dropdown filter, dual exact/partial Pfam probabilities, Probability Filter mode, hide uncharacterized checkbox, Syn3.0 full 149-protein FASTA, embedding + FAISS result caching, probability vs. rank line plot, "Save Embeddings (.npy)" button, detail panel for full sequences/metadata
- Fixed: float32 artifact where `sims.max()` = 1.000000238 (clamped with `min(lhat, 1.0)`)
- Fixed: `raw_matches` now calibrated before caching so probability plot has real data
- Fixed: `filter_by_query` TypeError (wrong args to `_build_prob_plot`)
- Removed α=0.001 from dropdown (threshold always hits >1.0)

**Open issue — Table cell clipping (see HANDOFF.md):**
- User wants Google Sheets-like cell behavior: clipped text, click to expand in-place
- `wrap=False` doesn't constrain column width in Gradio 5 — cells expand to fit content
- CSS `table-layout: fixed` didn't work — Gradio 5 scoped styles override external CSS
- Current workaround: explicit string truncation with "…" (readable but loses click-to-expand)
- **Next step**: Try JavaScript via `gr.Blocks(js=...)` for inline styles (highest specificity)
- See `HANDOFF.md` for full details and untried approaches

**Branch:** `gradio` (all changes uncommitted)
**Deploy:** `modal deploy modal_app.py`

---

### 2026-02-03 - Cleanup & Consolidation

**Completed:**
- Archived 16 redundant scripts to `scripts/archive/`
- Archived duplicate Python files from `notebooks/pfam/`
- Consolidated threshold CSVs (removed "simple" versions)
- Added full threshold tables to `GETTING_STARTED.md`
- Merged `SESSION_SUMMARY.md` into `DEVELOPMENT.md`
- Archived outdated `docs/QUICKSTART.md`
- Updated this file with working patterns

**FDR Job Status:**
- Job 1012664 (fdr-fast): 20 trials, α=0.1 verified as 0.99998006

**Final Structure:**
- 4 SLURM scripts (build, embed, fdr, fnr)
- 4 results CSVs (fdr, fnr, fnr_partial, dali)
- 51 tests passing

---

### 2026-02-02 - Verification & CLI

**Completed:**
- Verified Syn3.0: 59/149 = 39.6% ✅
- Fixed FDR bug (1D/2D array handling)
- Created CLI with `embed`, `search`, `verify` commands
- Created verification scripts for DALI, CLEAN
- Investigated data leakage in backup dataset

**Environment:**
- Conda: `conformal-s` (Python 3.11.10)
- Packages: faiss 1.9.0, torch 2.5.0, numpy 1.26.4

---

### 2026-01-28 - Initial Session

- Removed duplicate `src/protein_conformal/`
- Created `pyproject.toml` and test infrastructure
- Created initial documentation

---

## Best Practices

### Testing
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_util.py -v        # Just util tests
pytest tests/test_cli.py -v         # Just CLI tests
```

### Git Workflow
- Work on feature branches, not main
- Run tests before committing
- Use descriptive commits referencing paper figures/tables

### SLURM Jobs
```bash
squeue -u ronb                      # Check running jobs
cat logs/job_*.log | tail -20       # Check recent output (use Read tool)
scancel JOBID                       # Cancel a job
```

### Code Style
- Follow patterns in `protein_conformal/util.py`
- Use numpy for numerical operations
- Use FAISS for similarity search
- Notebooks for analysis, package for algorithms

### Gradio 6 Gotchas
- **`wrap=False` does NOT clip text** — it only sets `white-space: nowrap`. Columns still expand to fit content. You MUST also constrain column width for clipping to work.
- **External CSS often doesn't work** — Gradio uses scoped Svelte styles that override external CSS even with `!important`. Use `gr.Blocks(js=...)` with inline styles via JavaScript instead.
- **`interactive=True` on Dataframe** — clicking a cell enters edit mode, showing full content. This IS the click-to-expand behavior if the initial display is clipped.
- **Dropdown choices must be strings** — Gradio 6 does strict type comparison in dropdown preprocess. Use `["0.1", "0.2"]` not `[0.1, 0.2]`, and `float()` in the handler.
- **Test on compute nodes** — `pytest tests/` takes ~5 min, prefer SLURM `standard` partition.
- **Gradio version** — Modal currently installs Gradio 6.5.1. Check specific version behavior before assuming CSS/JS patterns work.

### Modal Deployment
- `modal deploy modal_app.py` — takes ~2s, deploys Gradio + GPU embedder
- `modal serve modal_app.py` — local testing with live logs (use for debugging)
- Container idle timeout: 20 min (`scaledown_window=60*20`)
- First request after cold start: loads ~1GB lookup embeddings from volume → builds FAISS index → slow. Subsequent requests within same container are fast (in-memory cache).
- Baked files: `results/*.csv`, `protein_conformal/`, `data/gene_unknown/unknown_aa_seqs.fasta` (at `/app/bundled/syn30.fasta`), `data/clean/ec_centroid_*` (at `/app/bundled/clean/`)
- **NEVER bake files to `/app/data/`** — it creates a real directory that blocks the `/app/data -> /vol/data` symlink. Always bake to `/app/bundled/` and copy at startup.
- Volume (`/vol`): lookup embeddings, metadata TSV, Protein-Vec models, HF cache, AFDB data
- **CLEAN_repo dependency** — `CLEAN_repo/app/data/pretrained/split100.pth` must exist at deploy time (baked into CLEANEmbedder image)
