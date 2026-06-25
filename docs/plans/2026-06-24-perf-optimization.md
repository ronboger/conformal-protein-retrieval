# Performance Optimization Plan — Embedding + Site Latency

**Date:** 2026-06-24
**Source:** Codex diagnosis (foreground rescue), validated against code.
**Decision:** NO warm containers (`min_containers` stays 0). Zero added monthly cost.
**Constraint:** All changes free or cost-reducing. Cold start accepted, only shrunk.

## STATUS (2026-06-24)
- DONE + tested: G2.1-2.3 (prebuilt index, read_index loader, drop copy) via
  build_index/lookup_index_path/load_or_build_index/load_lookup_index +
  scripts/prebuild_faiss.py + gradio wiring. G2.4 (singleflight build lock).
  G3.1 (display cap, download-all preserved). G3.3 (process-level embedding LRU,
  util.LRUCache + EMBEDDING_CACHE). G3.4 (cache-key collision fix, sequences_hash).
- NOT DONE: G2.5 ANN (gated on recall validation), G1.1/G1.2 (needs Modal+GPU run
  to verify). Skipped low-value: G1.4 length guard, G3.2 plot caching, G3.5 fonts.
- DEPLOY: run prebuild_faiss.py on the Modal volume so sidecars exist in prod.

---

## TDD discipline
Every item gets a failing test first (per CLAUDE.md), then implementation, then
verify nothing regresses (`pytest tests/`). Paper-reproduction numbers must still
match where touched.

---

## Group 1 — Modal / GPU throughput  (`modal_app.py`)
Independent file. Can run parallel to Group 3.

1. **Batch `Embedder.embed`** (modal_app.py:202) — embed sequences in batches by
   token budget instead of one-per-loop. Big win for multi-seq inputs (Syn3.0 = 149).
   Single-query unaffected. *Impact: High (multi-seq) / Effort: Med*
2. **Return pickled ndarray bytes** instead of `.tolist()` round-trip
   (modal_app.py:218 + gpu_embed call site). *Impact: Med / Effort: Low-Med*
3. **Fail-fast on missing volume data** (modal_app.py:102 `_check_volume_data`,
   gradio_interface.py:50) — error instead of silent dataset download that looks
   like a hang. *Impact: Med / Effort: Low*

DEFERRED (not in this batch): T4→A10G swap for Protein-Vec — benchmark only; A10G
costs more per second, needs measurement, not a blind change.

## Group 2 — FAISS index pipeline  (SEQUENTIAL chain; gated)
Touches `util.py` + new `scripts/prebuild_faiss.py` + `gradio_interface.py` loader.
Ordered, not parallel. Land before Group 3 (shared file).

1. **Offline prebuild script** `scripts/prebuild_faiss.py` — normalize once, build
   index, `faiss.write_index` to volume per database. *Impact: High / Effort: Med*
2. **Loader uses `read_index`** if a prebuilt index exists, else fall back to
   current build path (get_lookup_resources, gradio_interface.py:294;
   load_database, util.py:523). *Impact: High / Effort: Med*
3. **Drop `embeddings.copy()`** (gradio_interface.py:308) — normalize offline so no
   in-place mutation, saves ~1 GB peak RAM. *Impact: Med / Effort: Low*
4. **Singleflight build lock** in get_lookup_resources so concurrent first-hits
   don't each build the index. *Impact: High under load / Effort: Low-Med*
5. **[GATED] ANN index (HNSW/IVF)** for Swiss-Prot / AFDB / Euk; keep IndexFlat for
   CLEAN (5242). *Approximate — MUST validate recall against verify_syn30.py,
   verify_dali.py, and threshold tables before enabling. Opt-in per DB; exact stays
   default for calibration-sensitive paths.* *Impact: High / Effort: Med-High + validation*

## Group 3 — Gradio UI payload  (`gradio_interface.py`)
Land after Group 2. Independent of Group 1.

1. **Cap rendered table rows** to top-N (~100) pushed to the browser Dataframe;
   keep ALL matches in session state. **Download still returns everything** —
   `export_current_results` (gradio_interface.py:1185) already reads
   `session["results"]["matches"]`, fully decoupled from display. Retrieval k
   (max_results slider, default 1000) stays user-controlled so downloads stay
   complete. *Impact: High (SSE payload) / Effort: Med*
2. **Cache probability-plot data** — recompute only on new search, refilter cheaply
   on dropdown change instead of rebuilding over full set. *Impact: Med / Effort: Low-Med*
3. **Process-level LRU embedding cache** keyed on hash(sequence)+mode — common
   queries skip the GPU round-trip across sessions. *Impact: Med / Effort: Med*
4. **Fix weak cache key** (gradio_interface.py:876) — add separators + embedding
   mode to avoid concat collisions. *Impact: Low-Med / Effort: Low*
5. **Remove / self-host Google Fonts** (gradio_interface.py:1282) — kill per-load
   external request. *Impact: Low / Effort: Low*

DEFERRED: deferred ML imports on web image (util.py imports torch/transformers at
load) — Med effort, touches import structure broadly; do after the above land.

---

## Parallelization
- Group 1 (modal_app.py) ∥ Group 3-prep — different files, safe to do concurrently.
- Group 2 is internally sequential and must precede Group 3 (shared file).
- Recommended order: G1 + G2 in parallel → G3 → (optional) ANN validation → deferred items.

## Verification
- `pytest tests/` green before + after each group.
- Re-run verify_syn30.py / verify_dali.py if Group 2.5 (ANN) is enabled.
- Benchmark cold-start + per-query latency before/after using existing StageTimer
  (gradio_interface.py:205); add request-scoped structured logging to measure.
