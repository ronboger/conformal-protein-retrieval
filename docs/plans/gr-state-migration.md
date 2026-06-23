# Plan: Complete the `CURRENT_SESSION` → `gr.State()` migration

**Date:** 2026-06-16
**Branch:** `claude/funny-booth-01f089` (== `gradio`)
**File(s):** `protein_conformal/backend/gradio_interface.py`, `tests/test_session_isolation.py` (new), `tests/test_clean.py` (call-site updates)

## Problem

`on_submit` calls `process_input` / `process_clean_input`, which write results into the
**module-global `CURRENT_SESSION`**, then snapshot it: `session = dict(CURRENT_SESSION)`
(lines 1730, 1754). The `_cache` (embeddings + FAISS raw matches) also lives in that global.

With `default_concurrency_limit=5`, two users' handlers interleave: user B's `process_*`
overwrites the global between user A's write and A's snapshot, so **A receives B's results**.
Caching is cross-user for the same reason. This is the data race noted in the 2026-02-19 dev log.

Downstream handlers (`export_current_results`, `export_embeddings`, detail panel, query
filter, prob plot) already take the per-user `session` dict — only the search/cache path
still uses the global.

## Fix: thread `session` through the search functions (return-based contract)

Each search function takes an incoming `session: dict` (for cache reuse) and **returns** the
updated session as a 3rd value. `on_submit` stores the returned session in `gr.State`. The
global is deleted. Return-based (not in-place mutation) because Gradio `gr.State` deepcopy
semantics around input args vary by version — an explicit return is version-robust.

### Source changes (`gradio_interface.py`)

1. **Delete** global def `CURRENT_SESSION = {}` (lines 199–201) and the three `global`
   statements (607, 862, in the final build).
2. **`process_clean_input`** (485): add `session: Optional[dict] = None` param (before
   `progress`); build a local `new_session` instead of writing the global; return
   `(summary_json, df, session_out)`. Error/empty paths return the incoming `session or {}`
   unchanged (failed re-query keeps prior results).
3. **`_process_input_impl`** (785): add `session` param; `cached = (session or {}).get("_cache", {})`;
   track cache in a local `new_cache` (default = incoming cache, overwritten on fresh search at
   line 1035); final build → local `new_session` with `"_cache": new_cache`; return 3-tuple on
   every path (success returns `new_session`, error paths return `session or {}`).
4. **`process_input`** wrapper (742): add `session` param; pass it to `_process_input_impl`;
   return `(json, df, session_out)`.
5. **`on_submit`** (1721): pass `session=session` into both calls; capture the returned
   session; delete both `dict(CURRENT_SESSION)` snapshots.

### Tests (TDD — written and watched-fail first)

`tests/test_session_isolation.py` (new), mocks gradio like `test_clean.py`:

- **CLEAN path:** mock `run_embed_clean`; with centroid fixtures + thresholds, run two
  `process_clean_input` calls concurrently (threads + a `Barrier` to force interleaving) with
  distinct query headers; assert each returned session's matches reference only its own
  `query_meta`, the two sessions are distinct objects, and each carries its own `alpha`.
- **Protein-Vec path:** mock `run_embed_protein_vec`, `run_search`, and `load_results_dataframe`
  (dispatch by `required_columns` for thresholds vs calibration); run two `process_input` calls
  concurrently with distinct query headers; assert per-session isolation as above.

`tests/test_clean.py`: update the 3 existing `process_clean_input(...)` call sites to unpack
the new 3-tuple.

### Run

Worktree-aware SLURM job (`standard` partition, env `conformal-s`) — the existing
`scripts/slurm_test.sh` cd's into the **main** repo, so use a worktree-pinned invocation.
RED run (new tests only) → watch fail; implement; GREEN run (new tests); then full suite.

## Risks / what could break

- Adding a 3rd return value breaks any other caller of `process_input` / `process_clean_input`.
  Grep confirms the only callers are `on_submit` and the 3 test sites.
- `_cache` semantics must be preserved (embedding reuse + FAISS reuse) — covered by keeping
  `new_cache = incoming cache` default and overwriting only on a fresh search.
- CLEAN path has no `_cache`; its session simply has no `_cache` key (handlers already use
  `.get`).

## Out of scope

- Table cell clipping (HANDOFF.md) — already implemented in commits 6639be8/50e771f/0511f9b.
- Merge `gradio` → `main`, FDR table completion, Apptainer PyTorch bump (DEVELOPMENT.md).
