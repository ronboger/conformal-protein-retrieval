# HANDOFF — Performance + UX work

**Updated:** 2026-06-25 17:14 PDT
**Branch:** `claude/sweet-gauss-646781` (descends from `gradio`)
**PRs:** [#12](https://github.com/ronboger/conformal-protein-retrieval/pull/12) merged to `gradio`; [#13](https://github.com/ronboger/conformal-protein-retrieval/pull/13) open (OOM fix + search timer)
**Prod:** deployed and live — https://doudna-lab--cpr-gradio-ui.modal.run (Modal app `cpr-gradio`)

## Current state (what's live + tested)
All shipped to prod; 95 tests passing (`conda run -n conformal-s pytest tests/`):
- **Cold start:** prebuilt FAISS sidecar indexes on the `cpr-data` volume (skip ~1 GB np.load + build); CPU memory snapshot on the Embedder (**now `memory=32768`** — it was OOM-killing on creation, exit 137, causing ~86s reloads every cold start).
- **Embedding:** fp16 autocast on ProtTrans T5 (head stays fp32); GPU-resident embed (no per-seq GPU↔CPU round-trip); A10G; fan-out across containers for >600 seqs (genome-scale). All verified bit-identical (Syn3.0 59/149, cosine 1.0).
- **Caching/payload:** process-level embedding LRU; collision-safe cache key; display cap (200/query to browser, download returns all); input cap 5000 (genome-scale → CLI).
- **UI:** Search↔Stop toggle; uploaded FASTA shown in the box; **`⏱ Search completed in X.Xs`** above the summary.
- **CLI:** `cpr embed/search --fast` (fp16 "fast mode").

## Open next steps (priority order)
1. **Merge PR #13** (the OOM fix + timer; already on prod, just not merged).
2. **[RESOLVED 2026-06-25] 4×-gap diagnostic — it's GPU-bound; both candidate fixes dropped.**
   `modal run scripts/modal_diagnose.py` (A10G, torch 2.1+cu121+cudnn8 = exact cluster stack) measured:
   `tokenize 0.7 / h2d 0.2 / t5 56.3 / head 23.5 = 80.6 ms/seq`, `os.cpu_count=17`.
   - **t5+head = 79.8 of 80.6 ms** → genuinely GPU compute-bound. The A10G has ~half the A5000's
     fp16 tensor throughput, so the same code is ~80 ms/seq here vs ~35 ms/seq on the cluster A5000.
   - **The torch pin did NOT help** — the *exact* cluster stack still measured 80.6 ms/seq, proving the
     gap was never the software stack. (It also forces `numpy<2`, else `tensor.numpy()` → "Numpy is not
     available".) **Reverted.**
   - **`cpu=4` was unnecessary** — the container already sees 17 vCPUs unset and tokenize+h2d are only
     ~0.9 ms/seq. Not CPU-starved. **Reverted.**
   - Net: `modal_app.py` is back to the pre-pin (prod `fe99921`) embedding stack; commit `137523e` reverted
     in code (comments retained documenting the finding). No deploy needed (functionally == prod). Only a
     bigger GPU (A100) would close the gap; the Protein-Vec head (23.5 ms/seq, fp32) is a secondary target.
3. **Branch consolidation** (on hold by choice): `main` is an unrelated/stale history (last change 2026-02-04, 0 files unique to it). `gradio` is the real source of truth. To get one branch: force `main` = `gradio` + delete `gradio`/`claude/*`, OR repoint the GitHub default to `gradio`. Until done, `git clone` (default `main`) gets stale code without `--fast`.
4. **Paid perf (only if needed):** warm GPU pool (kills the ~32s cold start) or A100 (faster compute).

## Gotchas / context
- **Cluster → Modal/GitHub DNS is intermittent.** `*.modal.run` web URLs, `modal run`, and `git push`/`gh` to github.com fail randomly *from the cluster login node*; the Modal control plane (`modal deploy`) and `gh` API mostly work. **Run `modal run` diagnostics + git ops from your Mac.**
- **Every deploy invalidates the Embedder snapshot** (gradio_interface is in the GPU image), so the first search after any deploy is a one-time slow snapshot *creation* (~60–90s), then fast.
- **Don't re-try:** GPU memory snapshot (`enable_gpu_snapshot`) segfaults on restore (exit 139) on this config; embedding batching is bit-identical but *slower* (padding waste); **pinning the Embedder to the cluster torch 2.1+cu121+cudnn8 stack** (still 80.6 ms/seq on A10G — it's the GPU, not the stack; also forces numpy<2); **`cpu=4` on the Embedder** (already 17 vCPUs, tokenize+h2d only ~0.9 ms/seq); **fp16 on the Protein-Vec head** (`scripts/exp_head_fp16.py`: ~1.25x on the head, no cuDNN error when the cuDNN SDPA backend is excluded, per-protein cosine 1.000000 — BUT `verify_fp16.py` job 1142272 showed it flips **Syn3.0 59→60/149**: protein #136 tips across the ~0.99998 FDR threshold on a sub-1e-6 rounding diff. cosine ~1.0 is NOT sufficient for the conformal gate; head stays fp32) — all ruled out with evidence.
- **`modal_diagnose.py` needs `gpu_image.add_local_python_source("modal_app")`** under Modal 1.x (sibling modules aren't auto-mounted), else the container import of `modal_app` fails.
- Verify scripts: `scripts/verify_fp16.py`, `verify_batch.py`, `verify_gpu_resident.py` (+ `slurm_verify_*.sh`); diagnostics: `modal_diagnose.py`, `modal_bench_*.py`.
- Full changelog: the `## Development Log` section in `CLAUDE.md`.
