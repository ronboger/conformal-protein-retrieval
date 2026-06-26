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
2. **Run the 4×-gap diagnostic from a reliable connection** (the cluster's Modal/GitHub DNS is flaky — see gotchas):
   `modal run scripts/modal_diagnose.py`
   Prints the Modal A10G runtime stack + per-stage ms/seq. Mystery: identical embed code is ~35 ms/seq on a cluster A5000 but ~100 ms/seq on Modal A10G. Likely causes (Codex): dep-stack mismatch (loose torch/cuDNN vs cluster's pinned `torch 2.1+cu121+cudnn8`) or CPU starvation (no `cpu=` set → ~1–2 vCPUs). Act on the output:
   - old/mismatched torch/cuDNN → pin the Embedder image to the cluster stack
   - `cpu_count` 1–2 or high `tokenize`/`h2d` → add `cpu=4` to the Embedder `@app.cls`
   - `t5` dominates and is high → genuinely GPU-bound → only a bigger GPU helps
   If a fix lands, re-verify with `scripts/slurm_verify_fp16.sh` (Syn3.0 must stay 59/149).
3. **Branch consolidation** (on hold by choice): `main` is an unrelated/stale history (last change 2026-02-04, 0 files unique to it). `gradio` is the real source of truth. To get one branch: force `main` = `gradio` + delete `gradio`/`claude/*`, OR repoint the GitHub default to `gradio`. Until done, `git clone` (default `main`) gets stale code without `--fast`.
4. **Paid perf (only if needed):** warm GPU pool (kills the ~32s cold start) or A100 (faster compute).

## Gotchas / context
- **Cluster → Modal/GitHub DNS is intermittent.** `*.modal.run` web URLs, `modal run`, and `git push`/`gh` to github.com fail randomly *from the cluster login node*; the Modal control plane (`modal deploy`) and `gh` API mostly work. **Run `modal run` diagnostics + git ops from your Mac.**
- **Every deploy invalidates the Embedder snapshot** (gradio_interface is in the GPU image), so the first search after any deploy is a one-time slow snapshot *creation* (~60–90s), then fast.
- **Don't re-try:** GPU memory snapshot (`enable_gpu_snapshot`) segfaults on restore (exit 139) on this config; embedding batching is bit-identical but *slower* (padding waste) — both ruled out with evidence.
- Verify scripts: `scripts/verify_fp16.py`, `verify_batch.py`, `verify_gpu_resident.py` (+ `slurm_verify_*.sh`); diagnostics: `modal_diagnose.py`, `modal_bench_*.py`.
- Full changelog: the `## Development Log` section in `CLAUDE.md`.
