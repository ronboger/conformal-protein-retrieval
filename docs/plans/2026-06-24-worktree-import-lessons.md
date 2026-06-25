# Lesson: standalone scripts import the installed package, not the worktree

**Date:** 2026-06-24

## What happened
While building `scripts/prebuild_faiss.py` in a git worktree
(`.claude/worktrees/sweet-gauss-646781/`), running it as
`python scripts/prebuild_faiss.py` failed:

```
ImportError: cannot import name 'load_or_build_index' from 'protein_conformal.util'
  (/groups/doudna/projects/ronb/conformal-protein-retrieval/protein_conformal/util.py)
```

Note the path: the **main** checkout, not the worktree — even though the new
function existed in the worktree's `util.py`.

## Root cause
- `pytest` inserts the rootdir (the worktree) onto `sys.path`, so tests import the
  worktree's `protein_conformal` and saw the new functions — tests passed.
- A plain `python scripts/foo.py` puts the **script's** directory (`scripts/`) on
  `sys.path[0]`, not the repo root. `import protein_conformal` then resolves via
  the editable install / whatever is on the path, which pointed at the main
  checkout's copy without the new code.
- `python -c "..."` worked because `sys.path[0]` is `''` (cwd = worktree root),
  so it found `./protein_conformal`.

## Fix
Make repo scripts insert their own repo root before importing the package:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protein_conformal.util import ...
```

This makes the script use the `protein_conformal` that lives next to it (same
checkout), regardless of editable-install target. Also works on Modal
(`/app/scripts/..` -> `/app`).

## Gotcha for verifying scripts in a worktree
To run a worktree script against worktree code without the sys.path shim, either
run from the worktree root with the root on PYTHONPATH:
`PYTHONPATH=$(pwd) python scripts/foo.py`, or `python -m scripts.foo`.

## Unrelated gotcha hit same session
`conda run -n <env> python - <<'PY' ... PY` does **not** forward the heredoc to
the subprocess's stdin — `python -` reads empty stdin and runs nothing (exits 0,
no output). Write the snippet to a temp `.py` file and run that instead.
