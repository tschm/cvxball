# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

`cvxball` computes the smallest enclosing sphere (minimum enclosing ball) of a
set of points. The library lives in `src/cvxball/` and ships exactly one solver:

- `min_circle_active_set` — an active-set QP method on the dual (a QP over the
  unit simplex), maintaining the support set of points on the ball's boundary.
  Pure NumPy. It terminates at an exact vertex rather than an interior-point
  tolerance, and scales far better in the number of points.

**NumPy is the only runtime dependency**, and that is a constraint to preserve,
not an accident: `clarabel` and `scipy` are in the `dev` group, so anything added
to `src/cvxball/` that imports either of them breaks the promise that installing
this package pulls in NumPy and nothing else. `make deps` (deptry over `src`)
catches it.

Two *reference* implementations of the same problem live in `experiments/`,
which is not installed with the package:

- `experiments/clarabel_ball.py` — the second-order-cone program assembled by
  hand and handed to Clarabel. This used to be `min_circle_clarabel` in
  `cvxball.solver`; it moved when clarabel became dev-only.
- `experiments/welzl.py` — Welzl's randomised incremental algorithm, recursing on
  the boundary set so recursion depth stays at `d + 2`.

Both exist to be compared against, and `tests/test_solver.py` imports the
Clarabel one so CI keeps checking that two independent implementations agree —
that import is why a root `conftest.py` exists (`pytest.ini` is template-owned
and carries no `pythonpath`). `experiments/bench_seb.py` produces the tables in
`docs/paper/note.tex`.

Two properties of the active-set code are load-bearing and easy to break:
it is **scale-invariant** (no tolerance is tied to coordinates of magnitude 1 —
the affine-rank test runs on edge vectors, the feasibility slack is sized off the
cloud's extent, and the null-space direction is normalised before it meets a
weight tolerance) and it is **origin-invariant** (it recentres the cloud first,
and forms squared distances by differencing before squaring). Both are covered by
tests; see `tests/test_solver.py::test_active_set_handles_tiny_scales` and
`::test_active_set_far_from_origin` for what regressions look like.

## Ownership split: locally-owned vs Rhiza-synced

This repo syncs its dev infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The pinned
template version and platform profile live in `.rhiza/template.yml`; the exact
machine-generated list of synced files is the `files:` block of
`.rhiza/template.lock`. When assessing or changing this repo, keep the two sides
distinct — fix locally-owned code here, fix synced infrastructure upstream in
Rhiza (then re-sync).

### Locally owned (change these here)

- `src/cvxball/` — the library source.
- `tests/test_solver.py` — the project's own test suite. Note `tests/` is *not*
  wholly local: `tests/test_rhiza_packaging.py` is synced.
- `conftest.py` — puts the repo root on `sys.path` so the suite can import
  `experiments/`. Not template-owned, and the only place this can live: the
  usual `pythonpath = .` would go in the template-owned `pytest.ini`.
- `experiments/` — reference implementations and benchmarks. Outside `packages`
  and `testpaths`, so the coverage, docstring and type gates do not reach it;
  `ruff` and `ruff-format` do.
- `docs/paper/note.tex` — the four-page note, with the benchmark tables. It sits
  in the docs tree because that is where rhiza's `paper` task looks: `make paper`
  compiles the root document of `docs/paper/` with tectonic (rerunning until
  cross-references converge) and leaves the PDF beside the source, which is what
  lets `make book` publish it as a site asset with no copy step. `make
  paper-clean` removes it again. `note.tex` is the only `.tex` at that folder's
  top level, so it is unambiguously the root document; adding a second one there
  would hand the compile to whichever sorts first unless one is named `main.tex`.
  Both the PDF and the LaTeX aux files are gitignored by the template.
- `paper/ball.tex` — the long form, and deliberately *not* under `docs/paper/`:
  only one root document per folder gets compiled. It has no task of its own and
  builds with `pdflatex ball.tex` run twice.
- `pyproject.toml` — project metadata, dependencies, and local tool config
  (`[tool.deptry]`, `[tool.rhiza-task]`, `[tool.bumpversion]`).
- `README.md` and any project-specific documentation.
- `.github/workflows/release.yml` and `.github/workflows/audit.yml` — the repo's
  own workflows. Everything else under `.github/workflows/` is synced, so these
  two are deliberately *not* named `rhiza_*`.
- `.rhiza/template.yml` — selects the template version (`template-branch`), the
  platform profile (`profiles`) and any extra bundles on top of it (`templates`,
  which is where `github-paper` comes from). This file is *configuration you own*, even
  though it lives under `.rhiza/`.

### Rhiza-owned (do not edit in place — change upstream and re-sync)

Exactly the 26 paths in `.rhiza/template.lock`'s `files:` block — read that,
don't infer from a directory name. The notable ones:

- `.github/workflows/rhiza_*.yml` — CI, release, CodeQL, scorecard, benchmark,
  book, marimo, weekly, paper.
- `Makefile` — a shim that forwards every target to a pinned `rhiza-task`.
- `.pre-commit-config.yaml`, `pytest.ini`, `ruff.toml`, `.bandit`,
  `.editorconfig`, `.python-version`, `.gitignore`, `cliff.toml`.
- `.github/dependabot.yml`, `.github/release.yml`, `.github/secret_scanning.yml`.
- `tests/test_rhiza_packaging.py`, `.rhiza/semgrep.yml`, `docs/index.md`,
  `docs/mkdocs-base.yml`, `docs/development/rhiza.md`.

`rhiza_paper.yml` is the newest of those and the only one that did not arrive
with the `github-project` profile — see the `templates:` block in
`.rhiza/template.yml`. It triggers only on changes under `docs/paper/**`, runs
the same `paper` task as locally but under `--strict` (so a runner without
tectonic fails rather than reporting a skip as success), and publishes the PDF
three ways: the run artifact, the book, and a dedicated `paper` branch. That
last one is why this repo must never carry a `paper/<topic>` branch: git refs
are paths, so `refs/heads/paper` cannot coexist with one, and the workflow
preflights for exactly that.

The `check-managed-files` pre-commit hook rejects a commit touching any of them.
To change Rhiza-owned behavior, open a PR against `jebel-quant/rhiza`, cut a
template release, then run `/rhiza:update` (which bumps `template-branch` and
applies the sync).

## Quality gates

Since template v1.4 the gates are tasks of a pinned `rhiza-task` CLI, not make
recipes. `uvx rhiza-task list` is the authoritative catalogue; the `Makefile` is
a shim whose catch-all forwards any target to that CLI, so bare `make <target>`
still works (and matches the allow-listed rule). Because it forwards
*everything*, `make -n` cannot tell a real gate from a typo — check the task
list rather than trusting `make -n`.

| Target | Checks |
|---|---|
| `make fmt` | pre-commit hooks via `prek` (ruff format/check, markdownlint, bandit, actionlint, interrogate, the rhiza hooks, …) |
| `make typecheck` | `ty check src` — **only `ty`**; mypy is not part of this task |
| `make docs-coverage` | interrogate over `src` and `tests` (100% required) |
| `make deps` | deptry over `src` — unused/missing/misplaced dependencies |
| `make security` | `bandit -r src` — SAST over this repo's own source, **not** a dependency scan |
| `make rhiza-test` | the template's own checks, installed as `pytest-rhiza` (pyproject structure, README, docstrings, release tags) |
| `make test` | full suite with its coverage gate (`--cov-fail-under=90`) |

Two gates are narrower than they look, and the repo covers each with a job of
its own in `.github/workflows/audit.yml`:

- `security` is bandit alone, so nothing in the rhiza gates scans dependencies
  for advisories — the `dependency-advisories` job runs `pip-audit` over
  `uv.lock`.
- `typecheck` is `ty` alone — the `strict-types` job runs `mypy --strict src`.
  There is no longer a `[[tool.mypy.overrides]]` block: it existed to silence
  `import-untyped` for `clarabel` and `scipy`, and `src` imports neither since
  the cone program moved to `experiments/`. Re-add it if anything under `src`
  ever imports an untyped package again.

There is no `make validate`, and `make deptry` is gone (the target is `deps`).
Template drift is caught by the `check-managed-files` pre-commit hook instead.

The project test suite covers `src/cvxball/` at 100%, above the 90% gate. CI runs
it on ubuntu, macOS and Windows across Python 3.11–3.14; the OS list comes from
`ci-os-matrix` in `[tool.rhiza-task]`, without which it would default to
ubuntu alone.
