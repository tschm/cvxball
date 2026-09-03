# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

`cvxball` computes the smallest enclosing sphere (minimum enclosing ball) of a
set of points. The library lives in `src/cvxball/` and exposes two solvers that
share one interface — `(points, verbose=False) -> (radius, center)`, so they are
interchangeable and the test suite parametrises the shared cases over both:

- `min_circle_clarabel` — assembles the second-order-cone program directly and
  calls Clarabel, with no modelling-layer canonicalisation in between.
- `min_circle_active_set` — an active-set QP method on the dual (a QP over the
  unit simplex), maintaining the support set of points on the ball's boundary.
  Pure NumPy. It terminates at an exact vertex rather than an interior-point
  tolerance, and scales far better in the number of points.

Since then the package has grown a second, more general front end.
`src/cvxball/active_set.py` exposes `DefaultSolver(P, q, A, b, cones, settings)`
and `.solve()` — Clarabel's own constructor and method, with Clarabel's standard
form, solution attributes and `SolverStatus` values (re-exported from it) — so it
can be *substituted* for the `clarabel` module. It dispatches on the cones:
`ZeroConeT`/`NonnegativeConeT` blocks go to `src/cvxball/qp.py`, a dense dual
active-set method for convex QPs (Goldfarb–Idnani), and equally sized
`SecondOrderConeT` blocks in the enclosing-ball shape go to the support-set
method in `solver.py`; anything else is refused at construction. The QP core
needs `P` positive definite, or made so by the `static_regularization_*`
settings, and refuses an indefinite `P` outright — regularising until an
indefinite matrix factorises would answer a convexified question instead.

Three seams there are worth knowing before changing anything:

- `_ball_points` rebuilds the ball program from the recovered points with
  `_build_soc_program` and compares, rather than trusting the shape of the cone
  list. A program with the right cones but a different matrix must be refused,
  not answered.
- `solver._active_set_ball` is the support-set method with nothing discarded (the
  dual weights and the pivot count as well as the radius and centre);
  `min_circle_active_set` is a two-line wrapper over it. The weights *are* the
  conic dual, arranged as `z_i = u_i (1, -(p_i - x)/R)` per cone block, which is
  what lets the front end return a certificate rather than just a primal.
- `solver._ActiveSetError` carries a `status` name alongside the message. The
  public function lets it propagate as the `ValueError` its callers always saw;
  the front end catches it and reports `MaxIterations`/`NumericalError`, because
  the same event has to be an exception in one API and a status in the other.

`qp.solve_qp` is the only block near the cyclomatic-complexity ceiling (`make
complexity`, ceiling 15) — it sat at 17 before the direction, candidate-selection
and starting-point helpers were extracted, so keep new branches out of its loop.

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
- `tests/test_solver.py`, `tests/test_qp.py`, `tests/test_active_set.py` — the
  project's own test suite. Note `tests/` is *not* wholly local:
  `tests/test_rhiza_packaging.py` is synced.
- `pyproject.toml` — project metadata, dependencies, and local tool config
  (`[tool.deptry]`, `[[tool.mypy.overrides]]`, `[tool.rhiza-task]`,
  `[tool.bumpversion]`).
- `README.md` and any project-specific documentation.
- `.github/workflows/release.yml` and `.github/workflows/audit.yml` — the repo's
  own workflows. Everything else under `.github/workflows/` is synced, so these
  two are deliberately *not* named `rhiza_*`.
- `.rhiza/template.yml` — selects the template version (`template-branch`) and
  platform profile (`profiles`). This file is *configuration you own*, even
  though it lives under `.rhiza/`.

### Rhiza-owned (do not edit in place — change upstream and re-sync)

Exactly the 28 paths in `.rhiza/template.lock`'s `files:` block — read that,
don't infer from a directory name. The notable ones:

- `.github/workflows/rhiza_*.yml` — CI, release, CodeQL, scorecard, benchmark,
  book, marimo, weekly.
- `Makefile` — a shim that forwards every target to a pinned `rhiza-task`.
- `.pre-commit-config.yaml`, `pytest.ini`, `ruff.toml`, `.bandit`,
  `.editorconfig`, `.python-version`, `.gitignore`, `cliff.toml`.
- `.github/rulesets/*`, `.github/dependabot.yml`, `.github/release.yml`.
- `tests/test_rhiza_packaging.py`, `.rhiza/semgrep.yml`, `docs/index.md`,
  `docs/mkdocs-base.yml`, `docs/development/rhiza.md`.

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
- `typecheck` is `ty` alone, so `[[tool.mypy.overrides]]` would otherwise be dead
  config — the `strict-types` job runs `mypy --strict src`.

There is no `make validate`, and `make deptry` is gone (the target is `deps`).
Template drift is caught by the `check-managed-files` pre-commit hook instead.

The project test suite covers `src/cvxball/` at 100%, above the 90% gate. CI runs
it on ubuntu, macOS and Windows across Python 3.11–3.14; the OS list comes from
`ci-os-matrix` in `[tool.rhiza-task]`, without which it would default to
ubuntu alone.
