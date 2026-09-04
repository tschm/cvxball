# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

`cvxball` computes the smallest enclosing sphere (minimum enclosing ball) of a
set of points. The library lives in `src/cvxball/` and ships two solvers, which
answer the same question from opposite sides and agree on the answer:

- `min_circle_active_set` (`src/cvxball/solver.py`) — the default. An active-set
  QP method on the dual (a QP over the unit simplex), maintaining the support set
  of points on the ball's boundary. It terminates at an exact vertex rather than
  an interior-point tolerance, and scales far better in the number of points.
  Dual-feasible throughout, so its ball encloses nothing until the last iteration
  and its radius rises to the answer from below.
- `min_circle_fgk` (`src/cvxball/fischer_gaertner_kutz.py`) — the
  Fischer–Gärtner–Kutz pivoting method (ESA 2003): primal-feasible throughout,
  deflating an enclosing ball by walking the centre towards the circumcentre of an
  affinely independent support set. Implements the algorithm of the paper's Fig. 2
  including both pivot rules (`pivot_rule="bland"` for the one Theorem 1 proves
  terminating, `"heuristic"` for the faster one the paper's own code uses) and
  section 4's dynamic QR: its own `_Frame` carries `Q` and `R` for the edge matrix
  across pivots and repairs them in `O(dr)`, with `dynamic_qr=False` selecting the
  rebuild as the baseline that says what the data structure is worth.
  `ball_with_counts` is its fuller signature, returning the support set and the
  pivot counts beside the ball.

The active-set method is the default because it is the faster of the two on every
cloud measured (Table 3 of the note: within a factor of 1.1 to 1.6 from `d = 1000`
to `d = 16000`) and because its weights are a certificate the caller can check in
one pass. Both are public: `from cvxball import min_circle_active_set,
min_circle_fgk, ball_with_counts`. Both are held to the same expectations in
`tests/test_solver.py`, whose shared cases are parametrised over the solvers.

**NumPy and SciPy are the runtime dependencies, and clarabel is not.** SciPy
earns its place through exactly one thing: `src/cvxball/_frame.py` uses
`qr_insert`, `qr_delete` and `qr_update` — compiled Givens updates — to repair
the support's factorisation in `O(dr)` instead of rebuilding it in `O(dr^2)`.
That is worth 3.5x at `d = 8000`, nothing below `d ~ 100`, and a *loss* below
`d ~ 50`, which is why `min_circle_active_set` dispatches on dimension
(`_MAINTAIN_MIN_DIM`) rather than always taking that route. The pivoting solver
uses the same three routines for the same reason, plus
`scipy.linalg.solve_triangular`, so it adds no import of its own. Do not widen the
dependency set further without the same kind of measurement: `clarabel` stays in
the `dev` group because the cone program it serves lives in `experiments/` and is
a reference, not a solver this ships. `make deps` (deptry over `src`) is what
catches an undeclared or unused one.

Two *reference* implementations of the same problem live in `experiments/`,
which is not installed with the package:

- `experiments/clarabel_ball.py` — the second-order-cone program assembled by
  hand and handed to Clarabel. This used to be `min_circle_clarabel` in
  `cvxball.solver`; it moved when clarabel became dev-only.
- `experiments/welzl.py` — Welzl's randomised incremental algorithm, recursing on
  the boundary set so recursion depth stays at `d + 2`.

The pivoting method used to sit beside them and no longer does: it is a shipped
solver, so it lives in `src/` and is held to the src gates (100% docstrings, `ty`,
`mypy --strict`, bandit, the coverage gate). Both remaining references exist to be
compared against, and `tests/test_solver.py` imports the Clarabel one so CI keeps
checking that three independent implementations agree — that import is why a root
`conftest.py` exists (`pytest.ini` is template-owned and carries no `pythonpath`).
Welzl's method is *not* under test; it is reached only by the benchmark.
`experiments/bench_seb.py` produces the tables in `docs/paper/seb.tex`.

Both solvers are written to be scale- and origin-invariant, and for the
active-set code the two properties are load-bearing and easy to break:
it is **scale-invariant** (no tolerance is tied to coordinates of magnitude 1 —
the affine-rank test runs on edge vectors, the feasibility slack is sized off the
cloud's extent, the null-space direction is normalised before it meets a weight
tolerance, and the centred cloud is rescaled to unit extent by an exact *power of
two*, which keeps every squared quantity inside the exponent range that squaring
halves while moving no bits, so representable answers stay bit-exact) and it is **origin-invariant** (it recentres the cloud first,
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

- `src/cvxball/` — the library source: `solver.py` (the active-set method),
  `_frame.py` (its maintained factorisation), `fischer_gaertner_kutz.py` (the
  pivoting method, which moved here from `experiments/` when it became a shipped
  solver) and `__init__.py` (the public surface: both solvers plus
  `ball_with_counts` and `Ball`).
- `tests/test_solver.py` — the project's own test suite. Note `tests/` is *not*
  wholly local: `tests/test_rhiza_packaging.py` is synced.
- `conftest.py` — puts the repo root on `sys.path` so the suite can import
  `experiments/`. Not template-owned, and the only place this can live: the
  usual `pythonpath = .` would go in the template-owned `pytest.ini`.
- `experiments/` — reference implementations and benchmarks. Outside `packages`
  and `testpaths`, so the coverage, docstring and type gates do not reach it;
  `ruff` and `ruff-format` do.
- `docs/paper/seb.tex` — the six-page note, with the benchmark tables. It sits
  in the docs tree because that is where rhiza's `paper` task looks: `make paper`
  compiles the root document of `docs/paper/` with tectonic (rerunning until
  cross-references converge) and leaves the PDF beside the source, which is what
  lets `make book` publish it as a site asset with no copy step. `make
  paper-clean` removes it again. `seb.tex` is the only `.tex` at that folder's
  top level, so it is unambiguously the root document; adding a second one there
  would hand the compile to whichever sorts first unless one is named `main.tex`.
  Both the PDF and the LaTeX aux files are gitignored by the template.
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
  The `[[tool.mypy.overrides]]` block is back, naming `scipy.*` only: it had gone
  when the cone program moved to `experiments/` and left `src` importing nothing
  untyped, and returned with `src/cvxball/_frame.py`. `ty` needs its own
  suppression for the same imports — a per-name `# ty: ignore[unresolved-import]`
  on `qr_insert`, `qr_delete` and `qr_update`, which are re-exported from a Cython
  extension and so are absent from the stubs while present at run time. The
  directive has to sit on each imported name; on the opening parenthesis it is
  reported as unused *and* the errors still fire.

There is no `make validate`, and `make deptry` is gone (the target is `deps`).
Template drift is caught by the `check-managed-files` pre-commit hook instead.

The project test suite covers `src/cvxball/` at 100%, statements and branches
both, well above the 90% gate. Holding that costs something worth knowing: the
last lines to fall were the rebuild-on-refusal paths in both factorisations —
`_frame.py`'s and the pivoting solver's `_Frame` — and no input reaches them, since
the stability thresholds exist precisely to make a refused `qr_insert`,
`qr_delete` or `qr_update` rare. They are reached by patching the routine to raise
(`test_maintained_face_falls_back_when_an_update_is_refused`,
`test_fgk_frame_falls_back_when_an_update_is_refused`), which is the idiom to
follow for the next such path rather than lowering the bar. CI runs
it on ubuntu, macOS and Windows across Python 3.11–3.14; the OS list comes from
`ci-os-matrix` in `[tool.rhiza-task]`, without which it would default to
ubuntu alone.
