"""Core package for minimum enclosing ball utilities and solvers.

Exposes the version and both solvers, so the whole public surface is reachable as
``from cvxball import min_circle_active_set, min_circle_fgk``. The submodule paths
``cvxball.solver`` and ``cvxball.fischer_gaertner_kutz`` keep working, so this is
additive -- but the short form is the documented one, which leaves the module
layout free to change without breaking callers.

The two solvers answer the same question and agree on the answer, arriving from
opposite sides: :func:`cvxball.min_circle_active_set` ascends the dual and holds
no enclosing ball until it terminates, while :func:`cvxball.min_circle_fgk`
deflates an enclosing ball and is feasible throughout. The first is the default --
faster on every row of ``experiments/bench_seb.py``, and it returns the dual
weights as a certificate; :func:`cvxball.fischer_gaertner_kutz.ball_with_counts`
is the second one's fuller signature, reporting the support set and the pivot
counts alongside the ball.

The dependencies are NumPy and SciPy, which the re-exports pull in on import.
That is the intended trade: they are the package's only reason to exist, so an
``import cvxball`` that did not pull them in would be deferring work every caller
is about to need. Nothing else is imported, because nothing else is needed -- the
Clarabel cone program and Welzl's recursion, which the two solvers are measured
against, live in ``experiments/`` and are references rather than solvers this
ships.
"""

import importlib.metadata

from cvxball.fischer_gaertner_kutz import Ball, ball_with_counts, min_circle_fgk
from cvxball.solver import min_circle_active_set

__version__ = importlib.metadata.version("cvxball")

__all__ = [
    "Ball",
    "__version__",
    "ball_with_counts",
    "min_circle_active_set",
    "min_circle_fgk",
]
