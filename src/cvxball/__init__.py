"""Core package for minimum enclosing ball utilities and solvers.

Exposes the version and both solvers, so the whole public surface is reachable
as ``from cvxball import min_circle_clarabel, min_circle_active_set``. The
submodule paths under ``cvxball.solver`` keep working, so this is additive --
but the short form is the documented one, which leaves the module layout free
to change without breaking callers.

Two submodules come along for the ride, and are part of the public surface:
:mod:`cvxball.active_set` is a Clarabel-shaped solver front end -- construct it
with ``(P, q, A, b, cones, settings)`` and call ``solve()``, exactly as with
``clarabel.DefaultSolver`` -- and :mod:`cvxball.qp` is the dense active-set
method for convex quadratic programs that stands behind it.

The re-export means importing this package also imports clarabel, numpy and
scipy. That is the intended trade: they are the package's only reason to exist,
so an ``import cvxball`` that did not pull them in would be deferring work every
caller is about to need.
"""

import importlib.metadata

from cvxball import active_set, qp
from cvxball.solver import min_circle_active_set, min_circle_clarabel

__version__ = importlib.metadata.version("cvxball")

__all__ = ["__version__", "active_set", "min_circle_active_set", "min_circle_clarabel", "qp"]
