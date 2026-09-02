"""Core package for minimum enclosing ball utilities and solvers.

Exposes the version and the solver itself, so the whole public surface is
reachable as ``from cvxball import min_circle_clarabel``. The submodule path
``cvxball.solver.min_circle_clarabel`` keeps working, so this is additive --
but the short form is the documented one, which leaves the module layout free
to change without breaking callers.

The re-export means importing this package also imports clarabel, numpy and
scipy. That is the intended trade: they are the package's only reason to exist,
so an ``import cvxball`` that did not pull them in would be deferring work every
caller is about to need.
"""

import importlib.metadata

from cvxball.solver import min_circle_clarabel

__version__ = importlib.metadata.version("cvxball")

__all__ = ["__version__", "min_circle_clarabel"]
