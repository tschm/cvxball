"""Core package for minimum enclosing ball utilities and solvers.

Exposes the version and the solver, so the whole public surface is reachable as
``from cvxball import min_circle_active_set``. The submodule path
``cvxball.solver`` keeps working, so this is additive -- but the short form is
the documented one, which leaves the module layout free to change without
breaking callers.

The one dependency is NumPy, which the re-export pulls in on import. That is the
intended trade: it is the package's only reason to exist, so an ``import
cvxball`` that did not pull it in would be deferring work every caller is about
to need. Nothing else is imported, because nothing else is needed -- the
Clarabel cone program that used to ship alongside this method now lives in
``experiments/clarabel_ball.py``, where it serves as the reference the method is
measured against, and Clarabel is a development dependency.
"""

import importlib.metadata

from cvxball.solver import min_circle_active_set

__version__ = importlib.metadata.version("cvxball")

__all__ = ["__version__", "min_circle_active_set"]
