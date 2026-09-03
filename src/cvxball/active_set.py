"""A Clarabel-shaped front end for this package's active-set methods.

The point of this module is that it can be *substituted* for :mod:`clarabel`.
Code written against Clarabel's Python API

    solver = clarabel.DefaultSolver(p_mat, q, a_mat, b, cones, settings)
    solution = solver.solve()

runs unchanged against

    solver = active_set.DefaultSolver(p_mat, q, a_mat, b, cones, settings)
    solution = solver.solve()

with the same standard form -- minimise ``(1/2) x' P x + q' x`` subject to
``A x + s == b`` with ``s`` in a product of cones -- the same solution
attributes, and the same :class:`clarabel.SolverStatus` values, which are
re-exported here so that ``solution.status == clarabel.SolverStatus.Solved``
keeps working either way.

**What it is for.** Interior-point methods are the right tool for large sparse
cone programs.  They are not obviously the right tool for the small, dense,
inequality-heavy problems that dominate quantitative finance -- a mean-variance
portfolio with a budget row, box bounds and a handful of group limits is fifty
variables and a hundred constraints, all dense.  There, what an active-set
method offers is a different kind of answer: it identifies the optimal *active
set* exactly, and returns the solution of that face to machine precision,
instead of approaching the face from the interior and leaving the caller to
decide which near-zero slacks were meant to be zero.  For a portfolio that means
*which* holdings sit on a bound is a discrete answer, returned exactly; the bound
itself then holds to machine precision rather than to ``1e-8``, and so do the
shadow prices on the binding rows.

**What it handles.** The program is dispatched on its cones:

- ``ZeroConeT`` and ``NonnegativeConeT`` blocks, in any order -- an equality- and
  inequality-constrained convex QP -- go to :func:`cvxball.qp.solve_qp`, a dense
  dual active-set method.  ``P`` must be positive definite, or made so by
  Clarabel's own static-regularization settings.
- A program made of ``SecondOrderConeT`` blocks in the shape
  :func:`cvxball.solver._build_soc_program` produces -- the smallest enclosing
  ball of a point cloud, this package's original problem -- goes to the
  support-set method in :mod:`cvxball.solver`.
- Anything else is refused, with a message naming what it saw.  This is a
  deliberately narrow solver, and saying so at construction time is better than
  returning a status for a program that was never going to be solved here.

**Which settings are read.** ``verbose``, ``tol_feas``, and the two
``static_regularization_*`` fields.  The rest are interior-point controls with no
active-set meaning: ``max_iter`` counts a different kind of iteration (Clarabel's
default of 200 would cap a QP that needs one pivot per binding constraint), and
``tol_gap_abs``/``tol_gap_rel`` describe a duality gap this method closes exactly
rather than approaches.  They are ignored rather than approximated.
"""

import time
from dataclasses import dataclass
from typing import Any

import clarabel
import numpy as np
import scipy.sparse as sp

from cvxball.qp import FEAS_TOL, solve_qp, symmetrise
from cvxball.solver import _active_set_ball, _ActiveSetError, _build_soc_program, _validate

# Re-exported so this module is a drop-in for `clarabel` and not merely similar to
# it: a caller building cones or comparing statuses can take them from whichever
# of the two modules it imported.  These are the only things taken from Clarabel
# here -- the methods themselves are NumPy and SciPy.
DefaultSettings = clarabel.DefaultSettings  # ty: ignore[unresolved-attribute]
SolverStatus = clarabel.SolverStatus  # ty: ignore[unresolved-attribute]
ZeroConeT = clarabel.ZeroConeT  # ty: ignore[unresolved-attribute]
NonnegativeConeT = clarabel.NonnegativeConeT  # ty: ignore[unresolved-attribute]
SecondOrderConeT = clarabel.SecondOrderConeT  # ty: ignore[unresolved-attribute]

# Cones are recognised by class name rather than by `isinstance`, so a caller may
# pass Clarabel's cone objects, this module's re-exports of them, or any object
# with the same name and a `dim` -- which is all the information a cone carries.
_ZERO = "ZeroConeT"
_NONNEGATIVE = "NonnegativeConeT"
_SOC = "SecondOrderConeT"

# The two program families this front end recognises, as set by `__init__`.
_QP = "qp"
_BALL = "ball"


@dataclass(frozen=True)
class DefaultSolution:
    """The result of a solve, with the attributes Clarabel's solution object has.

    ``x``, ``z`` and ``s`` are the primal, the conic dual, and the slack, in
    Clarabel's convention: ``A x + s == b`` with ``s`` in the cone and ``z`` in its
    dual, so ``P x + A' z + q == 0`` at optimality.

    Two fields are defined slightly differently here, and the difference is worth
    knowing.  ``s`` is *constructed* as ``b - A x`` rather than iterated towards,
    so it satisfies the linear equality identically and cannot report an error
    there; ``r_prim`` therefore measures the only primal error that remains, which
    is how far ``s`` falls outside its cone.  ``r_dual`` is the infinity norm of
    ``P x + A' z + q``, relatively scaled.  Both are computed against the
    program as it was passed in, so if a singular ``P`` had to be regularised to
    factorise it, ``r_dual`` is where that shows up.
    """

    x: np.ndarray
    z: np.ndarray
    s: np.ndarray
    status: Any
    obj_val: float
    obj_val_dual: float
    solve_time: float
    iterations: int
    r_prim: float
    r_dual: float


def _dense(matrix: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    """Coerce a sparse or dense matrix argument to a dense float array.

    Args:
        matrix: A SciPy sparse matrix, a dense array, or ``None`` for a zero
                matrix (which is how a caller writes "no quadratic term").
        shape: The ``(rows, cols)`` the matrix must have.
        name: The argument's name, for the error message.

    Returns:
        The dense ``float64`` array.

    Raises:
        ValueError: If the matrix does not have the required shape.
    """
    if matrix is None:
        return np.zeros(shape)

    array = np.asarray(matrix.todense() if sp.issparse(matrix) else matrix, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")  # noqa: TRY003
    return array


def _cone_kinds(cones: list[Any]) -> list[tuple[str, int]]:
    """Reduce the cone list to the ``(kind, dim)`` pairs the dispatch needs.

    Args:
        cones: The cone objects, as passed to the solver.

    Returns:
        One ``(class name, dimension)`` pair per cone.

    Raises:
        ValueError: If the list is empty, or a cone has no usable dimension.
    """
    if not cones:
        raise ValueError("cones is empty: a program with no constraint blocks has nothing to solve")  # noqa: TRY003

    kinds = []
    for index, cone in enumerate(cones):
        dim = getattr(cone, "dim", None)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"cones[{index}] has no positive integer `dim`: {cone!r}")  # noqa: TRY003
        kinds.append((type(cone).__name__, dim))
    return kinds


def _linear_split(kinds: list[tuple[str, int]], rows: int) -> np.ndarray:
    """Mark which constraint rows are equalities.

    Args:
        kinds: The ``(kind, dim)`` pairs from :func:`_cone_kinds`, all of them
               either zero cones or non-negative cones.
        rows: The number of rows of ``A``.

    Returns:
        A boolean mask over the rows: ``True`` on a row belonging to a zero cone.
    """
    is_equality = np.zeros(rows, dtype=bool)
    start = 0
    for kind, dim in kinds:
        if kind == _ZERO:
            is_equality[start : start + dim] = True
        start += dim
    return is_equality


def _ball_points(a_mat: np.ndarray, b: np.ndarray, q: np.ndarray, p_mat: np.ndarray, dim: int) -> np.ndarray:
    """Recover the point cloud from a smallest-enclosing-ball cone program.

    The ball program has one second-order cone per point, and the point's
    coordinates sit in ``b``; so the cloud can be read straight out of the
    right-hand side.  What must not be skipped is the check that the rest of the
    program is the one those points imply, which is done by rebuilding it with
    :func:`cvxball.solver._build_soc_program` and comparing.  Reading ``b`` and
    trusting it would mean quietly answering a different question whenever the
    cone structure was only *nearly* this one.

    Args:
        a_mat: The dense constraint matrix.
        b: The right-hand side.
        q: The linear objective term.
        p_mat: The (symmetrised) quadratic objective term.
        dim: The ambient dimension, one less than each cone's dimension.

    Returns:
        The ``(n, dim)`` point cloud.

    Raises:
        ValueError: If the program is not the enclosing-ball program for those
                    points.
    """
    points = _validate(b.reshape(-1, dim + 1)[:, 1:])
    _, expected_q, expected_a, expected_b, _ = _build_soc_program(points)

    matches = (
        not p_mat.any()
        and np.array_equal(q, expected_q)
        and np.array_equal(b, expected_b)
        and not np.abs(a_mat - expected_a.toarray()).any()
    )
    if not matches:
        raise ValueError(  # noqa: TRY003
            "this second-order-cone program is not the smallest-enclosing-ball program. "
            "The active-set front end solves second-order cones only in that one shape "
            "(as built by cvxball.solver._build_soc_program); use clarabel.DefaultSolver "
            "for a general second-order-cone program."
        )
    return points


class DefaultSolver:
    """An active-set solver with Clarabel's constructor and :meth:`solve`.

    The program is inspected and classified at construction time, as Clarabel
    checks its arguments then: a program this solver cannot handle raises here,
    rather than coming back later as a status that would suggest the method tried
    and failed.  Everything numerical happens in :meth:`solve`.

    Example:
        A tiny portfolio problem: minimise variance subject to being fully
        invested and long-only.  The two assets are symmetric here, so the answer
        is the equally weighted portfolio.

        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from cvxball import active_set
        >>> covariance = sp.csc_matrix(np.array([[0.04, 0.01], [0.01, 0.04]]))
        >>> a_mat = sp.csc_matrix(np.vstack([np.ones((1, 2)), -np.eye(2)]))
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cones = [active_set.ZeroConeT(1), active_set.NonnegativeConeT(2)]
        >>> settings = active_set.DefaultSettings.default()
        >>> settings.verbose = False
        >>> solution = active_set.DefaultSolver(covariance, np.zeros(2), a_mat, b, cones, settings).solve()
        >>> solution.status == active_set.SolverStatus.Solved
        True
        >>> solution.x
        array([0.5, 0.5])
    """

    def __init__(
        self,
        p_mat: Any,
        q: np.ndarray,
        a_mat: Any,
        b: np.ndarray,
        cones: list[Any],
        settings: Any = None,
    ) -> None:
        """Accept and classify a cone program in Clarabel's standard form.

        Args:
            p_mat: The ``(n, n)`` quadratic objective term, sparse or dense, and
                   either symmetric or upper-triangular (Clarabel reads only the
                   upper triangle, so both conventions are accepted).  ``None``
                   means zero.
            q: The ``(n,)`` linear objective term.
            a_mat: The ``(m, n)`` constraint matrix, sparse or dense.
            b: The ``(m,)`` constraint right-hand side.
            cones: The cone blocks, whose dimensions must sum to ``m``.
            settings: A :class:`clarabel.DefaultSettings`, or any object with the
                      same attributes.  ``None`` takes the defaults.

        Raises:
            ValueError: If the arguments are inconsistent, or if the program is
                        not one of the two families this solver handles (see the
                        module docstring).
        """
        self._q = np.asarray(q, dtype=np.float64).ravel()
        self._b = np.asarray(b, dtype=np.float64).ravel()
        self._settings = DefaultSettings.default() if settings is None else settings

        n_vars, n_rows = self._q.size, self._b.size
        self._a = _dense(a_mat, (n_rows, n_vars), "A")
        self._p = symmetrise(_dense(p_mat, (n_vars, n_vars), "P"))

        kinds = _cone_kinds(cones)
        cone_rows = sum(dim for _, dim in kinds)
        if cone_rows != n_rows:
            raise ValueError(f"the cone dimensions sum to {cone_rows}, but A has {n_rows} rows")  # noqa: TRY003

        names = {kind for kind, _ in kinds}
        if names <= {_ZERO, _NONNEGATIVE}:
            self._kind = _QP
            self._is_equality = _linear_split(kinds, n_rows)
            self._points = np.zeros((0, 0))
        elif names == {_SOC} and len({dim for _, dim in kinds}) == 1:
            self._kind = _BALL
            self._is_equality = np.zeros(n_rows, dtype=bool)
            self._points = _ball_points(self._a, self._b, self._q, self._p, kinds[0][1] - 1)
        else:
            raise ValueError(  # noqa: TRY003
                f"unsupported cone combination {sorted(names)}: this solver handles a QP over "
                f"{_ZERO}/{_NONNEGATIVE} blocks, or the enclosing-ball program over equally "
                f"sized {_SOC} blocks. Use clarabel.DefaultSolver for anything else."
            )

    def solve(self) -> DefaultSolution:
        """Solve the program and return a Clarabel-shaped solution.

        Returns:
            The :class:`DefaultSolution`.  Its ``status`` is a
            :class:`clarabel.SolverStatus`: ``Solved`` normally,
            ``PrimalInfeasible`` when the method proves no feasible point exists,
            and ``MaxIterations`` or ``NumericalError`` when a run degenerates --
            in which case ``x``, ``s`` and ``z`` are filled with ``nan``, since
            an active-set iterate that never reached optimality is not a solution
            anyone should be tempted to read.

        Raises:
            ValueError: If ``P`` cannot be factorised (see
                        :func:`cvxball.qp._cholesky`), or if the point cloud of a
                        ball program is degenerate.
        """
        started = time.perf_counter()
        if self._kind == _QP:
            x, z, iterations, status = self._solve_qp()
        else:
            x, z, iterations, status = self._solve_ball()
        elapsed = time.perf_counter() - started

        return self._assemble(x, z, iterations, status, elapsed)

    def _solve_qp(self) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Solve a zero-cone/non-negative-cone program as a dense QP.

        The cone program's rows are split into the equalities ``E x == f`` and the
        inequalities ``G x <= h`` that :func:`cvxball.qp.solve_qp` takes, and the
        multipliers it returns are scattered back into one dual vector in the
        original row order -- non-negative on the inequality rows, unconstrained
        on the equality rows, which is exactly membership of the dual cone.

        Returns:
            The tuple ``(x, z, iterations, status)``.
        """
        equality = self._is_equality
        settings = self._settings

        regularization = 0.0
        if bool(getattr(settings, "static_regularization_enable", True)):
            regularization = float(getattr(settings, "static_regularization_constant", 1e-8)) + float(
                getattr(settings, "static_regularization_proportional", 0.0)
            )

        solution = solve_qp(
            self._p,
            self._q,
            self._a[equality],
            self._b[equality],
            self._a[~equality],
            self._b[~equality],
            tol_feas=float(getattr(settings, "tol_feas", FEAS_TOL)),
            regularization=regularization,
            verbose=bool(getattr(settings, "verbose", False)),
        )

        z = np.zeros(self._b.size)
        z[equality] = solution.y_eq
        z[~equality] = solution.y_in
        return solution.x, z, solution.iterations, solution.status

    def _solve_ball(self) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Solve the enclosing-ball cone program with the support-set method.

        The dual weights the method returns *are* the conic dual variables, up to
        the arrangement Clarabel expects: point ``i``'s cone gets
        ``z_i = u_i * (1, -(p_i - x) / R)``, which lies on the boundary of the
        second-order cone whenever ``u_i > 0`` -- the support points -- and is zero
        for every other point.  That is complementarity with the slack
        ``s_i = (R, p_i - x)``, and it is why the weights of the support set and
        the dual of the cone program are the same object.

        Returns:
            The tuple ``(x, z, iterations, status)``.
        """
        points = self._points
        verbose = bool(getattr(self._settings, "verbose", False))

        try:
            ball = _active_set_ball(points, verbose)
        except _ActiveSetError as failure:
            nan = np.full(self._q.size, np.nan)
            return nan, np.full(self._b.size, np.nan), 0, failure.status

        x = np.concatenate([[ball.radius], ball.centre])
        offsets = points - ball.centre
        # Dividing by the radius is what puts z on the cone boundary; at radius
        # zero every point coincides with the centre, the slack is the cone's apex,
        # and (u_i, 0) is complementary to it for any weights at all.
        scaled = offsets / ball.radius if ball.radius > 0.0 else offsets * 0.0
        z = np.column_stack([ball.weights, -ball.weights[:, None] * scaled]).ravel()
        return x, z, ball.iterations, "Solved"

    def _assemble(self, x: np.ndarray, z: np.ndarray, iterations: int, status: str, elapsed: float) -> DefaultSolution:
        """Build the solution object, objectives and residuals from a primal-dual pair.

        Args:
            x: The primal solution.
            z: The conic dual solution.
            iterations: The number of pivots taken.
            status: The name of the :class:`clarabel.SolverStatus` to report.
            elapsed: The wall-clock time the solve took, in seconds.

        Returns:
            The :class:`DefaultSolution`.
        """
        s = self._b - self._a @ x
        quadratic = 0.5 * float(x @ self._p @ x)
        primal = quadratic + float(self._q @ x)
        dual = -quadratic - float(self._b @ z)

        return DefaultSolution(
            x=x,
            z=z,
            s=s,
            status=getattr(SolverStatus, status),
            obj_val=primal,
            obj_val_dual=dual,
            solve_time=elapsed,
            iterations=iterations,
            r_prim=self._cone_violation(s),
            r_dual=self._dual_residual(x, z),
        )

    def _cone_violation(self, s: np.ndarray) -> float:
        """Measure how far the slack falls outside its cone, relative to its size.

        Args:
            s: The slack ``b - A x``.

        Returns:
            The relative violation: zero when ``s`` is in the cone.
        """
        scale = max(1.0, float(np.abs(self._b).max(initial=0.0)))
        if self._kind == _QP:
            equality = self._is_equality
            violation = max(
                float(np.abs(s[equality]).max(initial=0.0)),
                float(np.maximum(-s[~equality], 0.0).max(initial=0.0)),
            )
            return violation / scale

        # One second-order cone per point: the slack (r, p_i - x) is in the cone
        # exactly when its first entry dominates the norm of the rest.
        blocks = s.reshape(-1, self._q.size)
        norms = np.linalg.norm(blocks[:, 1:], axis=1)
        return float(np.maximum(norms - blocks[:, 0], 0.0).max(initial=0.0)) / scale

    def _dual_residual(self, x: np.ndarray, z: np.ndarray) -> float:
        """Measure the stationarity residual ``P x + A' z + q``, relatively scaled.

        Args:
            x: The primal solution.
            z: The conic dual solution.

        Returns:
            The relative infinity norm of the residual.
        """
        residual = self._p @ x + self._a.T @ z + self._q
        scale = max(1.0, float(np.abs(self._q).max(initial=0.0)))
        return float(np.abs(residual).max(initial=0.0)) / scale
