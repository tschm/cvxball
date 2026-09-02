"""Convex utilities for computing the minimum enclosing circle/ball.

Provides :func:`min_circle_clarabel`, which assembles the second-order-cone
program for the smallest enclosing ball by hand and calls the Clarabel solver
directly.
"""

from typing import Any

import clarabel
import numpy as np
import scipy.sparse as sp


def _validate(points: np.ndarray) -> np.ndarray:
    """Check that ``points`` is a usable point cloud, and return it as floats.

    Every rejection here is one the solver would otherwise take: a 1-D array used
    to fail when ``points.shape`` was unpacked into two names, and an empty or
    non-finite cloud reached Clarabel and came back as ``DualInfeasible`` or
    ``NumericalError`` -- a status describing the program rather than the input
    that produced it.  Refusing the same cases with the reason keeps the caller's
    error about the caller's data.

    Args:
        points: The candidate array of shape ``(n, d)``.

    Returns:
        ``points`` as a float64 array, ready for the solver.

    Raises:
        ValueError: If ``points`` is not two-dimensional, holds no points, has
                    no coordinates, or contains a NaN or an infinity.
    """
    array = np.asarray(points, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(f"points must be a 2-D (n, d) array, got {array.ndim}-D of shape {array.shape}")  # noqa: TRY003
    if array.shape[0] == 0:
        raise ValueError("points is empty: the smallest enclosing ball of no points is undefined")  # noqa: TRY003
    if array.shape[1] == 0:
        raise ValueError("points has no coordinates: shape (n, 0) describes no ambient space")  # noqa: TRY003
    if not np.isfinite(array).all():
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"points must be finite: found {bad} NaN or infinite coordinate(s)")  # noqa: TRY003

    return array


def _build_soc_program(
    points: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, list[Any]]:
    """Assemble the Clarabel second-order-cone program for the enclosing ball.

    The problem is written in Clarabel's standard form::

        minimise   (1/2) z' P z + q' z
        subject to A z + s = b,  s ∈ K

    where the decision vector is ``z = [r, x₁, …, x_d]`` (radius followed by
    the d centre coordinates), the objective is to minimise *r* (so ``P = 0``,
    ``q = e₀``), and the feasible set is a product of *n* second-order cones.

    For each point ``p_i`` we require ``[r, p_i - x] in Q^{d+1}``, which gives
    one SOC block of dimension ``d + 1`` per point.

    Args:
        points: A numpy array of shape ``(n, d)`` where *n* is the number of
                points and *d* is the ambient dimension.

    Returns:
        A tuple ``(p_mat, q, a_mat, b, cones)`` of the objective quadratic
        ``P``, the objective linear term ``q``, the constraint matrix ``A``,
        the constraint right-hand side ``b``, and the list of *n* second-order
        cones — the exact positional arguments Clarabel's ``DefaultSolver``
        expects.
    """
    n, d = points.shape
    n_vars = 1 + d  # decision vector: [r, x_1, ..., x_d]

    # --- Objective: minimise r -----------------------------------------------
    p_mat = sp.csc_matrix((n_vars, n_vars))
    q = np.zeros(n_vars)
    q[0] = 1.0

    # --- Constraints: one SOC block of size (d+1) per point ------------------
    # We need b - a_mat @ z = s  where s in K.
    # For point i the desired slack is  s = [r, p_i - x],  so:
    #   row i*(d+1)     : b = 0,       a_mat col 0   = -1  (gives s_0 = r)
    #   row i*(d+1)+j   : b = p_i[j], a_mat col j   = +1  (gives s_j = p_ij - x_j)
    total_rows = n * (d + 1)

    # Entries for the r column (column 0): -1 at each block's first row
    r_rows = np.arange(n) * (d + 1)

    # Entries for the x columns (columns 1..d): +1 at each block's inner rows
    x_row_offsets = np.arange(n)[:, None] * (d + 1) + np.arange(1, d + 1)[None, :]  # (n, d)
    x_rows = x_row_offsets.ravel()
    x_cols = np.tile(np.arange(1, d + 1), n)

    all_rows = np.concatenate([r_rows, x_rows])
    all_cols = np.concatenate([np.zeros(n, dtype=np.intp), x_cols])
    all_vals = np.concatenate([-np.ones(n), np.ones(n * d)])

    a_mat = sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(total_rows, n_vars))

    b = np.zeros(total_rows)
    b[x_rows] = points.ravel()

    # --- Cones: n SOC cones each of dimension (d+1) --------------------------
    cones = [clarabel.SecondOrderConeT(d + 1) for _ in range(n)]  # ty: ignore[unresolved-attribute]

    return p_mat, q, a_mat, b, cones


def min_circle_clarabel(points: np.ndarray, verbose: bool = False) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing circle for a set of points using Clarabel directly.

    The second-order-cone program is assembled by :func:`_build_soc_program`;
    this function then solves it and extracts the optimal radius and centre.

    Args:
        points: A numpy array of shape ``(n, d)`` where *n* is the number of
                points and *d* is the ambient dimension.
        verbose: If ``True``, print Clarabel's iteration log.  Defaults to
                 ``False``.

    Returns:
        A tuple ``(radius, center)`` where *radius* is the optimal enclosing
        radius (float) and *center* is a numpy array of shape ``(d,)``.

    Raises:
        ValueError: If ``points`` is not a finite, non-empty ``(n, d)`` array
                    (see :func:`_validate`), or if Clarabel does not return a
                    ``Solved`` status.

    Example:
        These three points form a right triangle, so the smallest enclosing
        circle is the one on its hypotenuse: centred at ``(0.5, 0.5)`` with
        radius ``sqrt(2) / 2``.  The values are asserted rather than merely
        assigned, so the example fails if the answer ever changes.

        The two are shown to different precisions because the solver resolves
        them to different precisions: the radius is the objective and lands
        within 1e-8, while the centre only converges to ~3e-5 here, so pinning
        it past three decimals would make this example fail on some platforms
        rather than catch a regression.

        >>> import numpy as np
        >>> from cvxball import min_circle_clarabel
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> radius, center = min_circle_clarabel(points)
        >>> round(radius, 6)
        0.707107
        >>> np.round(center, 3)
        array([0.5, 0.5])
    """
    p_mat, q, a_mat, b, cones = _build_soc_program(_validate(points))

    # --- Solve ---------------------------------------------------------------
    settings = clarabel.DefaultSettings.default()  # ty: ignore[unresolved-attribute]
    settings.verbose = verbose

    solver = clarabel.DefaultSolver(p_mat, q, a_mat, b, cones, settings)  # ty: ignore[unresolved-attribute]
    solution = solver.solve()

    if solution.status != clarabel.SolverStatus.Solved:  # ty: ignore[unresolved-attribute]
        # `_validate` has already ruled out the input-shaped causes -- empty, wrong
        # rank, non-finite -- so what reaches here is a well-formed cloud the solver
        # still could not handle. In practice that means conditioning: coordinates
        # spanning many orders of magnitude, or an extent negligible against the
        # cloud's distance from the origin. Both are recoverable by the caller, so
        # say so rather than only naming Clarabel's internal state.
        raise ValueError(  # noqa: TRY003
            f"Clarabel did not converge: status = {solution.status}. "
            f"The input is well-formed, so this usually means poor conditioning: "
            f"try recentring the points on their mean, or rescaling them to a "
            f"comparable magnitude, and solving again."
        )

    return float(solution.x[0]), np.asarray(solution.x[1:])
