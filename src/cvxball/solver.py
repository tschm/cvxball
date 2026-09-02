"""Convex utilities for computing the minimum enclosing circle/ball.

Two solvers, one interface — both take ``(points, verbose=False)`` and return
``(radius, center)``, so they are interchangeable:

- :func:`min_circle_clarabel` assembles the second-order-cone program for the
  smallest enclosing ball by hand and calls the Clarabel solver directly.
- :func:`min_circle_active_set` runs an active-set QP method on the *dual* of the
  same problem, keeping a support set of points on the ball's boundary.  It needs
  no solver dependency, each of its iterations costs one small dense linear solve,
  and it terminates at an exact vertex of the dual feasible set rather than at an
  interior-point tolerance.
"""

from typing import Any

import clarabel
import numpy as np
import scipy.sparse as sp


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
        ValueError: If Clarabel does not return a ``Solved`` status.

    Example:
        >>> import numpy as np
        >>> from cvxball.solver import min_circle_clarabel
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> radius, center = min_circle_clarabel(points)
    """
    p_mat, q, a_mat, b, cones = _build_soc_program(points)

    # --- Solve ---------------------------------------------------------------
    settings = clarabel.DefaultSettings.default()  # ty: ignore[unresolved-attribute]
    settings.verbose = verbose

    solver = clarabel.DefaultSolver(p_mat, q, a_mat, b, cones, settings)  # ty: ignore[unresolved-attribute]
    solution = solver.solve()

    if solution.status != clarabel.SolverStatus.Solved:  # ty: ignore[unresolved-attribute]
        raise ValueError(f"Clarabel did not converge: status = {solution.status}")  # noqa: TRY003

    return float(solution.x[0]), np.asarray(solution.x[1:])


# --- Active-set (support-set) method ------------------------------------------
#
# The enclosing-ball problem has the concave dual
#
#     maximise  sum_i u_i ||p_i||^2 - ||sum_i u_i p_i||^2   over  u >= 0, sum_i u_i = 1,
#
# a quadratic program over the unit simplex whose optimal centre is x = sum_i u_i p_i
# and whose optimal value is the squared radius. Writing the convex negated dual as
# g(u) = ||x||^2 - sum_i u_i ||p_i||^2, its KKT conditions read
#
#     ||p_i - x|| == R  for every i with u_i > 0   (support points sit on the boundary)
#     ||p_i - x|| <= R  for every i with u_i == 0  (every other point is enclosed),
#
# which is precisely the geometric optimality certificate for the smallest enclosing
# ball. The routines below run a primal active-set QP method on that dual: the free
# set is the current support, every subproblem is a small dense linear system, and
# each iteration either adds the farthest violating point or drops a support point
# whose weight reaches zero.

# Weights live on the unit simplex, so this is an absolute floor on a support weight.
_DROP_TOL = 1e-12
# Slack, relative to the squared radius, before a point counts as outside the ball.
_FEAS_RTOL = 1e-9
# Second, scale-aware slack, in multiples of the rounding error of a squared distance.
# That error tracks the squared extent of the centred cloud, so a constant absolute
# tolerance would carry units of length^2 and would silently declare optimality on
# small-magnitude inputs.
_FEAS_NOISE = 64.0
# Safety net only: the method is finite, so hitting this means numerical trouble.
_MAX_ITER_PER_POINT = 50


def _sq_dist(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Compute squared distances from every row of ``points`` to ``centre``.

    Differencing before squaring is what keeps this accurate.  Expanding instead to
    ``||p||^2 - 2 p'x + ||x||^2`` costs a cancellation between terms of size
    ``||x||^2``, so for a cloud sitting far from the origin the digits that survive
    are exactly the ones the answer needs — the part of size ``R^2``.

    Args:
        points: A ``(n, d)`` array of points.
        centre: A ``(d,)`` centre.

    Returns:
        The ``(n,)`` array of squared distances.
    """
    offsets = points - centre
    squared: np.ndarray = np.einsum("ij,ij->i", offsets, offsets)
    return squared


def _affine_null_space(face: np.ndarray) -> np.ndarray:
    """Find weight directions that reshuffle the support without moving the centre.

    A weight update ``p`` leaves both the simplex constraint and the centre
    ``x = face.T @ u`` untouched exactly when ``sum(p) == 0`` and ``face.T @ p == 0``.
    Eliminating ``p_0 = -(p_1 + ... + p_{m-1})`` collapses that pair of conditions to
    the single condition ``edges.T @ p[1:] == 0`` on the edge matrix
    ``edges[j] = q_j - q_0``, so such directions exist exactly when the face is
    affinely *dependent*.  Along one of them the quadratic term of ``g`` is frozen, so
    ``g`` is *linear* and the equality-constrained subproblem is unbounded — the active
    set then has to shrink instead of jumping to a minimiser.

    Testing the rank on the edges rather than on ``[ones; face.T]`` is what keeps this
    scale-invariant.  With the row of ones stacked on, its unit entries dominate the
    singular values, and every cloud whose extent falls below ``eps`` relative to 1
    would be misjudged affinely dependent.

    Args:
        face: A ``(m, d)`` array of the points currently in the support set.

    Returns:
        An ``(m, q)`` array whose columns span those directions.  It has ``q == 0``
        columns exactly when the points of ``face`` are affinely independent — the
        regular case, where the subproblem instead has the unique solution computed
        by :func:`_face_weights`.
    """
    edges = face[1:] - face[0]
    if edges.shape[0] == 0:
        return np.zeros((1, 0))

    left, singular_values, _ = np.linalg.svd(edges, full_matrices=True)
    cutoff = max(edges.shape) * float(np.finfo(np.float64).eps) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > cutoff))

    # Lift each edge-space null vector z back to a weight direction [-sum(z), z].
    tail = left[:, rank:]
    return np.vstack([-tail.sum(axis=0, keepdims=True), tail])


def _face_weights(face: np.ndarray) -> np.ndarray:
    """Solve the subproblem: put every point of one face on a common sphere.

    The centre is written as ``x = q_0 + D y``, where ``D`` holds the edges
    ``q_j - q_0`` and so confines ``x`` to the affine hull of the face.  Equating
    the distances from ``x`` to ``q_0`` and to each ``q_j`` then collapses to the
    tiny normal-equation system ``(D' D) y = h`` with ``h_j = ||q_j - q_0||^2 / 2``,
    which is non-singular precisely because the face is affinely independent.

    Args:
        face: A ``(m, d)`` array of affinely independent points.

    Returns:
        The ``(m,)`` weights, summing to one, that express the circumcentre as an
        affine combination of ``face``.  A negative entry means the circumcentre
        lies outside the simplex, so the caller has to drop a support point
        instead of taking the full step.
    """
    edges = face[1:] - face[0]
    if edges.shape[0] == 0:
        return np.ones(1)

    gram = edges @ edges.T
    rhs = 0.5 * np.einsum("ij,ij->i", edges, edges)
    y = np.linalg.solve(gram, rhs)
    return np.concatenate(([1.0 - float(y.sum())], y))


def min_circle_active_set(points: np.ndarray, verbose: bool = False) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing circle with an active-set method.

    A drop-in replacement for :func:`min_circle_clarabel` — same signature, same
    ``(radius, center)`` result — that runs an active-set QP method on the dual of
    the enclosing-ball problem instead of handing a cone program to Clarabel.  It
    maintains a *support set* of points held on the ball's boundary and repeatedly

    1. centres the ball on that support set by solving one small linear system
       (:func:`_face_weights`),
    2. shrinks the support when it cannot hold — either because a weight would turn
       negative, or because the set has become affinely dependent
       (:func:`_affine_null_space`) — moving as far as non-negativity allows, and
    3. adds the farthest point that is still outside the ball.

    Each subproblem is a ``k x k`` solve with ``k <= d``, so the cost per iteration
    is driven by the dimension rather than by the number of points, and the method
    stops at an exact vertex of the dual feasible set instead of at an
    interior-point tolerance.

    Args:
        points: A numpy array of shape ``(n, d)`` where *n* is the number of
                points and *d* is the ambient dimension.
        verbose: If ``True``, print the support size and radius per iteration.
                 Defaults to ``False``.

    Returns:
        A tuple ``(radius, center)`` where *radius* is the optimal enclosing
        radius (float) and *center* is a numpy array of shape ``(d,)``.

    Raises:
        ValueError: If the iteration limit is reached without the optimality
                    certificate holding.

    Example:
        >>> import numpy as np
        >>> from cvxball.solver import min_circle_active_set
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> radius, center = min_circle_active_set(points)
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    # Work in coordinates centred on the cloud, undoing the shift on the way out.
    # Every quantity below -- the subproblem's Gram matrix, the squared distances, the
    # rounding floor -- is then governed by the *extent* of the cloud rather than by
    # its distance from an arbitrary origin.  Without this, a cloud of extent 1 sitting
    # a million units out gets a rounding floor of the same order as its own radius,
    # and the optimality test below then accepts a ball that is visibly too small.
    shift = pts.mean(axis=0)
    pts = pts - shift
    sq_norms: np.ndarray = np.einsum("ij,ij->i", pts, pts)

    # Warm start with one point and whatever sits farthest from it: that puts both
    # ends of a near-diameter into the support straight away, which is usually
    # where the optimum keeps them.
    seed = int(np.argmax(_sq_dist(pts, pts[0])))
    # Rounding floor of a squared distance, set by the magnitude of the coordinates
    # that go into it rather than by any fixed constant.
    noise_floor = _FEAS_NOISE * float(np.finfo(np.float64).eps) * float(sq_norms.max())
    free = np.array([seed], dtype=np.intp)
    weights = np.ones(1)

    iteration_limit = _MAX_ITER_PER_POINT * (n + 1)
    for iteration in range(iteration_limit):
        face = pts[free]
        null_space = _affine_null_space(face)

        if null_space.size:
            # Affinely dependent support: g is linear on the null space, so follow
            # steepest descent there until some weight is driven down to zero.
            centre = face.T @ weights
            # grad_i = ||x||^2 - ||p_i - x||^2, less the constant ||x||^2: null
            # directions sum to zero, so dropping it leaves every projection alone
            # while avoiding the cancellation of ||x||^2 against ||p_i||^2.
            gradient = -_sq_dist(face, centre)
            descent = -(null_space @ (null_space.T @ gradient))
            # Rescale to a unit-max direction.  The gradient carries units of length^2,
            # so comparing it against the dimensionless weight tolerance below would
            # make the whole method scale-dependent: on a cloud of extent 1e-20 no
            # component would ever look binding.  Only the direction matters here --
            # `alpha` cancels any positive factor -- so normalising costs nothing.
            largest = float(np.abs(descent).max())
            step = descent / largest if largest > 0.0 else descent
        else:
            target = _face_weights(face)
            if target.min() >= -_DROP_TOL:
                # The subproblem solution is feasible: take it, then test the KKT
                # certificate and, if it fails, free the most violated point.
                weights = np.maximum(target, 0.0)
                weights /= weights.sum()
                centre = face.T @ weights
                dist_sq = _sq_dist(pts, centre)
                radius_sq = float(dist_sq[free].max())
                radius = float(np.sqrt(max(radius_sq, 0.0)))
                if verbose:
                    print(f"[{iteration:4d}] support={free.size:3d} radius={radius:.12g}")

                worst = int(np.argmax(dist_sq))
                if dist_sq[worst] <= radius_sq * (1.0 + _FEAS_RTOL) + noise_floor:
                    return radius, centre + shift

                keep = weights > _DROP_TOL
                free = np.append(free[keep], worst)
                weights = np.append(weights[keep], 0.0)
                continue

            step = target - weights

        # Longest step along `step` that keeps every weight non-negative.  A step
        # of zero is impossible: the newly freed point always has step > 0, so the
        # support strictly shrinks here and `g` strictly decreases.
        binding = step < -_DROP_TOL
        ratios = np.where(binding, -weights / np.where(binding, step, -1.0), np.inf)
        alpha = float(ratios.min())
        if not np.isfinite(alpha):
            # Nothing blocks the step, so the support cannot shrink.  Unreachable in
            # exact arithmetic; bail out rather than propagate a non-finite weight.
            raise ValueError("active-set method stalled: no support point blocks the step")  # noqa: TRY003
        weights = np.maximum(weights + alpha * step, 0.0)

        keep = weights > _DROP_TOL
        free, weights = free[keep], weights[keep]
        weights /= weights.sum()
        if verbose:
            print(f"[{iteration:4d}] support={free.size:3d} drop step={alpha:.6g}")

    raise ValueError(f"active-set method did not converge in {iteration_limit} iterations")  # noqa: TRY003
