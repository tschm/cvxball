"""Convex utilities for computing the minimum enclosing circle/ball.

:func:`min_circle_active_set` runs an active-set QP method on the *dual* of the
enclosing-ball problem -- a QP over the unit simplex -- keeping a support set of
points on the ball's boundary.  Each iteration costs one small dense linear
solve, and it terminates at an exact vertex of the dual feasible set rather than
at an interior-point tolerance.

It is the only solver this package ships. NumPy carries the method; SciPy
carries the factorisation of its support at large `d`, through the compiled
Givens updates in :mod:`cvxball._frame` -- see :data:`_MAINTAIN_MIN_DIM` for
where that starts to pay and why it is not used below it.
The cone-program route that used to sit beside it -- assembling the
second-order-cone program by hand and handing it to Clarabel -- now lives in
``experiments/clarabel_ball.py``, because that is what it had become: the
reference this method is measured against rather than a second way to get an
answer.  Moving it there is what lets Clarabel be a development dependency, so
installing this package pulls in NumPy and nothing else.
"""

import numpy as np

from cvxball._frame import _MaintainedFace


def _validate(points: np.ndarray) -> np.ndarray:
    """Check that ``points`` is a usable point cloud, and return it as floats.

    Every rejection here is one a solver would otherwise take: a 1-D array used to
    fail when ``points.shape`` was unpacked into two names, and an empty or
    non-finite cloud used to reach the cone solver and come back as
    ``DualInfeasible`` or ``NumericalError`` -- a status describing the program
    rather than the input that produced it.  Refusing the same cases with the
    reason keeps the caller's error about the caller's data.  The Clarabel route in
    ``experiments/clarabel_ball.py`` imports this function for that reason.

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
# Ambient dimension from which carrying the factorisation across iterations starts
# to pay. Below it the SciPy update calls cost more than the rebuild they save --
# measured at 1000 standard normal points on an Apple M4 Pro, best of many runs:
#
#     d      rebuild   maintained
#     20     0.79 ms      1.05 ms     0.75x   -- maintaining loses
#     50     2.18 ms      2.52 ms     0.86x
#    100     4.78 ms      4.60 ms     1.04x   -- parity, near enough
#    250    20.6  ms     16.2  ms     1.27x   -- and it grows from here
#   8000    11.3  s       3.19 s      3.54x
#
# So this is an empirical constant, not a derived one, and it is a threshold on
# the *dimension* rather than on the support size because the choice has to be
# made once, before the support exists. `maintain=` overrides it either way.
_MAINTAIN_MIN_DIM = 100


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

    # Only the left factor is read.  When ``edges`` is no taller than it is wide the
    # reduced decomposition already returns all ``m - 1`` of its columns -- a complete
    # orthonormal basis of the edge row space *and* its null space -- so asking for the
    # full one there buys nothing and costs the ``d x d`` right factor, built and then
    # discarded.  That discarded factor dominates everything at large ``d``: 2 GB per
    # call at ``d = 16000``, and about forty times the cost of the whole solve.
    #
    # The guard is not decoration.  Between a drop and the add that follows, the support
    # can reach ``d + 2`` points, and on such a face the reduced left factor is
    # ``(m - 1) x d`` -- one column short of spanning, and the column it is short of is
    # exactly the null direction this function exists to find.  Reporting that face as
    # affinely independent would send the caller to :func:`_face_weights` with a
    # singular system.
    complete = edges.shape[0] > edges.shape[1]
    left, singular_values, _ = np.linalg.svd(edges, full_matrices=complete)
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


def _shrink(support: "_RebuiltFace | _MaintainedFace", keep: np.ndarray) -> None:
    """Drop every support position whose weight has fallen to zero.

    Positions are removed back to front so the earlier ones keep their indices,
    which matters because a maintained factorisation is repaired per removal.

    Args:
        support: The face to shrink, in place.
        keep: Boolean mask over the current support; ``False`` entries go.
    """
    for position in sorted(np.flatnonzero(~keep).tolist(), reverse=True):
        support.remove(position)


class _RebuiltFace:
    """The support set, with its subproblem recomputed from the points each time.

    The counterpart of :class:`cvxball._frame._MaintainedFace`, and the cheaper of
    the two whenever the support is small: there is no factorisation to carry, so
    nothing to repair, and the whole cost is two small dense decompositions that
    NumPy dispatches straight into LAPACK. Below :data:`_MAINTAIN_MIN_DIM` that
    beats paying SciPy's per-update overhead.
    """

    def __init__(self, points: np.ndarray, seed: int) -> None:
        """Start from the one-point support ``{seed}``.

        Args:
            points: The ``(n, d)`` cloud, already centred and scaled.
            seed: Index of the point the support starts as.
        """
        self._points = points
        self.support: list[int] = [seed]
        self.fallbacks = 0

    @property
    def face(self) -> np.ndarray:
        """Return the ``(m, d)`` array of support points."""
        return self._points[self.support]

    def insert(self, index: int) -> None:
        """Take ``points[index]`` into the support.

        Args:
            index: Index into the cloud of the entering point.
        """
        self.support.append(index)

    def remove(self, position: int) -> None:
        """Drop the support point at ``position``.

        Args:
            position: Index within the support list.
        """
        del self.support[position]

    def null_space(self) -> np.ndarray:
        """Return the weight directions that leave the centre fixed."""
        return _affine_null_space(self.face)

    def circumcentre_weights(self) -> np.ndarray:
        """Return the barycentric weights of the face's circumcentre."""
        return _face_weights(self.face)


def min_circle_active_set(
    points: np.ndarray,
    verbose: bool = False,
    maintain: bool | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing circle with an active-set method.

    An active-set QP method on the dual of the enclosing-ball problem, in place of
    handing a cone program to a conic solver.  It
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
        maintain: Whether to carry the support's factorisation across iterations
                 and repair it, rather than rebuilding it each time.  ``None``,
                 the default, decides on the ambient dimension: repairing costs
                 ``O(d r)`` against ``O(d r^2)`` to rebuild, which is decisive by
                 ``d = 250`` and a net loss below ``d = 100`` where SciPy's
                 per-update overhead exceeds what it saves (see
                 :data:`_MAINTAIN_MIN_DIM`).  Both settings compute the same ball;
                 pass one explicitly only to measure the difference.

    Returns:
        A tuple ``(radius, center)`` where *radius* is the optimal enclosing
        radius (float) and *center* is a numpy array of shape ``(d,)``.

    Raises:
        ValueError: If ``points`` is not a finite, non-empty ``(n, d)`` array
                    (see :func:`_validate`), or if the iteration limit is reached
                    without the optimality certificate holding.

    Example:
        Three points forming a right triangle, whose smallest enclosing circle is
        the one on its hypotenuse.  Both values are pinned to full precision here,
        where the cone program in ``experiments/clarabel_ball.py`` can only pin its
        centre to three decimals — this method stops at an exact vertex of the
        dual feasible set, so on an input whose answer is exactly representable it
        returns that answer bit-for-bit.

        >>> import numpy as np
        >>> from cvxball import min_circle_active_set
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> radius, center = min_circle_active_set(points)
        >>> radius == 2**0.5 / 2
        True
        >>> center
        array([0.5, 0.5])
    """
    pts = _validate(points)
    n = pts.shape[0]

    # Work in coordinates centred on the cloud, undoing the shift on the way out.
    # Every quantity below -- the subproblem's Gram matrix, the squared distances, the
    # rounding floor -- is then governed by the *extent* of the cloud rather than by
    # its distance from an arbitrary origin.  Without this, a cloud of extent 1 sitting
    # a million units out gets a rounding floor of the same order as its own radius,
    # and the optimality test below then accepts a ball that is visibly too small.
    shift = pts.mean(axis=0)
    pts = pts - shift

    # Recentring fixes the origin but not the magnitude, and this method squares
    # everything it touches: the Gram matrix, the squared distances, the noise floor.
    # Squaring halves the usable exponent range, so a cloud of extent 1e-160 -- whose
    # coordinates are ordinary doubles -- has a Gram matrix of order 1e-320, deep in
    # the subnormals, and the solve for the circumcentre then overflows to infinity.
    # The far end fails too, and worse: past 1e+154 the squares saturate and the
    # method returns a radius of zero rather than raising.
    #
    # So normalise the extent to order one.  The factor is a *power of two*, which
    # makes the rescaling exact in binary floating point: not one bit of the answer
    # moves, so the method keeps returning representable answers bit-for-bit (see the
    # example above), and the whole iteration runs where doubles are well behaved.
    # Measuring the extent as a largest coordinate rather than a largest norm is what
    # keeps the measurement itself in range -- a norm would already have squared.
    largest = float(np.abs(pts).max())
    if largest == 0.0:
        # Every point is the same point, so the ball is that point with radius zero.
        return 0.0, shift
    exponent = int(np.frexp(largest)[1])
    pts = np.ldexp(pts, -exponent)

    sq_norms: np.ndarray = np.einsum("ij,ij->i", pts, pts)

    # Warm start with one point and whatever sits farthest from it: that puts both
    # ends of a near-diameter into the support straight away, which is usually
    # where the optimum keeps them.
    seed = int(np.argmax(_sq_dist(pts, pts[0])))
    # Rounding floor of a squared distance, set by the magnitude of the coordinates
    # that go into it rather than by any fixed constant.
    noise_floor = _FEAS_NOISE * float(np.finfo(np.float64).eps) * float(sq_norms.max())

    if maintain is None:
        maintain = pts.shape[1] >= _MAINTAIN_MIN_DIM
    support = _MaintainedFace(pts, seed) if maintain else _RebuiltFace(pts, seed)
    free = support.support
    weights = np.ones(1)

    iteration_limit = _MAX_ITER_PER_POINT * (n + 1)
    for iteration in range(iteration_limit):
        face = support.face
        null_space = support.null_space()

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
            target = support.circumcentre_weights()
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
                    print(f"[{iteration:4d}] support={len(free):3d} radius={radius:.12g}")

                worst = int(np.argmax(dist_sq))
                if dist_sq[worst] <= radius_sq * (1.0 + _FEAS_RTOL) + noise_floor:
                    # Undo the exact power-of-two normalisation, then the shift.
                    return float(np.ldexp(radius, exponent)), np.ldexp(centre, exponent) + shift

                keep = weights > _DROP_TOL
                _shrink(support, keep)
                support.insert(worst)
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
        _shrink(support, keep)
        weights = weights[keep]
        weights /= weights.sum()
        if verbose:
            print(f"[{iteration:4d}] support={len(free):3d} drop step={alpha:.6g}")

    raise ValueError(f"active-set method did not converge in {iteration_limit} iterations")  # noqa: TRY003
