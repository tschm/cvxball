"""The shipped active-set method, with its factorisation maintained across pivots.

An experiment, not a replacement. :func:`cvxball.min_circle_active_set` rebuilds
the linear algebra of its support from scratch every iteration; this asks what
that costs, by carrying an economic ``QR`` of the edge matrix instead and
repairing it as points enter and leave -- the same treatment
:mod:`experiments.fischer_gaertner_kutz` gives the pivoting method, and for the
same reason: at ``d`` in the thousands the refactorisation is most of the run.

It lives here rather than in ``src/`` because the updates come from
``scipy.linalg``, and NumPy is the shipped package's only runtime dependency.
That is the trade this module exists to price. Nothing below changes the
*algorithm* -- same dual, same active set, same pivots -- only where its
arithmetic comes from.

**The algebra that makes it cheap.** With ``D`` the ``r x d`` matrix of edges
``q_j - q_0`` and ``D' = QR`` its economic factorisation, three things follow,
and together they take every step off ``d``:

- ``D D' = R'R``. The Gram matrix the subproblem solves against is already
  factored: ``R`` *is* its Cholesky factor, so the circumcentre costs two
  triangular solves of order ``r`` instead of forming an ``O(r^2 d)`` product.
- ``null(D') = null(R)``. The affine-dependence test reads the ``r x r`` block,
  not the ``r x d`` edge matrix. Note it is the *right* null space of ``R`` --
  the left factor, which is what the edge matrix hands you, is the wrong one.
- ``||q_j - q_0||^2 = ||R[:,j]||^2``. Even the right-hand side comes from ``R``.

So an iteration costs ``O(r^3)`` on the small block plus ``O(r d)`` to repair the
factorisation, against ``O(r^2 d)`` to rebuild it. With ``r`` in the hundreds and
``d`` in the thousands that is the whole difference.

Pass ``dynamic=False`` to refactorise every iteration instead. The answers are
the same and it is the baseline the maintained version is measured against.
"""

import numpy as np
from scipy.linalg import qr_delete, qr_insert, qr_update, solve_triangular

from cvxball.solver import (
    _DROP_TOL,
    _FEAS_NOISE,
    _FEAS_RTOL,
    _MAX_ITER_PER_POINT,
    _sq_dist,
    _validate,
)

_EPS = float(np.finfo(np.float64).eps)


class _EdgeFrame:
    """The support set, and the economic ``QR`` of ``D'``, kept in step.

    The three updates are the ones :class:`experiments.fischer_gaertner_kutz._Frame`
    uses, because the two methods move their supports the same way: a point joins
    (append a column), a non-origin point leaves (delete one), or the origin
    leaves -- which has no column of its own, every column being measured *from*
    it, so ``q_1`` is promoted and the remaining edges ``a_j - a_1`` come from a
    deletion plus a rank-one update.
    """

    def __init__(self, points: np.ndarray, seed: int, dynamic: bool) -> None:
        """Start from the one-point support ``{seed}``, whose edge matrix is empty.

        Args:
            points: The ``(n, d)`` cloud, already centred and scaled.
            seed: Index of the point the support starts as.
            dynamic: Maintain the factorisation rather than rebuilding it.
        """
        self._points = points
        self._dynamic = dynamic
        self.support: list[int] = [seed]
        # How often an update was impossible and a refactorisation stood in. This
        # is the price of permitting affine dependence, and it is worth reporting
        # rather than hiding: if it is not rare, the data structure is not paying.
        self.fallbacks = 0
        self._q: np.ndarray = np.zeros((points.shape[1], 0))
        self._r: np.ndarray = np.zeros((0, 0))

    @property
    def face(self) -> np.ndarray:
        """Return the ``(m, d)`` array of support points."""
        return self._points[self.support]

    def _refactorise(self) -> None:
        """Rebuild ``Q`` and ``R`` from the current support."""
        face = self.face
        edges = (face[1:] - face[0]).T
        if edges.shape[1] == 0:
            self._q = np.zeros((edges.shape[0], 0))
            self._r = np.zeros((0, 0))
        else:
            self._q, self._r = np.linalg.qr(edges)

    def _economise(self) -> None:
        """Trim ``Q`` and ``R`` back to the economic shape after a scipy update.

        The updates leave ``Q`` as wide as it was, so deleting a column from a
        square factorisation returns a ``(d, d)`` ``Q`` beside a rectangular
        ``R``. ``R`` is upper triangular, so the surplus is zero and dropping it
        leaves ``QR = D'`` exactly -- but left in place it would make ``Q Q'`` the
        identity and every projection meaningless.
        """
        columns = len(self.support) - 1
        rows = min(self._points.shape[1], columns)
        if self._q.shape[1] != rows or self._r.shape != (rows, columns):
            self._q = self._q[:, :rows]
            self._r = self._r[:rows, :columns]

    def insert(self, index: int) -> None:
        """Take ``points[index]`` into the support.

        Args:
            index: Index into the cloud of the entering point.
        """
        column = self._points[index] - self.face[0]
        self.support.append(index)
        if not self._dynamic:
            self._refactorise()
        elif self._r.shape[0] == 0:
            length = float(np.linalg.norm(column))
            self._q = (column / length)[:, None]
            self._r = np.array([[length]])
        else:
            try:
                self._q, self._r = qr_insert(self._q, self._r, column, self._r.shape[1], which="col")
            except np.linalg.LinAlgError:
                # The entering point lies in aff(T), so the new column is already
                # in the span of Q and no economic factorisation can hold it. This
                # is the one place the two methods genuinely part company: the
                # pivoting method forbids the situation with its stability
                # threshold, while this method *permits* affine dependence and
                # answers it with the null-space descent step. So there is nothing
                # to update -- rebuild, let R come back singular, and let
                # `null_space` find the direction that shrinks the support.
                self.fallbacks += 1
                self._refactorise()
            else:
                self._economise()

    def remove(self, position: int) -> None:
        """Drop the support point at ``position``.

        Args:
            position: Index within the support list. Position 0 is the origin and
                takes the re-origining path.
        """
        if not self._dynamic:
            del self.support[position]
            self._refactorise()
            return

        if position > 0:
            del self.support[position]
            try:
                self._q, self._r = qr_delete(self._q, self._r, position - 1, which="col")
            except np.linalg.LinAlgError:
                self.fallbacks += 1
                self._refactorise()
            else:
                self._economise()
            return

        first_edge = self._points[self.support[1]] - self.face[0]
        remaining = len(self.support) - 2
        del self.support[0]
        if remaining <= 0:
            self._q = np.zeros((self._points.shape[1], 0))
            self._r = np.zeros((0, 0))
            return
        try:
            self._q, self._r = qr_delete(self._q, self._r, 0, which="col")
            rows = min(self._points.shape[1], remaining)
            self._q = self._q[:, :rows]
            self._r = self._r[:rows, :remaining]
            self._q, self._r = qr_update(self._q, self._r, -first_edge, np.ones(remaining))
        except (np.linalg.LinAlgError, ValueError):
            self.fallbacks += 1
            self._refactorise()
        else:
            self._economise()

    def null_space(self) -> np.ndarray:
        """Return weight directions that reshuffle the support without moving the centre.

        The shipped solver reads these off the left singular vectors of the edge
        matrix. Here they are ``null(R)``, which is the same subspace and costs
        ``O(r^3)`` rather than ``O(r^2 d)`` -- but it is the *right* null space,
        so the factor to read is ``Vh``, not ``U``.

        Returns:
            An ``(m, q)`` array of weight directions, with ``q == 0`` exactly when
            the support is affinely independent.
        """
        if self._r.shape[0] == 0:
            return np.zeros((1, 0))

        _, singular_values, right = np.linalg.svd(self._r, full_matrices=True)
        cutoff = max(self._r.shape[0], self._points.shape[1]) * _EPS * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > cutoff))
        tail = right[rank:, :].T
        return np.vstack([-tail.sum(axis=0, keepdims=True), tail])

    def circumcentre_weights(self) -> np.ndarray:
        """Solve the subproblem: put every support point on a common sphere.

        The shipped solver forms ``D D'`` and factors it. Here ``D D' = R'R`` is
        already factored, so this is two triangular solves and no contact with
        ``d`` at all -- the right-hand side ``||q_j - q_0||^2 / 2`` is the squared
        column norms of ``R``.

        Returns:
            The ``(m,)`` barycentric weights of the circumcentre, summing to one.
        """
        if self._r.shape[0] == 0:
            return np.ones(1)
        if self._r.shape[0] != self._r.shape[1]:
            # Only reachable on a dependent support, which `null_space` catches
            # first -- the caller never asks here.
            raise ValueError("circumcentre_weights needs an affinely independent support")  # noqa: TRY003

        rhs = 0.5 * np.einsum("ij,ij->i", self._r.T, self._r.T)
        forward = solve_triangular(self._r, rhs, lower=False, trans="T")
        tail = solve_triangular(self._r, forward, lower=False)
        return np.concatenate(([1.0 - float(tail.sum())], tail))


def _drop_positions(frame: _EdgeFrame, keep: np.ndarray) -> None:
    """Remove every support position whose weight fell to zero.

    Args:
        frame: The frame to shrink, in place.
        keep: Boolean mask over the current support; ``False`` entries go.
    """
    for position in sorted(np.flatnonzero(~keep).tolist(), reverse=True):
        frame.remove(position)


def min_circle_active_set_qr(
    points: np.ndarray,
    dynamic: bool = True,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing ball, maintaining the support factorisation.

    Step for step this is :func:`cvxball.min_circle_active_set`: the same dual QP
    over the simplex, the same support set, the same drop and add rules. Only the
    linear algebra differs, and only in where it comes from.

    Args:
        points: A ``(n, d)`` array with ``n >= 1``.
        dynamic: Repair the factorisation across pivots. ``False`` rebuilds it
            each iteration, which is what the shipped solver does.
        verbose: If ``True``, print the support size and radius per iteration.

    Returns:
        A tuple ``(radius, center)``, matching :func:`cvxball.min_circle_active_set`.

    Raises:
        ValueError: If ``points`` is not a finite, non-empty ``(n, d)`` array, or
            if the iteration limit is reached without the certificate holding.
    """
    pts = _validate(points)
    n = pts.shape[0]

    shift = pts.mean(axis=0)
    pts = pts - shift
    largest = float(np.abs(pts).max())
    if largest == 0.0:
        return 0.0, shift
    exponent = int(np.frexp(largest)[1])
    pts = np.ldexp(pts, -exponent)
    sq_norms: np.ndarray = np.einsum("ij,ij->i", pts, pts)

    seed = int(np.argmax(_sq_dist(pts, pts[0])))
    noise_floor = _FEAS_NOISE * _EPS * float(sq_norms.max())
    frame = _EdgeFrame(pts, seed, dynamic)
    weights = np.ones(1)

    iteration_limit = _MAX_ITER_PER_POINT * (n + 1)
    for iteration in range(iteration_limit):
        face = frame.face
        null_space = frame.null_space()

        if null_space.size:
            centre = face.T @ weights
            gradient = -_sq_dist(face, centre)
            descent = -(null_space @ (null_space.T @ gradient))
            scale = float(np.abs(descent).max())
            step = descent / scale if scale > 0.0 else descent
        else:
            target = frame.circumcentre_weights()
            if target.min() >= -_DROP_TOL:
                weights = np.maximum(target, 0.0)
                weights /= weights.sum()
                centre = face.T @ weights
                dist_sq = _sq_dist(pts, centre)
                radius_sq = float(dist_sq[frame.support].max())
                radius = float(np.sqrt(max(radius_sq, 0.0)))
                if verbose:
                    print(f"[{iteration:4d}] support={len(frame.support):3d} radius={radius:.12g}")

                worst = int(np.argmax(dist_sq))
                if dist_sq[worst] <= radius_sq * (1.0 + _FEAS_RTOL) + noise_floor:
                    return float(np.ldexp(radius, exponent)), np.ldexp(centre, exponent) + shift

                keep = weights > _DROP_TOL
                _drop_positions(frame, keep)
                frame.insert(worst)
                weights = np.append(weights[keep], 0.0)
                continue

            step = target - weights

        binding = step < -_DROP_TOL
        ratios = np.where(binding, -weights / np.where(binding, step, -1.0), np.inf)
        alpha = float(ratios.min())
        if not np.isfinite(alpha):
            raise ValueError("active-set method stalled: no support point blocks the step")  # noqa: TRY003
        weights = np.maximum(weights + alpha * step, 0.0)

        keep = weights > _DROP_TOL
        _drop_positions(frame, keep)
        weights = weights[keep]
        weights /= weights.sum()

    raise ValueError(f"active-set method did not converge in {iteration_limit} iterations")  # noqa: TRY003
