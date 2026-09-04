"""A support set whose factorisation is carried across iterations, not rebuilt.

:func:`cvxball.min_circle_active_set` changes its support by one point per
iteration, and every iteration needs two things from it: whether the support is
affinely independent, and where the circumcentre of its face lies. Recomputing
both from the raw points costs ``O(d r^2)`` for a support of ``r + 1`` points in
``d`` dimensions. Repairing a factorisation that already exists costs
``O(d r)``.

At the dimensions this module exists for, that is the whole running time. It is
also the entirety of why SciPy is a dependency of this package: ``qr_insert``,
``qr_delete`` and ``qr_update`` are the compiled Givens updates that make the
repair possible. Written as a Python loop they would lose to one vectorised
``numpy.linalg.qr``, and the exercise would be pointless.

**The algebra.** With ``D`` the ``r x d`` matrix of edges ``q_j - q_0`` and
``D' = QR`` its economic factorisation, three identities take every per-iteration
quantity off ``d`` and onto the small ``r x r`` block:

- ``D D' = R'R``, so ``R`` is already a Cholesky factor of the Gram matrix the
  circumcentre subproblem solves against: two triangular solves, no ``O(r^2 d)``
  product.
- ``null(D') = null(R)``, so the affine-dependence test reads ``R``. Note this is
  the *right* null space; the left factor, which the edge matrix itself would
  hand you, is the wrong one.
- ``||q_j - q_0||^2 = ||R[:,j]||^2``, so even the right-hand side comes from ``R``.

**Affine dependence is the awkward part**, and it is where this method differs
from the pivoting method of Fischer, Gärtner and Kutz. Theirs forbids a dependent
support outright; this one *permits* it and answers it with a descent direction
in the null space, which means the factorisation has to survive situations an
economic ``QR`` cannot represent: ``qr_insert`` refuses a column already in the
span, and a dependent support can outgrow ``d + 1`` and leave ``R`` rectangular.
Both are handled by refactorising, which ``numpy.linalg.qr`` accepts happily,
returning the singular ``R`` that :meth:`_MaintainedFace.null_space` then reads.
Such fallbacks are counted rather than hidden: if they stopped being rare the
data structure would have stopped paying.
"""

import numpy as np

# The three update routines are re-exported from a Cython extension, so the type
# stubs do not declare them on `scipy.linalg` even though they are there at run
# time. Importing them from the private module instead would be worse.
from scipy.linalg import (
    qr_delete,  # ty: ignore[unresolved-import]
    qr_insert,  # ty: ignore[unresolved-import]
    qr_update,  # ty: ignore[unresolved-import]
    solve_triangular,
)

_EPS = float(np.finfo(np.float64).eps)


class _MaintainedFace:
    """The support set, and the economic ``QR`` of its edge matrix, kept in step.

    Three updates cover every move the solver makes. A point joining the support
    appends a column, a point other than the origin leaving deletes one, and the
    origin leaving has no column of its own -- every column being measured *from*
    it -- so ``q_1`` is promoted and the remaining edges ``a_j - a_1`` come from a
    deletion plus a rank-one update.
    """

    def __init__(self, points: np.ndarray, seed: int) -> None:
        """Start from the one-point support ``{seed}``, whose edge matrix is empty.

        Args:
            points: The ``(n, d)`` cloud, already centred and scaled.
            seed: Index of the point the support starts as.
        """
        self._points = points
        self.support: list[int] = [seed]
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
        """Trim ``Q`` and ``R`` back to the economic shape after an update.

        SciPy's updates leave ``Q`` as wide as it was, so deleting a column from a
        square factorisation returns a ``(d, d)`` ``Q`` beside a rectangular
        ``R``. Left alone that breaks the triangular solve loudly and the
        projection silently -- with ``Q`` square, ``Q Q'`` is the identity.
        ``R`` is upper triangular, so the surplus is zero and dropping it leaves
        ``QR = D'`` exactly.
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
        if self._r.shape[0] == 0:
            length = float(np.linalg.norm(column))
            if length == 0.0:
                self._refactorise()
                return
            self._q = (column / length)[:, None]
            self._r = np.array([[length]])
            return
        try:
            self._q, self._r = qr_insert(self._q, self._r, column, self._r.shape[1], which="col")
        except np.linalg.LinAlgError:
            # The entering point lies in the affine hull of the support, so the
            # new column is already in the span of Q and no economic
            # factorisation can hold it. Rebuilding always can, and the singular
            # R that comes back is exactly what `null_space` needs to see.
            self.fallbacks += 1
            self._refactorise()
        else:
            self._economise()

    def remove(self, position: int) -> None:
        """Drop the support point at ``position``.

        Args:
            position: Index within the support list. Position 0 is the origin,
                which takes the re-origining path.
        """
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
        """Find weight directions that reshuffle the support without moving the centre.

        The same object :func:`cvxball.solver._affine_null_space` returns, read off
        the ``r x r`` block instead of the ``r x d`` edge matrix: ``null(D')`` is
        ``null(R)``, which is the *right* null space, so the factor to take is
        ``Vh`` rather than ``U``.

        Returns:
            An ``(m, q)`` array of weight directions, empty exactly when the
            support is affinely independent.
        """
        if self._r.shape[0] == 0:
            return np.zeros((1, 0))

        _, singular_values, right = np.linalg.svd(self._r, full_matrices=True)
        cutoff = max(self._r.shape[1], self._points.shape[1]) * _EPS * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > cutoff))
        tail = right[rank:, :].T
        return np.vstack([-tail.sum(axis=0, keepdims=True), tail])

    def circumcentre_weights(self) -> np.ndarray:
        """Solve the subproblem: put every support point on a common sphere.

        ``D D' = R'R`` is already factored, so this is two triangular solves and
        never touches ``d`` -- the right-hand side ``||q_j - q_0||^2 / 2`` is the
        squared column norms of ``R``.

        Returns:
            The ``(m,)`` weights, summing to one, that express the circumcentre as
            an affine combination of the support. A negative entry means the
            circumcentre lies outside the simplex.
        """
        if self._r.shape[0] == 0:
            return np.ones(1)

        rhs = 0.5 * np.einsum("ij,ij->i", self._r.T, self._r.T)
        forward = solve_triangular(self._r, rhs, lower=False, trans="T")
        tail = solve_triangular(self._r, forward, lower=False)
        return np.concatenate(([1.0 - float(tail.sum())], tail))
