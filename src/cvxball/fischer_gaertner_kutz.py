"""The Fischer-Gärtner-Kutz pivoting method, the second solver this package ships.

Kaspar Fischer, Bernd Gärtner and Martin Kutz, *Fast Smallest-Enclosing-Ball
Computation in High Dimensions*, ESA 2003, LNCS 2832, 630-641.

:func:`min_circle_fgk` answers the same question as
:func:`cvxball.min_circle_active_set`, to the same exactness, and reaches it from
the opposite side. The active-set method works on the *dual* -- a QP over the unit
simplex, whose iterates are weights -- and its ball encloses the cloud only once
the KKT test passes. This method is *primal-feasible throughout*: every iterate is
an honest enclosing ball, and the algorithm deflates it. Hopp and Reeve proposed
the same picture as a heuristic; the contribution of Fischer et al. is the proof
of termination, and the pivot rule that makes it hold under degeneracy.

**Which one to call.** Either: they agree on the ball and on the support set, and
on Gaussian clouds from ``d = 1000`` to ``d = 16000`` they are within a factor of
1.1 to 1.6 in time (``experiments/bench_seb.py``). The active-set
method is the default because it is the faster of the two on every row measured
and returns the dual weights, which are a certificate the caller can check in one
pass. Call this one when a *feasible* ball matters before convergence -- its
iterates enclose the cloud and its radius falls monotonically, where the
active-set radius rises to the answer from below and its ball encloses nothing
until the last iteration -- or when the support set and pivot counts of
:func:`ball_with_counts` are what you are after.

The state is a pair ``(c, T)`` carrying the invariant (Fig. 2 of the paper)::

    B(c, T) contains S,   T is on the boundary of B(c, T),   T affinely independent

where ``B(c, T)`` is the ball about ``c`` through the farthest point of ``T``.
By Lemma 1 (Seidel) the ball is optimal exactly when ``c = cc(T)``, the
circumcentre of ``T``, *and* ``c`` lies in the convex hull of ``T`` -- so the
algorithm loops until ``c`` is in ``conv(T)``, each iteration being a *dropping*
phase (only when ``c`` has landed in ``aff(T)``) followed by a *walking* phase:

- **Dropping.** ``c`` in ``aff(T)`` but not in ``conv(T)`` means some affine
  coefficient of ``c`` with respect to ``T`` is negative. Drop such a point.
  Lemmata 4 and 5 together say the dropped point cannot immediately stop the walk
  that follows, which is what stops the two phases undoing each other.
- **Walking.** Move ``c`` along the straight line towards ``cc(T)``. Lemma 3 says
  that segment is orthogonal to ``aff(T)``, that ``T`` stays on the boundary all
  the way along it, and that the radius strictly decreases -- the deflation. Stop
  early at the first point of ``S`` to hit the shrinking boundary, and take it
  into ``T``; otherwise arrive at ``cc(T)``.

Two things follow from Lemma 3 that this module leans on, both verified
numerically before they were relied upon:

- Because ``[c, cc(T)]`` is orthogonal to ``aff(T)``, the circumcentre *is* the
  orthogonal projection of ``c`` onto ``aff(T)`` (paper, section 4). So no
  circumsphere is ever solved for -- one projection does the job, which is the
  reformulation the paper's own implementation uses.
- The stopping fraction has a closed form. Walking ``c(a) = c + a e`` with
  ``e = cc(T) - c``, a point ``p`` joins the boundary when ``||p - c(a)||`` equals
  the common support distance; the ``a^2`` terms cancel and, using
  ``<q - c, e> = ||e||^2`` for every ``q`` in ``T``, that leaves

      a_p = (R^2 - ||p - c||^2) / (2 (||e||^2 - <p - c, e>)).

  Lemma 4's criterion (1) -- ``p`` is "behind" ``aff(T)`` and so uncritical -- is
  exactly the statement that this denominator is non-positive, so one sign test
  separates the candidates and the numerator is then non-negative by the
  invariant. The walk takes ``a = min(1, min_p a_p)``.

**Scope.** Both halves of the paper are here: the pivoting algorithm of Fig. 2,
and section 4's dynamic QR-decomposition that makes it fast. :class:`_Frame`
carries ``Q`` and ``R`` for the edge matrix ``A = [q_1 - q_0, ...]`` across pivots
and repairs them in ``O(d r)`` as points enter and leave, rather than
refactorising in ``O(d r^2)``; ``dynamic_qr=False`` selects the rebuild instead,
which gives the same answers and is the baseline that says what the data
structure is worth.

Implementing it at all is only worthwhile because ``scipy.linalg`` exposes the
updates -- ``qr_insert``, ``qr_delete``, ``qr_update`` -- as compiled
LAPACK-backed routines. Written as a Python loop the Givens sweeps would lose to a
single vectorised ``numpy.linalg.qr``, and a measurement of this method against
the shipped one would then be a measurement of the interpreter. Those three
routines are already why SciPy is a dependency of this package, for
:mod:`cvxball._frame`, so this module adds no import that ``import cvxball`` did
not already pull in.

Both of the paper's pivot rules are here, selected by ``pivot_rule``:

- ``"bland"`` -- Bland's rule adapted to this setting: fix an order on ``S``, drop
  the negative-coefficient point of smallest rank, and admit the smallest-rank
  point when several stop the walk at once. Theorem 1 proves termination with it,
  degeneracies included.
- ``"heuristic"`` (the default, and what the paper's own code runs) -- drop the
  point of *minimal* coefficient, and among points that stop the walk at
  effectively the same place admit the one farthest from ``aff(T)``. The paper
  reports Bland's rule as correct but slow, and this as both faster and more
  robust to roundoff.

Like the shipped solver, and unlike the paper's fixed epsilon, this is written to
be scale- and origin-invariant: the cloud is recentred on its mean before the
first iteration, and the one tolerance carrying units -- the margin a point must
clear from ``aff(T)`` before it is allowed into the support, the paper's stability
threshold -- is sized off the cloud's extent. The coefficient tolerance needs no
such treatment, being dimensionless and bounded by Lemma 2 (every coefficient of a
centre in ``conv(T)`` is at most 1/2).
"""

from typing import Literal, NamedTuple

import numpy as np

# The three update routines are re-exported from a Cython extension, so the type
# stubs do not declare them on `scipy.linalg` even though they are there at run
# time -- the same suppression `cvxball._frame` needs, for the same three names.
from scipy.linalg import (
    qr_delete,  # ty: ignore[unresolved-import]
    qr_insert,  # ty: ignore[unresolved-import]
    qr_update,  # ty: ignore[unresolved-import]
    solve_triangular,
)

from cvxball.solver import _validate

# How far a point must sit from aff(T), relative to the cloud's extent, before it
# may enter the support. This is the paper's stability threshold (section 4): it
# is what keeps T affinely independent in floating point, since a point *on*
# aff(T) would make the edge matrix rank-deficient and the next projection
# meaningless. Points behind aff(T) are discarded anyway, by Lemma 4.
_AFF_RTOL = 1e-13
# Affine coefficients are dimensionless and, by Lemma 2, at most 1/2 in the
# optimal configuration -- so this needs no scaling, unlike the margin above.
_COEFF_TOL = 1e-13
# The smallest angle, as a reciprocal condition number, at which a point may join
# the support. `qr_insert` refuses anything below roughly machine epsilon here;
# this sits far enough above that the update never has to be second-guessed.
_RCOND_FLOOR = 1e-12
# Two stopping points count as tied, and so as competing under the pivot rule,
# when their stopping fractions agree to this much.
_TIE_RTOL = 1e-10
# Safety net only. Termination is a theorem under Bland's rule, so reaching this
# means numerical trouble rather than a slow instance.
_MAX_ITER_PER_POINT = 50

PivotRule = Literal["heuristic", "bland"]


class Ball(NamedTuple):
    """A ball, plus the work that went into finding it.

    Attributes:
        radius: The radius of the enclosing ball.
        centre: The centre, of shape ``(d,)``.
        support: Indices into the input of the final support set ``T``, whose
            circumsphere is the ball. At most ``d + 1`` of them.
        iterations: How many turns of the main loop ran.
        drops: How many points left the support set.
        insertions: How many points entered it.
    """

    radius: float
    centre: np.ndarray
    support: np.ndarray
    iterations: int
    drops: int
    insertions: int


class _Frame:
    """The support set, and the economic ``QR`` of its edge matrix, kept in step.

    This is section 4 of the paper: rather than refactorise the edge matrix
    ``A = [q_1 - q_0, ..., q_r - q_0]`` from scratch at every pivot, carry ``Q``
    and ``R`` and repair them as one point enters or leaves. Rebuilding costs
    ``O(d r^2)``; each repair costs ``O(d r)``, and with ``r`` in the hundreds at
    the dimensions this method is built for, that is the difference the paper's
    engineering is about.

    Three updates cover every move the algorithm makes, and only the third needs
    thought:

    - **A point joins ``T``.** One column is appended: ``qr_insert``.
    - **A point other than the origin leaves.** One column is dropped:
      ``qr_delete``.
    - **The origin leaves.** No single column corresponds to ``q_0``, since every
      column is measured *from* it, so dropping it changes them all. Promote
      ``q_1`` to origin and the new edges are ``a_j - a_1``: delete column ``a_1``,
      then subtract it from what remains, which is a rank-one update. Two compiled
      calls, both ``O(d r)``, which is the "appropriate rank-1-update" the paper
      mentions without spelling out.

    All three come from ``scipy.linalg``, so the Givens sweeps run compiled. That
    is what makes reproducing section 4 worthwhile here: written as a Python loop
    they would lose to a single vectorised ``numpy.linalg.qr``, and the comparison
    would measure the interpreter rather than the data structure. Pass
    ``dynamic=False`` to get exactly that rebuild-every-pivot behaviour, which is
    what the two are measured against each other with.
    """

    def __init__(self, points: np.ndarray, origin: int, dynamic: bool) -> None:
        """Start from the one-point support ``{origin}``, whose edge matrix is empty.

        Args:
            points: The ``(n, d)`` cloud, in centred coordinates.
            origin: Index of the single point the support starts as.
            dynamic: Maintain ``Q`` and ``R`` across changes, rather than
                refactorising after each one.
        """
        self._points = points
        self._dynamic = dynamic
        self.support = [origin]
        self.rebuilds = 0
        # Updates that could not be applied and were refactorised instead. The
        # stability threshold makes a dependent insert rare but does not make it
        # impossible: it bounds the entering point's distance from aff(T) from
        # below, which is a weaker statement than scipy's condition-number test.
        self.fallbacks = 0
        self._q: np.ndarray = np.zeros((points.shape[1], 0))
        self._r: np.ndarray = np.zeros((0, 0))

    @property
    def origin(self) -> np.ndarray:
        """Return ``q_0``, the point every edge is measured from."""
        point: np.ndarray = self._points[self.support[0]]
        return point

    @property
    def basis(self) -> np.ndarray:
        """Return ``Q``: orthonormal columns spanning the direction space of ``aff(T)``."""
        return self._q

    def _refactorise(self) -> None:
        """Rebuild ``Q`` and ``R`` from the current support, from scratch."""
        self.rebuilds += 1
        face = self._points[self.support]
        edges = (face[1:] - face[0]).T
        if edges.shape[1] == 0:
            self._q = np.zeros((edges.shape[0], 0))
            self._r = np.zeros((0, 0))
        else:
            self._q, self._r = np.linalg.qr(edges)

    def _economise(self) -> None:
        """Trim ``Q`` and ``R`` back to the economic shape after an update.

        scipy's updates leave ``Q`` as wide as it was, so deleting a column from a
        square factorisation returns ``Q`` of shape ``(d, d)`` beside an ``R`` of
        shape ``(d, r-1)`` -- no longer the economic pair. Both consumers break on
        that, and one of them breaks *silently*: with ``Q`` square, ``Q Q'`` is the
        identity, so the projection onto ``aff(T)`` would come back as the point
        itself and the walk direction would collapse to zero, reading as "already
        on the hull" at every subsequent pivot.

        ``R`` is upper triangular, so its surplus rows are zero and the surplus
        columns of ``Q`` multiply nothing: dropping them leaves ``QR = A`` exactly.
        """
        columns = len(self.support) - 1
        if self._q.shape[1] != columns or self._r.shape != (columns, columns):
            self._q = self._q[:, :columns]
            self._r = self._r[:columns, :columns]

    def insert(self, index: int) -> None:
        """Take ``points[index]`` into the support, appending one edge column.

        Args:
            index: Index into the cloud of the entering point.
        """
        column = self._points[index] - self.origin
        self.support.append(index)
        if not self._dynamic:
            self._refactorise()
        elif self._r.shape[0] == 0:
            # The first edge: its economic QR is just the normalised column, and
            # scipy has no zero-column factorisation to insert into.
            length = float(np.linalg.norm(column))
            self._q = (column / length)[:, None]
            self._r = np.array([[length]])
        else:
            try:
                self._q, self._r = qr_insert(self._q, self._r, column, self._r.shape[1], which="col")
            except np.linalg.LinAlgError:
                # The column is already in the span of Q, so no economic
                # factorisation can hold it. Rebuilding always can: numpy's QR
                # accepts a rank-deficient matrix and simply returns a singular R.
                self.fallbacks += 1
                self._refactorise()
            else:
                self._economise()

    def remove(self, position: int) -> None:
        """Drop the support point at ``position``, repairing the factorisation.

        Args:
            position: Index *within the support list*, not into the cloud.
                Position 0 is the origin, and takes the re-origining path.
        """
        if not self._dynamic:
            del self.support[position]
            self._refactorise()
            return

        if position > 0:
            # An ordinary column, sitting at position - 1 of the edge matrix.
            del self.support[position]
            try:
                self._q, self._r = qr_delete(self._q, self._r, position - 1, which="col")
            except np.linalg.LinAlgError:
                self.fallbacks += 1
                self._refactorise()
            else:
                self._economise()
            return

        # The origin leaves, so q_1 is promoted and every edge is re-measured from
        # it. Deleting a_1's column leaves [a_2, ..., a_r]; the rank-one update
        # then turns those into [a_2 - a_1, ..., a_r - a_1].
        first_edge = self._points[self.support[1]] - self.origin
        remaining = self._r.shape[0] - 1
        if remaining == 0:
            self._q = np.zeros((self._points.shape[1], 0))
            self._r = np.zeros((0, 0))
            del self.support[0]
        else:
            del self.support[0]
            try:
                self._q, self._r = qr_delete(self._q, self._r, 0, which="col")
                self._q = self._q[:, :remaining]
                self._r = self._r[:remaining, :remaining]
                self._q, self._r = qr_update(self._q, self._r, -first_edge, np.ones(remaining))
            except (np.linalg.LinAlgError, ValueError):
                self.fallbacks += 1
                self._refactorise()
            else:
                self._economise()

    def admits(self, index: int) -> bool:
        """Report whether ``points[index]`` can join the support safely.

        The paper's stability threshold is absolute -- it asks that the entering
        point sit some distance from ``aff(T)`` measured against the cloud's
        extent. A factorisation cares about something else: the *angle*, i.e. that
        distance relative to the entering column's own length, which is exactly
        the reciprocal condition number ``qr_insert`` tests. The two disagree on a
        cloud carrying a far outlier, where a point can clear the absolute margin
        and still be numerically inside the span.

        Testing what the factorisation tests is what keeps the two in step. It
        matters more here than it would for the shipped solver, because this
        algorithm has no answer to a dependent support -- Fig. 2's invariant
        requires ``T`` affinely independent, and there is no null-space descent
        step to fall back on.

        Args:
            index: Index into the cloud of the candidate point.

        Returns:
            ``True`` when the candidate is safely off ``aff(T)``.
        """
        offset = self._points[index] - self.origin
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return False
        residual = offset - self._q @ (self._q.T @ offset)
        return bool(float(np.linalg.norm(residual)) / length > _RCOND_FLOOR)

    def direction_to_circumcentre(self, centre: np.ndarray) -> np.ndarray:
        """Return ``cc(T) - centre``, the walking direction.

        By Lemma 3(i) that segment is orthogonal to ``aff(T)``, so the
        circumcentre is the orthogonal projection of ``centre`` onto the hull and
        ``Q Q'`` is all that is needed to find it.

        Args:
            centre: The current centre.

        Returns:
            The ``(d,)`` step from ``centre`` to the circumcentre of the support.
        """
        offset = centre - self.origin
        step: np.ndarray = self._q @ (self._q.T @ offset) - offset
        return step

    def coefficients(self, centre: np.ndarray) -> np.ndarray:
        """Express ``centre`` as an affine combination of the support.

        The paper's route, verbatim: solve ``R x = Q' (centre - q_0)`` by back
        substitution. The entries of ``x`` are the coefficients of ``q_1, ..., q_r``
        and the missing one for ``q_0`` follows from their summing to one.

        Meaningful only when ``centre`` lies in ``aff(T)``, which the caller
        guarantees by asking solely after a walk has run to completion.

        Args:
            centre: The current centre.

        Returns:
            The ``(m,)`` coefficients. A negative entry certifies that ``centre``
            lies outside ``conv(T)``, which is what drives the drop.
        """
        if self._r.shape[0] == 0:
            return np.ones(1)
        rhs = self._q.T @ (centre - self.origin)
        try:
            tail = solve_triangular(self._r, rhs, lower=False)
        except np.linalg.LinAlgError:
            # A support can drift into near-dependence despite the stability
            # threshold, and a *rebuilt* factorisation then puts an exact zero on
            # R's diagonal where the incrementally updated one keeps it merely
            # small -- so this fires on `dynamic=False` and not on the maintained
            # factorisation. The least-squares solution agrees with back
            # substitution wherever the system is solvable at all, so falling back
            # costs nothing and keeps the rebuild baseline usable as a comparison.
            tail = np.linalg.lstsq(self._r, rhs, rcond=None)[0]
        return np.concatenate(([1.0 - float(tail.sum())], tail))


def _leaving_point(coefficients: np.ndarray, support: list[int], pivot_rule: PivotRule) -> int | None:
    """Choose which support point to drop, or report that the ball is optimal.

    Args:
        coefficients: The affine coefficients of the centre with respect to the
            support, as returned by :meth:`_Frame.coefficients`.
        support: The indices currently in the support, in insertion order.
        pivot_rule: ``"bland"`` takes the negative-coefficient point of smallest
            rank in the fixed order on ``S`` -- here the index into the input,
            which is the arbitrary order Theorem 1 asks us to fix.
            ``"heuristic"`` takes the most negative coefficient.

    Returns:
        The *position within* ``support`` of the point to drop, or ``None`` when
        every coefficient is non-negative -- the centre is then in ``conv(T)``,
        which by Lemma 1 is the optimality certificate.
    """
    negative = np.flatnonzero(coefficients < -_COEFF_TOL)
    if negative.size == 0:
        return None
    if pivot_rule == "bland":
        return int(negative[np.argmin(np.asarray(support)[negative])])
    return int(negative[np.argmin(coefficients[negative])])


def _entering_point(
    fractions: np.ndarray,
    shortest: float,
    points: np.ndarray,
    frame: _Frame,
    pivot_rule: PivotRule,
) -> int:
    """Choose which point stopping the walk should enter the support.

    Args:
        fractions: The ``(n,)`` stopping fractions, ``inf`` where a point cannot
            stop the walk at all.
        shortest: The smallest of them, the distance actually walked.
        points: The ``(n, d)`` cloud, in centred coordinates.
        frame: The current support and its factorisation.
        pivot_rule: ``"bland"`` takes the smallest index among the points that
            stop the walk at the same place, as Theorem 1 requires.
            ``"heuristic"`` takes the one farthest from ``aff(T)``, the paper's
            roundoff-motivated choice: the farther the new point is from the
            hull it is joining, the better conditioned the enlarged support.

    Returns:
        The index into ``points`` of the entering point.
    """
    tied = np.flatnonzero(fractions <= shortest + _TIE_RTOL * max(abs(shortest), 1.0))
    if tied.size == 1 or pivot_rule == "bland":
        return int(tied[0])

    basis = frame.basis
    offsets = points[tied] - frame.origin
    residuals = offsets - (offsets @ basis) @ basis.T
    return int(tied[np.argmax(np.einsum("ij,ij->i", residuals, residuals))])


def ball_with_counts(
    points: np.ndarray,
    pivot_rule: PivotRule = "heuristic",
    dynamic_qr: bool = True,
    verbose: bool = False,
) -> Ball:
    """Solve the smallest enclosing ball by the pivoting method, reporting the work done.

    Args:
        points: A ``(n, d)`` array with ``n >= 1``.
        pivot_rule: ``"heuristic"`` (the paper's own code) or ``"bland"`` (the
            rule Theorem 1 proves terminating).
        dynamic_qr: Carry the factorisation across pivots, repairing it in
            ``O(d r)`` as section 4 does. ``False`` refactorises from scratch at
            every pivot, in ``O(d r^2)`` -- the same answers, and the baseline the
            data structure is worth measuring against.
        verbose: If ``True``, print the phase, support size and radius per turn.

    Returns:
        The :class:`Ball`, including the support set and the pivot counts.

    Raises:
        ValueError: If ``points`` is not a finite, non-empty ``(n, d)`` array (see
            :func:`cvxball.solver._validate`), if ``pivot_rule`` is not one of the
            two rules, or if the iteration limit is reached -- which, termination
            being a theorem, means numerical trouble rather than a hard instance.
    """
    if pivot_rule not in ("heuristic", "bland"):
        raise ValueError(f"pivot_rule must be 'heuristic' or 'bland', got {pivot_rule!r}")  # noqa: TRY003

    pts = _validate(points)
    n = pts.shape[0]

    # Recentre, so that every tolerance below is governed by the extent of the
    # cloud rather than by its distance from an arbitrary origin, and difference
    # before squaring for the same reason the shipped solver does.
    shift = pts.mean(axis=0)
    pts = pts - shift
    extent = float(np.sqrt(np.einsum("ij,ij->i", pts, pts).max()))
    if extent == 0.0:
        # Every point is the same point; the ball is that point, radius zero.
        return Ball(0.0, shift.copy(), np.zeros(1, dtype=np.intp), 0, 0, 0)
    margin = _AFF_RTOL * extent

    # Initialisation, from Fig. 2: c is any point of S, and T the single point of
    # S farthest from it -- which is what makes B(c, T) enclose S to begin with.
    centre = pts[0].copy()
    offsets = pts - centre
    frame = _Frame(pts, int(np.argmax(np.einsum("ij,ij->i", offsets, offsets))), dynamic_qr)
    support = frame.support
    # Whether c is known to lie in aff(T). It is not, initially: c is one point of
    # the cloud and aff(T) is a different one. It becomes true exactly when a walk
    # runs to completion, since the centre is then the projection onto that hull.
    on_hull = False

    drops = insertions = 0
    limit = _MAX_ITER_PER_POINT * (n + 1)
    for iteration in range(limit):
        if on_hull:
            leaving = _leaving_point(frame.coefficients(centre), support, pivot_rule)
            if leaving is None:
                # c is in conv(T): by Lemma 1 this ball is SEB(S).
                radius = float(np.linalg.norm(pts[support] - centre, axis=1).max())
                if verbose:
                    print(f"[{iteration:4d}] optimal   support={len(support):3d} radius={radius:.12g}")
                return Ball(radius, centre + shift, np.array(support, dtype=np.intp), iteration, drops, insertions)
            if verbose:
                print(f"[{iteration:4d}] drop      support={len(support):3d} point={support[leaving]}")
            frame.remove(leaving)
            drops += 1
            on_hull = False
            continue

        # --- Walking phase ---------------------------------------------------
        # cc(T) is the orthogonal projection of c onto aff(T), by Lemma 3(i).
        direction = frame.direction_to_circumcentre(centre)
        direction_sq = float(direction @ direction)
        if np.sqrt(direction_sq) <= margin:
            # Already on the hull -- reached whenever the support has grown to
            # d + 1 points, whose affine hull is the whole space.
            on_hull = True
            continue

        offsets = pts - centre
        squared = np.einsum("ij,ij->i", offsets, offsets)
        radius_sq = float(squared[support].max())

        # Lemma 4: p can stop the walk only if <p - c, e> < <e, e>, i.e. only if
        # this denominator is positive. Requiring it to clear `margin * ||e||`
        # rather than zero is the stability threshold, and it is exactly the
        # right test: e is orthogonal to aff(T), so denom / ||e|| is p's distance
        # from aff(T) measured along e, and thus a lower bound on its full
        # distance from aff(T). Clearing the margin here therefore certifies that
        # T stays affinely independent when p joins it -- and the points it
        # excludes have a near-zero denominator, hence an enormous stopping
        # fraction, so they were never going to be the minimiser anyway.
        denominator = direction_sq - offsets @ direction
        stoppers = denominator > margin * np.sqrt(direction_sq)
        stoppers[support] = False

        fractions = np.full(n, np.inf)
        np.divide(
            np.maximum(radius_sq - squared, 0.0),
            2.0 * denominator,
            out=fractions,
            where=stoppers,
        )
        # Walk to the nearest stopper the support can actually take. A candidate
        # that fails `admits` is numerically inside aff(T): taking it would break
        # the affine independence Fig. 2's invariant rests on, so it is passed
        # over and the next-nearest considered. Usually the first one is fine.
        while True:
            shortest = float(fractions.min())
            if shortest >= 1.0:
                # Nothing stops the walk: the centre reaches cc(T) and lands on the hull.
                centre = centre + direction
                on_hull = True
                if verbose:
                    radius = float(np.linalg.norm(pts[support] - centre, axis=1).max())
                    print(f"[{iteration:4d}] arrive    support={len(support):3d} radius={radius:.12g}")
                break

            entering = _entering_point(fractions, shortest, pts, frame, pivot_rule)
            if not frame.admits(entering):
                fractions[entering] = np.inf
                continue

            centre = centre + shortest * direction
            frame.insert(entering)
            insertions += 1
            if verbose:
                radius = float(np.linalg.norm(pts[support] - centre, axis=1).max())
                print(f"[{iteration:4d}] insert    support={len(support):3d} radius={radius:.12g} step={shortest:.6g}")
            break

    raise ValueError(f"pivoting method did not converge in {limit} iterations")  # noqa: TRY003


def min_circle_fgk(points: np.ndarray, verbose: bool = False) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing ball, in this package's solver signature.

    Args:
        points: A numpy array of shape ``(n, d)`` where *n* is the number of
                points and *d* is the ambient dimension.
        verbose: If ``True``, print one line per pivot step. Defaults to ``False``.

    Returns:
        A tuple ``(radius, center)``, matching :func:`cvxball.min_circle_active_set`.

    Raises:
        ValueError: If ``points`` is not a finite, non-empty ``(n, d)`` array.

    Example:
        The right triangle whose smallest enclosing circle is the one on its
        hypotenuse.

        The values are rounded here, as they are for the cone program, and the
        reason is worth stating because it is easy to assume otherwise: this
        method terminates at an exact *combinatorial* configuration -- the
        support set ``{(1, 0), (0, 1)}`` -- but the centre it reports is not the
        exact circumcentre of that set. It is the running sum of the walks that
        got there, so it lands a few ulp out (four, on this input). The shipped
        active-set method solves afresh for the centre of its final support and
        so returns ``sqrt(2) / 2`` bit-for-bit; the difference is one of
        arithmetic, not of which ball the two methods identify.

        >>> import numpy as np
        >>> from cvxball import min_circle_fgk
        >>> radius, center = min_circle_fgk(np.array([[0, 0], [1, 0], [0, 1]]))
        >>> round(radius, 12)
        0.707106781187
        >>> np.round(center, 12)
        array([0.5, 0.5])
    """
    ball = ball_with_counts(points, verbose=verbose)
    if verbose:
        print(f"fgk: iterations={ball.iterations} drops={ball.drops} insertions={ball.insertions}")
    return ball.radius, ball.centre
