"""The Fischer-Gärtner-Kutz pivoting method, as a reference implementation.

Kaspar Fischer, Bernd Gärtner and Martin Kutz, *Fast Smallest-Enclosing-Ball
Computation in High Dimensions*, ESA 2003, LNCS 2832, 630-641.

Like :mod:`experiments.welzl` and :mod:`experiments.clarabel_ball` this exists to
be *compared against* rather than used: :func:`cvxball.min_circle_active_set` is
the solver this package ships. What makes this one worth having is that it is the
closest published relative of the shipped method and yet reaches the answer from
the opposite side. The active-set method works on the *dual* -- a QP over the unit
simplex, whose iterates are weights -- and its ball encloses the cloud only once
the KKT test passes. This method is *primal-feasible throughout*: every iterate is
an honest enclosing ball, and the algorithm deflates it. Hopp and Reeve proposed
the same picture as a heuristic; the contribution of Fischer et al. is the proof
of termination, and the pivot rule that makes it hold under degeneracy.

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

**Scope.** This is the *algorithm* of Fig. 2, not the C++ engineering around it.
Section 4's dynamic QR-decomposition -- Givens rotations updating ``Q`` and ``R``
in place as a point enters or leaves ``T``, quadratic in ``d`` instead of the
cubic cost of rebuilding -- is deliberately not reproduced, and the QR here is
recomputed from scratch each iteration. That is a considered choice rather than a
shortcut: a Givens sweep written as a Python loop would be slower than one
vectorised LAPACK call to ``numpy.linalg.qr``, so implementing it would
*misrepresent* the paper's performance claim rather than demonstrate it. What is
kept is the paper's formulation of the linear algebra -- the projection and the
affine coefficients both read off a QR of the edge matrix ``A = [q_1 - q_0, ...]``
-- so the arithmetic per iteration is theirs even where its scheduling is not.

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
from scipy.linalg import solve_triangular

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


def _edge_frame(face: np.ndarray) -> np.ndarray:
    """Compute an orthonormal basis of the direction space of ``aff(face)``.

    This is the ``Q`` of the paper's ``QR = A`` for the edge matrix
    ``A = [q_1 - q_0, ..., q_r - q_0]``. Only ``Q`` is needed to project onto
    ``aff(face)``, since ``A x* = Q Q' b`` for the least-squares solution ``x*``;
    ``R`` is recovered by :func:`_affine_coefficients` when the coefficients
    themselves are wanted.

    Args:
        face: A ``(m, d)`` array of affinely independent points, ``m >= 1``.

    Returns:
        A ``(d, m - 1)`` array with orthonormal columns spanning the edge space.
        It has no columns when ``face`` is a single point, whose affine hull is
        that point alone.
    """
    edges = (face[1:] - face[0]).T
    basis: np.ndarray = np.linalg.qr(edges)[0]
    return basis


def _affine_coefficients(face: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Express ``centre`` as an affine combination of ``face``.

    The paper's route, verbatim: with ``QR = A`` for the edge matrix ``A``, solve
    ``R x = Q' (centre - q_0)`` by back substitution. The entries of ``x`` are the
    coefficients of ``q_1, ..., q_r``, and the missing coefficient of ``q_0``
    follows from the affine constraint that they sum to one.

    This is meaningful only when ``centre`` lies in ``aff(face)``, which the
    caller guarantees by only asking after a walk has run to completion -- the
    centre is then the projection onto that hull, by construction.

    Args:
        face: A ``(m, d)`` array of affinely independent points.
        centre: The ``(d,)`` point to express.

    Returns:
        The ``(m,)`` coefficients, summing to one. A negative entry certifies that
        ``centre`` lies outside ``conv(face)``, which is what drives the drop.
    """
    if face.shape[0] == 1:
        return np.ones(1)

    basis, upper = np.linalg.qr((face[1:] - face[0]).T)
    tail = solve_triangular(upper, basis.T @ (centre - face[0]), lower=False)
    return np.concatenate(([1.0 - float(tail.sum())], tail))


def _leaving_point(coefficients: np.ndarray, support: list[int], pivot_rule: PivotRule) -> int | None:
    """Choose which support point to drop, or report that the ball is optimal.

    Args:
        coefficients: The affine coefficients of the centre with respect to the
            support, as returned by :func:`_affine_coefficients`.
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
    face: np.ndarray,
    basis: np.ndarray,
    pivot_rule: PivotRule,
) -> int:
    """Choose which point stopping the walk should enter the support.

    Args:
        fractions: The ``(n,)`` stopping fractions, ``inf`` where a point cannot
            stop the walk at all.
        shortest: The smallest of them, the distance actually walked.
        points: The ``(n, d)`` cloud, in centred coordinates.
        face: The ``(m, d)`` current support points.
        basis: The orthonormal edge basis from :func:`_edge_frame`.
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

    offsets = points[tied] - face[0]
    residuals = offsets - (offsets @ basis) @ basis.T
    return int(tied[np.argmax(np.einsum("ij,ij->i", residuals, residuals))])


def ball_with_counts(
    points: np.ndarray,
    pivot_rule: PivotRule = "heuristic",
    verbose: bool = False,
) -> Ball:
    """Solve the smallest enclosing ball by the pivoting method, reporting the work done.

    Args:
        points: A ``(n, d)`` array with ``n >= 1``.
        pivot_rule: ``"heuristic"`` (the paper's own code) or ``"bland"`` (the
            rule Theorem 1 proves terminating).
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
    support = [int(np.argmax(np.einsum("ij,ij->i", offsets, offsets)))]
    # Whether c is known to lie in aff(T). It is not, initially: c is one point of
    # the cloud and aff(T) is a different one. It becomes true exactly when a walk
    # runs to completion, since the centre is then the projection onto that hull.
    on_hull = False

    drops = insertions = 0
    limit = _MAX_ITER_PER_POINT * (n + 1)
    for iteration in range(limit):
        face = pts[support]

        if on_hull:
            leaving = _leaving_point(_affine_coefficients(face, centre), support, pivot_rule)
            if leaving is None:
                # c is in conv(T): by Lemma 1 this ball is SEB(S).
                radius = float(np.linalg.norm(face - centre, axis=1).max())
                if verbose:
                    print(f"[{iteration:4d}] optimal   support={len(support):3d} radius={radius:.12g}")
                return Ball(radius, centre + shift, np.array(support, dtype=np.intp), iteration, drops, insertions)
            if verbose:
                print(f"[{iteration:4d}] drop      support={len(support):3d} point={support[leaving]}")
            del support[leaving]
            drops += 1
            on_hull = False
            continue

        # --- Walking phase ---------------------------------------------------
        # cc(T) is the orthogonal projection of c onto aff(T), by Lemma 3(i).
        basis = _edge_frame(face)
        direction = (basis @ (basis.T @ (centre - face[0]))) - (centre - face[0])
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
        shortest = float(fractions.min())

        if shortest >= 1.0:
            # Nothing stops the walk: the centre reaches cc(T) and lands on the hull.
            centre = centre + direction
            on_hull = True
            if verbose:
                radius = float(np.linalg.norm(pts[support] - centre, axis=1).max())
                print(f"[{iteration:4d}] arrive    support={len(support):3d} radius={radius:.12g}")
        else:
            entering = _entering_point(fractions, shortest, pts, face, basis, pivot_rule)
            centre = centre + shortest * direction
            support.append(entering)
            insertions += 1
            if verbose:
                radius = float(np.linalg.norm(pts[support] - centre, axis=1).max())
                print(f"[{iteration:4d}] insert    support={len(support):3d} radius={radius:.12g} step={shortest:.6g}")

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
        >>> from experiments.fischer_gaertner_kutz import min_circle_fgk
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
