"""Welzl's randomised incremental algorithm, as a reference implementation.

This exists to be *compared against*, not to be used: :func:`cvxball.min_circle_active_set`
is the solver this package ships. What is wanted here is an independent second
opinion on the answer, and a benchmark whose cost profile is genuinely that of
Welzl's method rather than of a vectorised rewrite of it.

The recursion is the standard one from Welzl (1991), in the form that recurses on
the boundary set rather than on a move-to-front list::

    welzl(m, R):
        b = trivial(R)                      # smallest ball with R on its sphere
        if |R| == d + 1: return b           # R already determines b
        for i in 0 .. m-1:
            if p_i not in b:
                b = welzl(i, R + [p_i])
        return b

Recursion depth is bounded by ``d + 2``, since every nested call adds one point
to ``R`` -- so this needs no recursion-limit games even at ``n = 10**5``. Points
are shuffled once at entry, which is what the expected-time analysis assumes.

Two deliberate implementation choices, because a benchmark is only as honest as
its slowest-path details:

- The containment scan is a scalar Python loop. It could be vectorised into one
  NumPy call per level, but that would measure a different algorithm: Welzl's
  cost is *number of predicate evaluations*, and the recursion restarts at the
  first violator rather than after examining the whole prefix.
- Prefixes are passed as an integer length into one shared array, never sliced,
  so the recursion does no copying that the algorithm does not require.

:func:`ball_with_counts` reports the number of basis computations and containment
predicates alongside the ball. Those counts are properties of the algorithm and
the input, not of the language it is written in, which makes them the fair thing
to quote next to a wall clock.
"""

from typing import NamedTuple

import numpy as np

# A containment test that is exact would make the recursion sensitive to the last
# bit of a squared distance.  The slack is relative, so it carries no problem scale.
_INSIDE_RTOL = 1e-12


class Ball(NamedTuple):
    """A ball, plus the work that went into finding it.

    Attributes:
        radius: The radius. Negative for the empty ball, which contains nothing.
        centre: The centre, of shape ``(d,)``.
        bases: How many times a circumsphere was computed.
        predicates: How many containment tests were evaluated.
    """

    radius: float
    centre: np.ndarray
    bases: int
    predicates: int


class _Counter:
    """Mutable tallies threaded through the recursion."""

    def __init__(self) -> None:
        """Start both tallies at zero."""
        self.bases = 0
        self.predicates = 0


def _circumsphere(boundary: np.ndarray, counter: _Counter) -> tuple[float, np.ndarray]:
    """Compute the smallest ball carrying every row of ``boundary`` on its sphere.

    The centre is confined to the affine hull of ``boundary``, which makes it
    unique: writing ``x = q_0 + D' y`` for the edge matrix ``D``, equating the
    distances to ``q_0`` and each ``q_j`` gives ``(D D') y = (||e_j||^2 / 2)_j``.

    Args:
        boundary: A ``(m, d)`` array of affinely independent points, ``m >= 1``.
        counter: The tally to charge this basis computation to.

    Returns:
        The ``(radius, centre)`` pair.
    """
    counter.bases += 1
    if boundary.shape[0] == 1:
        return 0.0, boundary[0].copy()

    edges = boundary[1:] - boundary[0]
    gram = edges @ edges.T
    rhs = 0.5 * np.einsum("ij,ij->i", edges, edges)
    try:
        y = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        # Affinely dependent boundary: impossible in exact arithmetic, since the
        # recursion only adds a point that lies outside the ball the others span.
        y = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    offset = edges.T @ y
    return float(np.linalg.norm(offset)), boundary[0] + offset


def _inside(point: np.ndarray, radius: float, centre: np.ndarray, counter: _Counter) -> bool:
    """Test containment, charging the test to ``counter``.

    Args:
        point: The point of shape ``(d,)``.
        radius: The ball's radius; negative means the empty ball.
        centre: The ball's centre.
        counter: The tally to charge this predicate to.

    Returns:
        ``True`` if ``point`` lies in the ball, up to a relative slack.
    """
    counter.predicates += 1
    if radius < 0.0:
        return False
    offset = point - centre
    return bool(float(offset @ offset) <= radius * radius * (1.0 + _INSIDE_RTOL))


def _welzl(points: np.ndarray, m: int, boundary: list[int], counter: _Counter) -> tuple[float, np.ndarray]:
    """Find the smallest ball over ``points[:m]`` with ``boundary`` on its sphere.

    Args:
        points: The shuffled ``(n, d)`` cloud.
        m: How many of its leading rows this call must enclose.
        boundary: Indices held on the sphere.
        counter: The tallies.

    Returns:
        The ``(radius, centre)`` pair.
    """
    if boundary:
        radius, centre = _circumsphere(points[boundary], counter)
    else:
        radius, centre = -1.0, np.zeros(points.shape[1])

    # d + 1 points already determine the sphere; nothing more may be added.
    if len(boundary) == points.shape[1] + 1:
        return radius, centre

    for i in range(m):
        if not _inside(points[i], radius, centre, counter):
            radius, centre = _welzl(points, i, [*boundary, i], counter)
    return radius, centre


def ball_with_counts(points: np.ndarray, seed: int = 0) -> Ball:
    """Solve the smallest enclosing ball, reporting the work done.

    Args:
        points: A ``(n, d)`` array with ``n >= 1``.
        seed: Seed for the shuffle the expected-time analysis assumes.

    Returns:
        The :class:`Ball`, including the basis and predicate counts.

    Raises:
        ValueError: If ``points`` is not a non-empty 2-D array.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(f"points must be a non-empty (n, d) array, got shape {pts.shape}")  # noqa: TRY003

    shuffled = np.random.default_rng(seed).permutation(pts)
    counter = _Counter()
    radius, centre = _welzl(shuffled, shuffled.shape[0], [], counter)
    return Ball(radius, centre, counter.bases, counter.predicates)


def min_circle_welzl(points: np.ndarray, verbose: bool = False) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing ball, in this package's solver signature.

    Args:
        points: A ``(n, d)`` array.
        verbose: If ``True``, print the basis and predicate counts.

    Returns:
        The ``(radius, center)`` pair, matching :func:`cvxball.min_circle_active_set`.
    """
    ball = ball_with_counts(points)
    if verbose:
        print(f"welzl: bases={ball.bases} predicates={ball.predicates}")
    return ball.radius, ball.centre
