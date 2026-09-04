"""Tests for the convex minimum enclosing circle solver utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.optimize import linprog

import cvxball._frame as frame_module
import cvxball.fischer_gaertner_kutz as fgk_module
import cvxball.solver as solver_module

# The Fischer-Gärtner-Kutz pivoting method, the package's second solver. Being
# shipped, it is under the same obligations as the active-set method -- every
# shared expectation below runs against both -- and it is also what the active-set
# method is checked against, because it is primal-feasible throughout: every
# iterate is already an enclosing ball, where the active-set method reaches
# feasibility only at the optimum. Two methods that are wrong in the same way
# would have to be wrong for unrelated reasons.
from cvxball.fischer_gaertner_kutz import ball_with_counts, min_circle_fgk
from cvxball.solver import min_circle_active_set

# The Clarabel route is not part of the package any more -- it lives in
# `experiments/` and clarabel is a development dependency. It is still imported
# here, because an independent second implementation of the same problem is the
# strongest check the shipped one has, and the point of moving it was to stop
# shipping it, not to stop testing against it.
from experiments.clarabel_ball import min_circle_clarabel

# All three routes share one interface, so the shared expectations below run against each.
SOLVERS = [min_circle_clarabel, min_circle_active_set, min_circle_fgk]
SOLVER_IDS = ["clarabel", "active_set", "fgk"]

# Bounded, finite coordinates keep the conic programs well-conditioned.
_coords = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=64)


@st.composite
def _point_clouds(draw: st.DrawFn) -> np.ndarray:
    """Draw an (n, d) array of n points in d dimensions with finite coordinates."""
    n = draw(st.integers(min_value=1, max_value=15))
    d = draw(st.integers(min_value=1, max_value=4))
    return draw(arrays(dtype=np.float64, shape=(n, d), elements=_coords))


# --- Minimality: the dual certificate (issue #11) ------------------------------


def _assert_is_minimal_ball(points: np.ndarray, radius: float, center: np.ndarray, rtol: float = 1e-3) -> None:
    """Assert (radius, center) is the *minimum* enclosing ball, not merely a tight one.

    Containment plus tightness -- every point inside, and the radius equal to the
    farthest distance -- is necessary but **not** sufficient: any centre at all
    satisfies both once its radius is set to its own farthest point. What separates
    the minimum ball from a merely tight one is the dual (KKT) certificate: the
    centre must lie in the convex hull of the points *on the boundary*.

    Hull membership is a feasibility question -- do non-negative weights summing to
    one reproduce the centre -- so it is asked as a linear program with no objective.

    The reproduction has to be tolerant, not exact. Whenever fewer than ``d + 1``
    points are on the boundary the hull is lower-dimensional -- two points in the
    plane span a segment, a measure-zero set -- so an equality constraint is
    infeasible for any centre carrying even one ULP of solver error, and the test
    would fail on correct answers. Asking instead for the hull point to be within
    ``tol`` of the centre keeps the geometry and drops the false precision.

    ``tol`` is scaled by the extent of the cloud because the solver's own accuracy
    is: an interior-point method resolves this centre to roughly ``1e-4`` *relative*,
    so on the obtuse triangle below -- coordinates running to 10, boundary hull a
    segment -- the centre lands 7e-4 off it. A fixed absolute tolerance either fails
    there or is far too loose for a unit-scale cloud.
    """
    tol = rtol * max(1.0, float(np.abs(points).max()))
    distances = np.linalg.norm(points - center, axis=1)
    assert distances.max() <= radius + tol, "ball does not enclose every point"

    boundary = points[distances >= distances.max() - tol]
    assert len(boundary) > 0, "no point lies on the boundary, so nothing pins the ball"

    # Weights w >= 0 with sum(w) == 1 and |boundary.T @ w - center| <= tol,
    # the two-sided bound written as the pair of inequalities linprog takes.
    n = len(boundary)
    result = linprog(
        c=np.zeros(n),
        A_ub=np.vstack([boundary.T, -boundary.T]),
        b_ub=np.concatenate([center + tol, tol - center]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    assert result.status == 0, f"centre {center} is outside the convex hull of its {n} boundary point(s)"


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_known_solution(solver):
    """Both solvers reproduce a simple 2D example with a known solution."""
    p = np.array([[2.0, 4.0], [0.0, 0.0], [2.5, 2.0]])
    radius, center = solver(p)

    assert radius == pytest.approx(2.2360679626271796, 1e-6)
    assert center == pytest.approx([1.0, 2.0], 1e-4)


def test_min_circle_clarabel_non_converged():
    """Raise ValueError when Clarabel returns a non-Solved status."""
    import clarabel

    p = np.array([[0.0, 0.0], [1.0, 1.0]])
    bad_solution = MagicMock()
    bad_solution.status = clarabel.SolverStatus.AlmostSolved  # ty: ignore[unresolved-attribute]

    with patch("clarabel.DefaultSolver") as mock_solver_cls:  # ty: ignore[unresolved-attribute]
        mock_solver_cls.return_value.solve.return_value = bad_solution
        with pytest.raises(ValueError, match="Clarabel did not converge") as excinfo:
            min_circle_clarabel(p)

    # The status alone is not actionable, so the message must also say what the
    # caller can do about it.
    assert "recentring" in str(excinfo.value)
    assert "rescaling" in str(excinfo.value)


# --- Property-based tests (issue #267) -----------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@settings(deadline=None, max_examples=25)
@given(points=_point_clouds())
def test_ball_encloses_all_points(solver, points: np.ndarray) -> None:
    """Each solver returns a ball that contains every point and is tight."""
    radius, center = solver(points)
    distances = np.linalg.norm(points - center, axis=1)
    # Containment: every point lies inside (or on) the ball.
    assert np.all(distances <= radius + 1e-4 + 1e-5 * abs(radius))
    # Tightness/minimality: the radius equals the farthest distance (binding constraint).
    assert distances.max() == pytest.approx(radius, abs=1e-3, rel=1e-5)


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(points=_point_clouds())
def test_active_set_agrees_with_clarabel(points: np.ndarray) -> None:
    """The active-set method and the cone program find the same ball."""
    radius_c, center_c = min_circle_clarabel(points)
    radius_a, center_a = min_circle_active_set(points)

    assert radius_a == pytest.approx(radius_c, abs=1e-5, rel=1e-5)
    # The active-set answer is held to its own (exact) standard...
    _assert_is_minimal_ball(points, radius_a, center_a)
    # ...while the centres only have to agree to interior-point accuracy. Clarabel is
    # the loose one here: on a cloud whose points all sit on the optimal sphere it
    # misplaces the centre by ~3e-4, where the active-set method is exact.
    assert np.linalg.norm(center_a - center_c) <= 1e-3 * max(1.0, radius_c)


# --- The Fischer-Gärtner-Kutz pivoting method ----------------------------------


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(points=_point_clouds())
def test_fgk_agrees_with_active_set(points: np.ndarray) -> None:
    """The pivoting method and the active-set method find the same ball.

    Both are exact combinatorial methods, so unlike the Clarabel comparison this
    one is held to a near-machine standard rather than to interior-point accuracy.
    The centre is compared relative to the radius: the two methods build it
    differently -- one solves afresh for the circumcentre of its final support, the
    other accumulates the walks that reached it -- so they agree on the ball
    without agreeing bit-for-bit on its coordinates.
    """
    radius_f, center_f = min_circle_fgk(points)
    radius_a, center_a = min_circle_active_set(points)

    assert radius_f == pytest.approx(radius_a, abs=1e-9, rel=1e-9)
    assert np.linalg.norm(center_f - center_a) <= 1e-8 * max(1.0, radius_a)
    _assert_is_minimal_ball(points, radius_f, center_f)


@pytest.mark.parametrize("pivot_rule", ["heuristic", "bland"])
def test_fgk_pivot_rules_agree_on_cospherical_points(pivot_rule: str) -> None:
    """Both pivot rules solve the paper's hardest case: points on a common sphere.

    Cospherical input is where the degeneracy the rules exist for actually bites --
    every point is a candidate for the support, so points enter it only to be
    dropped again. Bland's rule is what Theorem 1 proves terminating; the heuristic
    is what the paper's own code runs. Both must land on the unit sphere.
    """
    rng = np.random.default_rng(11)
    points = rng.normal(size=(200, 5))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    ball = ball_with_counts(points, pivot_rule=pivot_rule)

    assert ball.radius == pytest.approx(1.0, rel=1e-9)
    assert np.linalg.norm(ball.centre) <= 1e-9


def test_fgk_support_is_on_the_boundary_and_affinely_independent() -> None:
    """The returned support set carries the ball, which is the method's invariant.

    Every support point must sit on the sphere, there can be at most ``d + 1`` of
    them, and they must be affinely independent -- the invariant of Fig. 2 that the
    stability threshold exists to protect in floating point.
    """
    rng = np.random.default_rng(3)
    points = rng.normal(size=(300, 6))

    ball = ball_with_counts(points)
    support = points[ball.support]

    assert 1 <= len(ball.support) <= points.shape[1] + 1
    distances = np.linalg.norm(support - ball.centre, axis=1)
    assert distances == pytest.approx(ball.radius, rel=1e-9)

    edges = support[1:] - support[0]
    assert np.linalg.matrix_rank(edges) == len(edges), "support is affinely dependent"


def test_fgk_centre_lies_in_the_hull_of_its_support() -> None:
    """Lemma 1: the ball is optimal exactly when the centre is in ``conv(T)``.

    This is the certificate the algorithm terminates on, so it is worth asserting
    against the support the method actually returns rather than against the
    boundary points recovered by distance in :func:`_assert_is_minimal_ball`.
    """
    rng = np.random.default_rng(5)
    points = rng.normal(size=(150, 4))

    ball = ball_with_counts(points)
    support = points[ball.support]

    result = linprog(
        c=np.zeros(len(support)),
        A_eq=np.vstack([support.T, np.ones(len(support))]),
        b_eq=np.concatenate([ball.centre, [1.0]]),
        bounds=[(0.0, None)] * len(support),
        method="highs",
    )
    assert result.status == 0, "centre is not a convex combination of the support"


@pytest.mark.parametrize("cloud", ["gaussian", "cospherical", "duplicated", "flat"])
def test_fgk_dynamic_qr_matches_refactorising(cloud: str) -> None:
    """Maintaining the factorisation across pivots gives the same ball as rebuilding.

    :class:`_Frame` repairs ``Q`` and ``R`` in place as points enter and leave the
    support, which is section 4 of the paper and what makes the method usable at
    the dimensions it was written for. ``dynamic_qr=False`` refactorises instead.
    The two are different arithmetic reaching the same combinatorial answer, so
    the support sets must agree exactly and the radii to near machine precision.

    The awkward update is dropping the *origin*: no column of the edge matrix
    corresponds to it, so it becomes a column deletion plus a rank-one update.
    Cospherical and duplicated clouds are here because they are what drive drops.
    """
    rng = np.random.default_rng(4)
    points = rng.normal(size=(180, 7))
    if cloud == "cospherical":
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    elif cloud == "duplicated":
        points = np.repeat(points[:20], 9, axis=0)
    elif cloud == "flat":
        points[:, 4:] = 0.0

    dynamic = ball_with_counts(points, dynamic_qr=True)
    rebuilt = ball_with_counts(points, dynamic_qr=False)

    # Compare the support *points*, not their indices: on a cloud carrying nine
    # copies of every row the two paths may name different copies of the same
    # point, which is not a disagreement about the ball.
    dynamic_rows = sorted(tuple(np.round(row, 12)) for row in points[dynamic.support])
    rebuilt_rows = sorted(tuple(np.round(row, 12)) for row in points[rebuilt.support])

    assert dynamic_rows == rebuilt_rows
    assert dynamic.radius == pytest.approx(rebuilt.radius, rel=1e-10)
    assert dynamic.centre == pytest.approx(rebuilt.centre, abs=1e-10 * max(rebuilt.radius, 1.0))


def test_fgk_dynamic_qr_survives_dropping_the_origin() -> None:
    """The re-origining update keeps ``QR`` equal to the edge matrix it stands for.

    Dropping support position 0 is the one update with no column to delete, so it
    is done as a deletion plus a rank-one correction. This drives a frame through
    inserts and an origin drop and checks the invariant directly, rather than only
    through the ball that comes out.
    """
    rng = np.random.default_rng(9)
    points = rng.normal(size=(6, 5))
    frame = fgk_module._Frame(points, 0, dynamic=True)
    for index in range(1, 5):
        frame.insert(index)

    frame.remove(0)

    face = points[frame.support]
    edges = (face[1:] - face[0]).T
    assert frame.basis.shape == (5, len(frame.support) - 1)
    np.testing.assert_allclose(frame.basis @ frame._r, edges, atol=1e-12)
    # Q must still have orthonormal columns after the update.
    identity = frame.basis.T @ frame.basis
    np.testing.assert_allclose(identity, np.eye(identity.shape[0]), atol=1e-12)


def test_fgk_frame_refactorises_back_to_a_single_point() -> None:
    """A support that shrinks to one point has an empty edge matrix, not a broken one.

    The rebuild path has to produce the ``(d, 0)`` and ``(0, 0)`` pair the frame
    starts life with, since every consumer of ``Q`` and ``R`` is written against
    that shape. Reached whenever the last drop of a walk leaves one point behind.
    """
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    frame = fgk_module._Frame(points, 0, dynamic=False)
    frame.insert(1)

    frame.remove(1)

    assert frame.support == [0]
    assert frame.basis.shape == (3, 0)
    assert frame.coefficients(points[0]) == pytest.approx(np.ones(1))


@pytest.mark.parametrize(
    ("routine", "error", "act"),
    [
        ("qr_insert", np.linalg.LinAlgError, lambda frame: frame.insert(4)),
        ("qr_delete", np.linalg.LinAlgError, lambda frame: frame.remove(2)),
        ("qr_update", ValueError, lambda frame: frame.remove(0)),
    ],
    ids=["insert", "delete", "reorigin"],
)
def test_fgk_frame_falls_back_when_an_update_is_refused(routine, error, act) -> None:
    """A refused update is met by rebuilding, and the rebuild is counted.

    scipy's updates decline what an economic ``QR`` cannot represent -- a column
    already in the span, or a shape its rank-one update will not take. The
    stability threshold makes that rare rather than impossible, so each of the
    three update sites has a rebuild behind it. Forcing the refusal is the only
    way to reach them deliberately; that the counters move is the point, because
    a fallback that stopped being rare would mean the data structure had stopped
    paying.
    """

    def refuse(*_args, **_kwargs):
        raise error("refused")

    rng = np.random.default_rng(11)
    points = rng.normal(size=(6, 5))
    frame = fgk_module._Frame(points, 0, dynamic=True)
    for index in (1, 2, 3):
        frame.insert(index)

    with patch.object(fgk_module, routine, refuse):
        act(frame)

    assert frame.fallbacks == 1
    assert frame.rebuilds == 1
    face = points[frame.support]
    np.testing.assert_allclose(frame.basis @ frame._r, (face[1:] - face[0]).T, atol=1e-12)


def test_fgk_frame_drops_the_origin_of_a_two_point_support() -> None:
    """Re-origining a support of two leaves the survivor with no edges at all.

    The general path deletes a column and applies a rank-one update to re-measure
    the rest from the promoted point. With nothing left to re-measure there is no
    update to apply, and the frame has to fall back to the empty pair directly.
    """
    points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    frame = fgk_module._Frame(points, 0, dynamic=True)
    frame.insert(1)

    frame.remove(0)

    assert frame.support == [1]
    assert frame.basis.shape == (2, 0)


def test_fgk_frame_refuses_a_duplicate_of_its_origin() -> None:
    """A point coincident with ``q_0`` cannot join the support.

    Its offset has length zero, so the angle test that guards every insertion has
    no denominator. Duplicated clouds are ordinary input, so this is the guard
    that keeps them from reaching ``qr_insert`` as a zero column.
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    frame = fgk_module._Frame(points, 0, dynamic=True)

    assert frame.admits(1)
    assert not frame.admits(2)


def test_fgk_coefficients_fall_back_to_least_squares() -> None:
    """A singular ``R`` is answered by least squares, not by an exception.

    Back substitution is what the paper does and what the maintained factorisation
    supports, but a *rebuilt* one can put an exact zero on the diagonal where the
    updated one keeps it merely small. The two solutions agree wherever the system
    is solvable at all, which is what makes the rebuild baseline comparable.
    """
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    frame = fgk_module._Frame(points, 0, dynamic=True)
    frame.insert(1)
    frame.insert(2)
    centre = np.array([1.0, 1.0])

    exact = frame.coefficients(centre)

    def refuse(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular")

    with patch.object(fgk_module, "solve_triangular", refuse):
        fallback = frame.coefficients(centre)

    assert fallback == pytest.approx(exact)
    assert points[frame.support].T @ fallback == pytest.approx(centre)


def test_fgk_skips_a_candidate_the_factorisation_will_not_take() -> None:
    """A refused entering point is masked for the rest of the walk, not forced in.

    ``admits`` is the angle test that keeps ``T`` affinely independent, and it can
    refuse a point the stopping fractions nominated -- the absolute margin of the
    paper and the reciprocal condition number of the factorisation disagree on a
    cloud carrying a far outlier. Fig. 2's invariant has no answer to a dependent
    support, so the walk must drop the candidate and look again.

    The refusal is forced here, on a cloud of triplicated points, because that is
    the case where it costs nothing: the sibling copy of the refused point enters
    instead, so the mechanics can be checked against the ball the shipped
    active-set method returns. Refusing an arbitrary point on an ordinary cloud
    would change the answer, which is why the guard is an angle test and not a
    preference.
    """
    admits = fgk_module._Frame.admits
    calls: list[int] = []

    def refuse_once(self, index: int) -> bool:
        calls.append(index)
        return False if len(calls) == 1 else bool(admits(self, index))

    points = np.repeat(np.random.default_rng(0).normal(size=(20, 4)), 3, axis=0)
    with patch.object(fgk_module._Frame, "admits", refuse_once):
        ball = ball_with_counts(points)

    expected_radius, expected_centre = min_circle_active_set(points)
    assert calls[1] != calls[0]
    assert calls[0] not in ball.support
    assert ball.radius == pytest.approx(expected_radius)
    assert ball.centre == pytest.approx(expected_centre, abs=1e-12)


def test_fgk_bland_rule_drops_the_lowest_indexed_point() -> None:
    """The two pivot rules choose different points, and each chooses its own.

    Bland's rule is what Theorem 1 proves terminating: among the negative
    coefficients it takes the point of smallest rank in the fixed order on ``S``,
    here the index into the input. The heuristic takes the most negative
    coefficient instead. Given a case where the two disagree, each must pick the
    point its rule names -- the anti-cycling guarantee is exactly this choice.
    """
    coefficients = np.array([0.5, -0.75, -0.25])
    support = [7, 9, 3]

    assert fgk_module._leaving_point(coefficients, support, "bland") == 2
    assert fgk_module._leaving_point(coefficients, support, "heuristic") == 1
    assert fgk_module._leaving_point(np.array([0.5, 0.5]), [1, 2], "bland") is None


def test_fgk_rejects_an_unknown_pivot_rule() -> None:
    """A misspelled rule fails loudly rather than silently picking a default."""
    with pytest.raises(ValueError, match="pivot_rule must be"):
        ball_with_counts(np.array([[0.0, 0.0], [1.0, 0.0]]), pivot_rule="dantzig")


def test_fgk_verbose_logs_phases(capsys) -> None:
    """Verbose mode names the phase of each pivot step and the final ball."""
    rng = np.random.default_rng(1)
    min_circle_fgk(rng.normal(size=(80, 3)), verbose=True)

    out = capsys.readouterr().out
    assert "optimal" in out
    assert "fgk: iterations=" in out


def test_fgk_iteration_limit(monkeypatch) -> None:
    """The safety net fires rather than looping forever if termination fails."""
    monkeypatch.setattr(fgk_module, "_MAX_ITER_PER_POINT", 0)
    with pytest.raises(ValueError, match="did not converge"):
        min_circle_fgk(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


def test_fgk_does_not_mutate_input() -> None:
    """Recentring must not be done in place on the caller's array."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    original = points.copy()

    min_circle_fgk(points)

    np.testing.assert_array_equal(points, original)


# --- The maintained factorisation (src/cvxball/_frame.py) ----------------------


@pytest.mark.parametrize("cloud", ["gaussian", "cospherical", "duplicated", "flat"])
def test_maintained_factorisation_matches_rebuilding(cloud: str) -> None:
    """Carrying the factorisation changes the arithmetic, not the answer.

    ``d`` is above :data:`_MAINTAIN_MIN_DIM` here so the maintained path is the
    one the solver picks by default; ``maintain=False`` forces the rebuild it
    would use in lower dimensions. Both must find the same ball.
    """
    rng = np.random.default_rng(6)
    points = rng.normal(size=(150, 120))
    if cloud == "cospherical":
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    elif cloud == "duplicated":
        points = np.repeat(points[:20], 8, axis=0)
    elif cloud == "flat":
        points[:, 60:] = 0.0

    radius_kept, centre_kept = min_circle_active_set(points, maintain=True)
    radius_built, centre_built = min_circle_active_set(points, maintain=False)

    assert radius_kept == pytest.approx(radius_built, rel=1e-10)
    assert centre_kept == pytest.approx(centre_built, abs=1e-10 * max(radius_built, 1.0))
    _assert_is_minimal_ball(points, radius_kept, centre_kept)


@pytest.mark.property
@settings(deadline=None, max_examples=25)
@given(points=_point_clouds())
def test_maintained_factorisation_agrees_on_small_clouds(points: np.ndarray) -> None:
    """Forcing the maintained path in low dimensions still gives the same ball.

    The dispatch would not choose it here -- these clouds are far below the
    threshold -- but the two paths must agree wherever both are legal, or the
    threshold would be hiding a disagreement rather than picking a faster route.
    """
    radius_kept, centre_kept = min_circle_active_set(points, maintain=True)
    radius_built, centre_built = min_circle_active_set(points, maintain=False)

    assert radius_kept == pytest.approx(radius_built, abs=1e-12, rel=1e-12)
    assert np.linalg.norm(centre_kept - centre_built) <= 1e-10 * max(1.0, radius_built)


def test_maintained_face_updates_keep_qr_equal_to_the_edge_matrix() -> None:
    """Every update leaves ``QR`` equal to the edge matrix it stands for.

    Walks a frame through the three updates -- inserts, an ordinary removal, and
    the re-origining removal of position 0, which has no column of its own -- and
    checks the invariant directly after each, rather than only through the ball
    that eventually comes out.
    """
    rng = np.random.default_rng(9)
    points = rng.normal(size=(9, 6))
    frame = frame_module._MaintainedFace(points, 0)

    def check() -> None:
        """Assert QR reproduces the edges and Q still has orthonormal columns."""
        face = points[frame.support]
        edges = (face[1:] - face[0]).T
        np.testing.assert_allclose(frame._q @ frame._r, edges, atol=1e-12)
        gram = frame._q.T @ frame._q
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-12)

    for index in range(1, 6):
        frame.insert(index)
        check()
    frame.remove(3)
    check()
    frame.remove(0)
    check()
    assert frame.fallbacks == 0, "points in general position need no rebuild"


def test_maintained_face_singleton_support() -> None:
    """A one-point support has an empty edge matrix and a trivial subproblem."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    frame = frame_module._MaintainedFace(points, 1)

    assert frame.null_space().shape == (1, 0)
    assert frame.circumcentre_weights() == pytest.approx([1.0])
    np.testing.assert_array_equal(frame.face, points[[1]])

    # Removing back down to a singleton takes the `remaining <= 0` branch.
    frame.insert(0)
    frame.remove(0)
    assert frame.support == [0]
    assert frame.null_space().shape == (1, 0)


def test_maintained_face_rebuilds_on_a_dependent_insert() -> None:
    """An economic QR cannot hold a dependent column, so the frame refactorises.

    This is where the method differs from the pivoting one: it *permits* affine
    dependence and answers it with a null-space descent step, so the
    factorisation has to survive a state ``qr_insert`` refuses outright. After
    the rebuild ``R`` is singular, which is exactly what lets ``null_space`` find
    the direction that shrinks the support.
    """
    # The support must span *less* than the whole space for the refusal to fire:
    # with Q already square the new column needs no fresh direction and scipy
    # simply widens R. So this is a square lying in a plane of three-space.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    frame = frame_module._MaintainedFace(points, 0)
    frame.insert(1)
    frame.insert(2)

    frame.insert(3)  # (1, 1, 0) is in the affine hull of the first three

    assert frame.fallbacks == 1
    null_space = frame.null_space()
    assert null_space.shape[1] > 0, "the dependent support was not detected"
    for column in null_space.T:
        assert column.sum() == pytest.approx(0.0, abs=1e-12)
        assert points[frame.support].T @ column == pytest.approx(np.zeros(3), abs=1e-12)


def test_maintained_face_widens_r_when_the_support_spans_the_space() -> None:
    """A dependent column needs no rebuild once ``Q`` is already square.

    The complement of the case above, and the reason ``_economise`` has to cope
    with a rectangular ``R``: when the support already spans the whole space
    there is no new direction to find, so ``qr_insert`` succeeds and simply adds
    a column to ``R``.
    """
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    frame = frame_module._MaintainedFace(points, 0)
    frame.insert(1)
    frame.insert(2)

    frame.insert(3)

    assert frame.fallbacks == 0
    assert frame._r.shape == (2, 3), "R should have widened, not squared"
    assert frame.null_space().shape[1] > 0


def test_maintained_face_duplicate_point_is_rebuilt() -> None:
    """Inserting a point identical to the origin gives a zero-length column."""
    points = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    frame = frame_module._MaintainedFace(points, 0)

    frame.insert(1)  # identical to the origin, so the first edge has zero length

    assert frame.support == [0, 1]
    assert frame.null_space().shape[1] > 0


def test_solver_dispatches_on_dimension() -> None:
    """The default picks the maintained path only above the measured threshold."""
    assert solver_module._MAINTAIN_MIN_DIM == 100

    rng = np.random.default_rng(2)
    low = rng.normal(size=(40, 5))
    high = rng.normal(size=(40, 150))

    # Both dimensions must solve correctly under the automatic choice.
    for cloud in (low, high):
        radius, centre = min_circle_active_set(cloud)
        forced, _ = min_circle_active_set(cloud, maintain=cloud.shape[1] < 100)
        assert radius == pytest.approx(forced, rel=1e-10)
        _assert_is_minimal_ball(cloud, radius, centre)


# --- Degenerate inputs (issue #269) --------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_single_point(solver) -> None:
    """A single point gives radius 0 centred on that point."""
    radius, center = solver(np.array([[3.0, -1.0]]))
    assert radius == pytest.approx(0.0, abs=1e-5)
    assert center == pytest.approx([3.0, -1.0], abs=1e-4)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_duplicate_points_match_unique(solver) -> None:
    """Duplicated points yield the same ball as the deduplicated set."""
    unique = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
    duplicated = np.vstack([unique, unique, unique[:1]])
    radius_u, center_u = solver(unique)
    radius_d, center_d = solver(duplicated)
    assert radius_d == pytest.approx(radius_u, rel=1e-5)
    assert center_d == pytest.approx(center_u, abs=1e-4)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_collinear_points(solver) -> None:
    """Collinear points: the two extremes form the diameter."""
    radius, center = solver(np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]))
    assert radius == pytest.approx(2.0, abs=1e-4)  # (4 - 0) / 2
    assert center == pytest.approx([2.0, 0.0], abs=1e-4)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_one_dimensional_points(solver) -> None:
    """1-D inputs: radius = (max - min) / 2, centred at the midpoint."""
    radius, center = solver(np.array([[-3.0], [1.0], [5.0]]))
    assert radius == pytest.approx(4.0, abs=1e-4)  # (5 - (-3)) / 2
    assert center == pytest.approx([1.0], abs=1e-4)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_all_points_identical(solver) -> None:
    """A cloud of coincident points collapses to a zero-radius ball."""
    radius, center = solver(np.tile([2.0, -3.0], (5, 1)))
    assert radius == pytest.approx(0.0, abs=1e-5)
    assert center == pytest.approx([2.0, -3.0], abs=1e-4)


# --- Active-set specific: the two ways its working set changes ------------------
#
# The method either jumps to the subproblem solution and frees the farthest
# violating point, or it has to shrink the support first. Shrinking happens for two
# distinct reasons, and each one is a separate code path worth pinning down.


def test_active_set_drops_negative_weight() -> None:
    """A support point whose subproblem weight turns negative is dropped.

    Here the circumcentre of an intermediate support set falls outside that set's
    simplex, so the full step is infeasible and the method must take a partial step.
    """
    points = np.array([[0.3, -3.5], [4.6, -1.0], [-2.0, 3.5], [-3.8, 2.3], [-3.1, -1.1]])
    radius, center = min_circle_active_set(points)

    _assert_is_minimal_ball(points, radius, center)
    assert radius == pytest.approx(min_circle_clarabel(points)[0], abs=1e-6)


def test_active_set_affinely_dependent_support() -> None:
    """An affinely dependent support set is resolved along the null-space direction.

    Freeing a violating point can make the support affinely dependent (here: four
    points in the plane). The subproblem is then unbounded, and the method has to
    move along a direction that leaves the centre fixed until a weight hits zero.
    """
    points = np.array([[3.2, 4.3], [-3.4, 3.2], [2.8, -2.6], [-2.1, 4.6], [-1.4, -2.1]])
    radius, center = min_circle_active_set(points)

    _assert_is_minimal_ball(points, radius, center)
    assert radius == pytest.approx(min_circle_clarabel(points)[0], abs=1e-6)


def test_active_set_dependent_and_dropping() -> None:
    """A cloud that exercises both support-shrinking paths in a single run."""
    points = np.array([[-0.9, -3.5], [-4.9, 1.4], [-2.7, 4.6], [-3.0, -1.8], [4.8, 0.2]])
    radius, center = min_circle_active_set(points)

    _assert_is_minimal_ball(points, radius, center)
    assert radius == pytest.approx(min_circle_clarabel(points)[0], abs=1e-6)


@pytest.mark.parametrize(
    ("face", "expected"),
    [
        # m - 1 <= d, affinely independent: the cheap reduced path, no null directions.
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0),
        # m - 1 <= d, affinely dependent: three collinear points in space.
        (np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), 1),
        # m - 1 > d: five points in the plane. Here the reduced left factor is 4 x 2
        # and stops one column short, so an unguarded reduced SVD reports *no* null
        # directions and the caller then solves a singular system.
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]]), 2),
        # m - 1 > d in one dimension, the tightest form of the same thing.
        (np.array([[0.0], [1.0], [2.0]]), 1),
    ],
    ids=["independent", "dependent", "wider-support", "one-dimensional"],
)
def test_affine_null_space_spans_when_the_support_outgrows_the_dimension(face, expected) -> None:
    """The null space is complete whether or not the support fits inside ``d``.

    :func:`_affine_null_space` reads only the left factor of the SVD, so it asks for
    the reduced decomposition -- the full one would build and discard a ``d x d``
    right factor, which is the dominant cost of the whole solver at large ``d``. That
    substitution is exact only while ``edges`` is no taller than it is wide, and the
    support does reach ``d + 2`` points between a drop and the following add. The last
    two cases are that situation: an unguarded reduced SVD returns zero null
    directions for both, silently reclassifying a degenerate face as independent.
    """
    null_space = solver_module._affine_null_space(face)

    assert null_space.shape == (len(face), expected)
    for column in null_space.T:
        # A null direction reshuffles the weights without moving the centre, so it
        # must sum to zero and be annihilated by the points.
        assert column.sum() == pytest.approx(0.0, abs=1e-12)
        assert face.T @ column == pytest.approx(np.zeros(face.shape[1]), abs=1e-12)


def test_active_set_is_exact_on_cospherical_points() -> None:
    """Points placed exactly on a sphere are recovered to machine precision.

    This is the configuration an interior-point solver only approaches to its own
    tolerance, whereas the active-set method terminates at the exact vertex.
    """
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(40, 3))
    points = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    radius, center = min_circle_active_set(points)
    assert radius == pytest.approx(1.0, abs=1e-12)
    assert center == pytest.approx(np.zeros(3), abs=1e-12)


def test_active_set_verbose_logs_iterations(capsys) -> None:
    """`verbose=True` prints one line per iteration, mirroring Clarabel's log."""
    points = np.array([[-0.9, -3.5], [-4.9, 1.4], [-2.7, 4.6], [-3.0, -1.8], [4.8, 0.2]])
    min_circle_active_set(points, verbose=True)

    out = capsys.readouterr().out
    assert "support=" in out
    assert out.count("\n") >= 2


def test_active_set_iteration_limit(monkeypatch) -> None:
    """Exhausting the iteration budget raises rather than returning a wrong ball."""
    monkeypatch.setattr(solver_module, "_MAX_ITER_PER_POINT", 0)
    with pytest.raises(ValueError, match="did not converge"):
        min_circle_active_set(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]))


def test_active_set_stall_guard(monkeypatch) -> None:
    """A step that nothing blocks raises instead of yielding non-finite weights.

    Exact arithmetic rules this out — the point just freed always pushes the step in
    a direction some weight blocks — so the guard is driven here by forcing a
    degenerate (all-zero) null-space direction.
    """
    monkeypatch.setattr(solver_module, "_affine_null_space", lambda face: np.zeros((face.shape[0], 1)))
    with pytest.raises(ValueError, match="stalled"):
        min_circle_active_set(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]))


def test_active_set_handles_tiny_scales() -> None:
    """Clouds many orders of magnitude below 1 are solved on their own scale.

    A rank test that stacks a row of ones onto the coordinates would call these two
    points affinely dependent, and an absolute tolerance on *squared* distances would
    declare the degenerate radius 0 optimal.
    """
    for scale in (1e-7, 1e-30, 1e-55):
        points = np.array([[scale], [0.0]])
        radius, center = min_circle_active_set(points)
        assert radius == pytest.approx(scale / 2, rel=1e-12)
        assert center == pytest.approx([scale / 2], rel=1e-12)


def test_active_set_handles_huge_scales() -> None:
    """The same cloud scaled up stays exact, so no tolerance is hard-wired to 1."""
    points = np.array([[0.0, 0.0], [6e12, 8e12], [1e12, 1e12]])
    radius, center = min_circle_active_set(points)
    assert radius == pytest.approx(5e12, rel=1e-12)
    assert center == pytest.approx([3e12, 4e12], rel=1e-12)


def test_active_set_survives_squaring_the_exponent_range() -> None:
    """Extreme but ordinary magnitudes stay solvable, because the method squares.

    A double reaches to ~1e308, so a *squared* quantity reaches only ~1e154. This
    method squares everything it touches -- the Gram matrix, the distances, the
    noise floor -- which halves the usable exponent range. Before the cloud was
    normalised by a power of two, a cloud of extent 1e-160 produced a Gram matrix
    in the subnormals whose solve overflowed to infinity and crashed with an
    `IndexError` from an emptied support set, and a cloud of extent 1e+160
    saturated to a silently reported `radius == 0` -- the worse of the two.

    Both ends are checked here, well past where the failures used to start, and
    the answer is exact at every one: the normalisation is a power of two.
    """
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for exponent in (-300, -200, -160, -155, -50, 0, 50, 155, 160, 200, 300):
        points = base * 10.0**exponent
        radius, center = min_circle_active_set(points)

        assert radius == pytest.approx(2**0.5 / 2 * 10.0**exponent, rel=1e-12)
        assert center == pytest.approx([0.5 * 10.0**exponent] * 2, rel=1e-12)


def test_active_set_solves_the_subnormal_gram_regression() -> None:
    """The cloud Hypothesis found: a 1-D pair whose squared extent is subnormal.

    ``6.8e-156`` squares to ``4.7e-311``, a subnormal, and the circumcentre solve
    on that Gram matrix returned ``[-inf, inf]``. The weights then went to NaN, the
    support emptied, and the next iteration indexed row 0 of an empty face.
    """
    points = np.array([[6.82355645e-156], [0.0]])
    radius, center = min_circle_active_set(points)

    assert radius == pytest.approx(6.82355645e-156 / 2, rel=1e-12)
    assert center == pytest.approx([6.82355645e-156 / 2], rel=1e-12)


def test_active_set_normalisation_preserves_bit_exactness() -> None:
    """Rescaling by a power of two moves no bits, so exact answers stay exact.

    This is the property the normalisation was chosen to protect: any other factor
    (the extent itself, say) would round, and the documented bit-for-bit results
    would become approximate.
    """
    radius, center = min_circle_active_set(np.array([[0, 0], [1, 0], [0, 1]]))
    assert radius == 2**0.5 / 2
    assert (center == 0.5).all()

    # ...and the same cloud shifted by whole binary octaves is exact there too.
    for exponent in (-500, -100, 100, 500):
        scaled = np.ldexp(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), exponent)
        radius, center = min_circle_active_set(scaled)
        assert radius == np.ldexp(2**0.5 / 2, exponent)
        assert (center == np.ldexp(0.5, exponent)).all()


def test_active_set_far_from_origin() -> None:
    """A small cloud sitting far from the origin is solved on its own extent.

    The rounding floor of a squared distance measured against an arbitrary origin is
    of order ``eps * ||p||^2``.  At this offset that is ~280 while the answer's own
    ``radius**2`` is only 25, so a solver that sizes its tolerance off the raw
    coordinates accepts the very first candidate and reports ``radius == 0``.
    Recentring on the cloud ties the tolerance to the extent instead.
    """
    offset = np.array([1e8, -3e8])
    points = offset + np.array([[0.0, 0.0], [6.0, 8.0], [1.0, 1.0]])
    radius, center = min_circle_active_set(points)

    assert radius == pytest.approx(5.0, rel=1e-9)
    # The centre is only recoverable to the resolution of the inputs themselves,
    # which is eps * 3e8 ~ 7e-8 here.
    assert center == pytest.approx(offset + np.array([3.0, 4.0]), abs=1e-6)


def test_active_set_does_not_mutate_input() -> None:
    """Recentring works on a copy, so the caller's array is left untouched."""
    points = np.array([[1.0, 2.0], [7.0, 9.0], [3.0, 4.0]])
    original = points.copy()
    min_circle_active_set(points)
    assert np.array_equal(points, original)


@pytest.mark.parametrize(
    ("name", "points"),
    [
        ("right triangle", np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
        ("obtuse triangle", np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 1.0]])),
        ("known example", np.array([[2.0, 4.0], [0.0, 0.0], [2.5, 2.0]])),
        ("collinear", np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])),
        ("cocircular square", np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])),
        ("interior point ignored", np.array([[0.0, 0.0], [6.0, 8.0], [1.0, 1.0]])),
        ("single point", np.array([[3.0, -1.0]])),
        ("one dimension", np.array([[-3.0], [1.0], [5.0]])),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_ball_is_minimal(solver, name: str, points: np.ndarray) -> None:
    """Each solver's ball satisfies the dual certificate, so it is genuinely minimal."""
    radius, center = solver(points)
    _assert_is_minimal_ball(points, radius, center)


@pytest.mark.property
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@settings(deadline=None, max_examples=25)
@given(points=_point_clouds())
def test_ball_is_minimal_property(solver, points: np.ndarray) -> None:
    """The dual certificate holds for each solver across randomly generated clouds."""
    radius, center = solver(points)
    _assert_is_minimal_ball(points, radius, center)


# --- Input validation (issue #9) -----------------------------------------------


@pytest.mark.parametrize(
    ("points", "match"),
    [
        (np.array([1.0, 2.0, 3.0]), "must be a 2-D"),
        (np.zeros((2, 2, 2)), "must be a 2-D"),
        (np.zeros((0, 2)), "is empty"),
        (np.zeros((3, 0)), "no coordinates"),
        (np.array([[0.0, 0.0], [np.nan, 1.0]]), "must be finite"),
        (np.array([[0.0, 0.0], [np.inf, 1.0]]), "must be finite"),
    ],
    ids=["1d", "3d", "no-points", "no-coords", "nan", "inf"],
)
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_rejects_malformed_input(solver, points: np.ndarray, match: str) -> None:
    """Malformed input is refused with a message about the input, not the solver.

    Each of these used to surface as an internal error: a 1-D array failed while
    unpacking ``points.shape``, the rest reached Clarabel and returned as
    ``DualInfeasible`` or ``NumericalError``, and in the active-set method a 1-D
    array surfaced a raw NumPy ``einsum`` complaint from deep inside ``_sq_dist``.

    Parametrising over both solvers is what keeps this honest: the guarantee is a
    property of the package's public surface, not of whichever solver happened to
    be written first.
    """
    with pytest.raises(ValueError, match=match):
        solver(points)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_accepts_integer_input(solver) -> None:
    """Integer coordinates are valid and are converted rather than refused."""
    radius, center = solver(np.array([[0, 0], [3, 4]]))
    assert radius == pytest.approx(2.5, abs=1e-4)
    assert center == pytest.approx([1.5, 2.0], abs=1e-4)
