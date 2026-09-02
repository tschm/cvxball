"""Tests for the convex minimum enclosing circle solver utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.optimize import linprog

import cvxball.solver as solver_module
from cvxball.solver import min_circle_active_set, min_circle_clarabel

# Every solver in the package shares one interface, so the shared expectations below
# run against each of them.
SOLVERS = [min_circle_clarabel, min_circle_active_set]
SOLVER_IDS = ["clarabel", "active_set"]

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
