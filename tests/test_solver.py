"""Tests for the convex minimum enclosing circle solver utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.optimize import linprog

from cvxball.solver import min_circle_clarabel

# Bounded, finite coordinates keep the conic programs well-conditioned.
_coords = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=64)


@st.composite
def _point_clouds(draw: st.DrawFn) -> np.ndarray:
    """Draw an (n, d) array of n points in d dimensions with finite coordinates."""
    n = draw(st.integers(min_value=1, max_value=15))
    d = draw(st.integers(min_value=1, max_value=4))
    return draw(arrays(dtype=np.float64, shape=(n, d), elements=_coords))


def test_clarabel_direct():
    """Validate `min_circle_clarabel` on a simple 2D example with known solution."""
    p = np.array([[2.0, 4.0], [0.0, 0.0], [2.5, 2.0]])
    radius, center = min_circle_clarabel(p)

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
@settings(deadline=None, max_examples=25)
@given(points=_point_clouds())
def test_ball_encloses_all_points_clarabel(points: np.ndarray) -> None:
    """`min_circle_clarabel` returns a ball that contains every point and is tight."""
    radius, center = min_circle_clarabel(points)
    distances = np.linalg.norm(points - center, axis=1)
    # Containment: every point lies inside (or on) the ball.
    assert np.all(distances <= radius + 1e-4 + 1e-5 * abs(radius))
    # Tightness/minimality: the radius equals the farthest distance (binding constraint).
    assert distances.max() == pytest.approx(radius, abs=1e-3, rel=1e-5)


# --- Degenerate inputs (issue #269) --------------------------------------------


def test_single_point() -> None:
    """A single point gives radius 0 centred on that point."""
    radius, center = min_circle_clarabel(np.array([[3.0, -1.0]]))
    assert radius == pytest.approx(0.0, abs=1e-5)
    assert center == pytest.approx([3.0, -1.0], abs=1e-4)


def test_duplicate_points_match_unique() -> None:
    """Duplicated points yield the same ball as the deduplicated set."""
    unique = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
    duplicated = np.vstack([unique, unique, unique[:1]])
    radius_u, center_u = min_circle_clarabel(unique)
    radius_d, center_d = min_circle_clarabel(duplicated)
    assert radius_d == pytest.approx(radius_u, rel=1e-5)
    assert center_d == pytest.approx(center_u, abs=1e-4)


def test_collinear_points() -> None:
    """Collinear points: the two extremes form the diameter."""
    radius, center = min_circle_clarabel(np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]))
    assert radius == pytest.approx(2.0, abs=1e-4)  # (4 - 0) / 2
    assert center == pytest.approx([2.0, 0.0], abs=1e-4)


def test_one_dimensional_points() -> None:
    """1-D inputs: radius = (max - min) / 2, centred at the midpoint."""
    radius, center = min_circle_clarabel(np.array([[-3.0], [1.0], [5.0]]))
    assert radius == pytest.approx(4.0, abs=1e-4)  # (5 - (-3)) / 2
    assert center == pytest.approx([1.0], abs=1e-4)


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
def test_ball_is_minimal(name: str, points: np.ndarray) -> None:
    """The returned ball satisfies the dual certificate, so it is genuinely minimal."""
    radius, center = min_circle_clarabel(points)
    _assert_is_minimal_ball(points, radius, center)


@pytest.mark.property
@settings(deadline=None, max_examples=25)
@given(points=_point_clouds())
def test_ball_is_minimal_property(points: np.ndarray) -> None:
    """The dual certificate holds across randomly generated clouds too."""
    radius, center = min_circle_clarabel(points)
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
def test_rejects_malformed_input(points: np.ndarray, match: str) -> None:
    """Malformed input is refused with a message about the input, not the solver.

    Each of these used to surface as an internal error: a 1-D array failed while
    unpacking ``points.shape``, and the rest reached Clarabel and returned as
    ``DualInfeasible`` or ``NumericalError``.
    """
    with pytest.raises(ValueError, match=match):
        min_circle_clarabel(points)


def test_accepts_integer_input() -> None:
    """Integer coordinates are valid and are converted rather than refused."""
    radius, center = min_circle_clarabel(np.array([[0, 0], [3, 4]]))
    assert radius == pytest.approx(2.5, abs=1e-4)
    assert center == pytest.approx([1.5, 2.0], abs=1e-4)
