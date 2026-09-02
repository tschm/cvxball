"""Tests for the convex minimum enclosing circle solver utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

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
        with pytest.raises(ValueError, match="Clarabel did not converge"):
            min_circle_clarabel(p)


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
