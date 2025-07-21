"""Tests for the solver module.

This module tests the functionality of the min_circle_cvx function from the
cvx.ball.solver module, which computes the smallest enclosing ball for a set
of points using convex optimization.
"""

import numpy as np
import pytest

from cvx.ball.solver import min_circle_cvx


def test_random() -> None:
    """Test the solver with a specific set of points.

    This test creates a set of three points and verifies that the solver
    correctly computes the radius and center of the smallest enclosing ball.

    Verifies:
        The computed radius and center match the expected values within
        the specified tolerance.
    """
    # Create a test array with three points
    p: np.ndarray = np.array([[2.0, 4.0], [0.0, 0.0], [2.5, 2.0]])

    # Compute the smallest enclosing ball
    radius, center = min_circle_cvx(p, solver="CLARABEL")

    # Verify the results match the expected values
    assert radius == pytest.approx(2.2360679626271796, 1e-6)
    assert center == pytest.approx([1.0, 2.0], 1e-4)
