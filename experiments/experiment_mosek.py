"""MOSEK implementation and performance comparison for the minimum enclosing ball problem.

This module provides a direct implementation of the minimum enclosing ball algorithm
using MOSEK's Fusion API and compares its performance with other implementations.
"""

import statistics
import timeit as tt
from typing import Any

import mosek.fusion as mf
import numpy as np
from cvx.ball.utils.circle import Circle

from cvx.ball.solver import min_circle_cvx


def min_circle_mosek(points: np.ndarray, **kwargs: Any) -> Circle:
    """Find the minimum enclosing ball for a set of points using MOSEK's Fusion API.

    This implementation directly uses MOSEK's Fusion API to solve the minimum
    enclosing ball problem using second-order cone constraints.

    Args:
        points: Array of points with shape (n, d) where n is the number of points
                and d is the dimension
        **kwargs: Additional keyword arguments to pass to the MOSEK solver

    Returns:
        Circle object containing the radius and center of the minimum enclosing ball
    """
    with mf.Model() as model:
        # Create variables for radius and midpoint
        r = model.variable("Radius", 1)
        x = model.variable("Midpoint", [1, points.shape[1]])

        k = points.shape[0]

        # Repeat the quantities for each point
        r0 = mf.Var.repeat(r, k)
        x0 = mf.Var.repeat(x, k)

        # Create second-order cone constraints
        # Based on: https://github.com/MOSEK/Tutorials/blob/master/minimum-ellipsoid/minimum-ellipsoid.ipynb
        model.constraint(mf.Expr.hstack(r0, mf.Expr.sub(x0, points)), mf.Domain.inQCone())

        # Set objective to minimize radius
        model.objective("obj", mf.ObjectiveSense.Minimize, r)
        # Solve the optimization problem
        model.solve(**kwargs)

        # Return Circle object with results
        return Circle(radius=r.level(), center=x.level())


if __name__ == "__main__":
    # Generate random test points (5000 points in 10-dimensional space)
    points = np.random.rand(5000, 10)

    # Define test functions for performance comparison
    def m1() -> Circle:
        """Test direct MOSEK implementation."""
        return min_circle_mosek(points)

    def m2() -> tuple[float, np.ndarray]:
        """Test CVXPY with MOSEK solver."""
        return min_circle_cvx(points, solver="MOSEK")

    def m3() -> tuple[float, np.ndarray]:
        """Test CVXPY with CLARABEL solver."""
        return min_circle_cvx(points, solver="CLARABEL")

    # Measure performance of direct MOSEK implementation
    times_mosek = tt.repeat(m1, number=1, repeat=5)
    print("Direct MOSEK implementation times:")
    print(times_mosek)
    print(f"Mean: {statistics.mean(times_mosek):.4f} seconds")

    # Measure performance of CVXPY with MOSEK solver
    times_cvx_mosek = tt.repeat(m2, number=1, repeat=5)
    print("\nCVXPY with MOSEK solver times:")
    print(times_cvx_mosek)
    print(f"Mean: {statistics.mean(times_cvx_mosek):.4f} seconds")

    # Measure performance of CVXPY with CLARABEL solver
    times_cvx_clarabel = tt.repeat(m3, number=1, repeat=5)
    print("\nCVXPY with CLARABEL solver times:")
    print(times_cvx_clarabel)
    print(f"Mean: {statistics.mean(times_cvx_clarabel):.4f} seconds")
