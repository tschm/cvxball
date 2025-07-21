"""Alternative implementation 2 for the minimum enclosing ball problem.

This module provides an alternative implementation of the minimum enclosing ball
algorithm using CVXPY with norm2 constraints.
"""

from typing import Any

import cvxpy as cp
import numpy as np


def min_circle_cvx(points: np.ndarray, **kwargs: Any) -> tuple[float, np.ndarray]:
    """Find the minimum enclosing ball for a set of points using CVXPY with norm2 constraints.

    This implementation uses the norm2 constraint formulation (||x - point|| <= r)
    to find the smallest enclosing ball for the given points.

    Args:
        points: Array of points with shape (n, d) where n is the number of points
                and d is the dimension
        **kwargs: Additional keyword arguments to pass to the CVXPY solver

    Returns:
        Tuple containing:
            - radius (float): The radius of the minimum enclosing ball
            - midpoint (np.ndarray): The center coordinates of the ball
    """
    # cvxpy variable for the radius
    r = cp.Variable(shape=1, name="Radius")
    # cvxpy variable for the midpoint
    x = cp.Variable(points.shape[1], name="Midpoint")
    # Define the objective function to minimize the radius
    objective = cp.Minimize(r)
    # Create norm2 constraints for each point: ||x - point|| <= r
    constraints = [cp.norm2(x - point) <= r for point in points]

    # Create and solve the optimization problem
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(**kwargs)

    return r.value[0], x.value
