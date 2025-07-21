"""Solver module for the CVX Ball package.

This module provides functions to compute the smallest enclosing ball
for a set of points using convex optimization with CVXPY.
"""

from typing import Any

import cvxpy as cp
import numpy as np


def min_circle_cvx(points: np.ndarray, **kwargs: dict[str, Any]) -> tuple[float, np.ndarray]:
    """Compute the smallest enclosing ball for a set of points using convex optimization.

    This function formulates the minimum enclosing ball problem as a second-order cone
    program (SOCP) and solves it using CVXPY.

    Args:
        points: A numpy array of shape (n, d) where n is the number of points
               and d is the dimension.
        **kwargs: Additional keyword arguments to pass to the CVXPY solver.
                 Common options include 'solver' to specify which solver to use.

    Returns:
        A tuple containing:
            - The radius of the smallest enclosing ball (float)
            - The center point of the ball (numpy array of shape (d,))

    Note:
        The problem is formulated as:
        minimize r
        subject to ||p_i - x||_2 <= r for all points p_i
        where r is the radius and x is the center of the ball.
    """
    # Create cvxpy variable for the radius
    r = cp.Variable(shape=1, name="Radius")

    # Create cvxpy variable for the midpoint (center of the ball)
    x = cp.Variable(points.shape[1], name="Midpoint")

    # Set the objective to minimize the radius
    objective = cp.Minimize(r)

    # Create constraints: for each point p_i, ||p_i - x||_2 <= r
    # This is formulated as a second-order cone constraint
    constraints = [
        cp.SOC(
            r * np.ones(points.shape[0]),  # t in the SOC constraint t >= ||u||_2
            points - cp.outer(np.ones(points.shape[0]), x),  # u = p_i - x for all i
            axis=1,  # Apply the constraint along axis 1 (for each point)
        )
    ]

    # Create and solve the optimization problem
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(**kwargs)

    # Return the optimal radius and midpoint
    return r.value[0], x.value
