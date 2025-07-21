"""Server module for the CVX Ball package.

This module provides a server implementation that computes the smallest enclosing ball
for a set of points using convex optimization.
"""

import numpy as np
from flight import Server

from .solver import min_circle_cvx


class BallServer(Server):
    """Server that computes the smallest enclosing ball for a set of points.

    This server extends the flight.Server class and implements the computation
    of the smallest enclosing ball for a set of points using convex optimization.
    """

    def f(self, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute the smallest enclosing ball for a set of points.

        Args:
            matrices: A dictionary containing input matrices. Expected to have
                     an 'input' key with a numpy array of shape (n, d) where n
                     is the number of points and d is the dimension.

        Returns:
            A dictionary containing:
                - 'radius': The radius of the smallest enclosing ball
                - 'midpoint': The center point of the ball
                - 'points': The original input points

        Raises:
            ValueError: If the input matrix is empty (has no points)
        """
        self.logger.info(f"Matrices: {matrices.keys()}")
        matrix = matrices["input"]

        if matrix.shape[0] == 0:
            # no points were given
            raise ValueError("Matrix has no values")

        self.logger.info(f"Matrix: {matrix}")

        # Compute the smallest enclosing ball using the solver
        self.logger.info("Computing smallest enclosing ball...")
        radius, midpoint = min_circle_cvx(matrix, solver="CLARABEL")

        # Return a dictionary with the results
        return {"radius": radius, "midpoint": midpoint, "points": matrix}


if __name__ == "__main__":  # pragma: no cover
    # Start the server when this module is run directly
    BallServer.start()
