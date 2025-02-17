import numpy as np

from .numpy_server import NumpyServer
from .solver import min_circle_cvx


class BallServer(NumpyServer):
    def f(self, matrices: dict[str, np.ndarray]):
        self.logger.info(f"Matrices: {matrices.keys()}")
        matrix = matrices["input"]

        self.logger.info(f"Matrix: {matrix}")

        # Compute the smallest enclosing ball
        self.logger.info("Computing smallest enclosing ball...")
        radius, midpoint = min_circle_cvx(matrix, solver="CLARABEL")

        # Create result table
        return NumpyServer.results_table(
            {"radius": NumpyServer.scalar(radius), "midpoint": NumpyServer.vector(midpoint)}
        )
