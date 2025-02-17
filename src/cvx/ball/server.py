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

        return NumpyServer.np_2_pa({"radius": radius, "midpoint": midpoint, "points": matrix})
