import numpy as np
import pyarrow as pa
import pyarrow.flight as fl
from np.server import Server
from np.server.utils.alter import np_2_pa

from cvx.ball.solver import min_circle_cvx


class BallServer(Server):
    def f(self, matrices: dict[str, np.ndarray]) -> pa.Table:
        self.logger.info(f"Matrices: {matrices.keys()}")
        matrix = matrices["input"]

        self.logger.info(f"Matrix: {matrix}")

        # Compute the smallest enclosing ball
        self.logger.info("Computing smallest enclosing ball...")
        radius, midpoint = min_circle_cvx(matrix, solver="CLARABEL")

        return np_2_pa({"radius": radius, "midpoint": midpoint, "points": matrix})

    @classmethod
    def descriptor(cls):
        command = cls.__name__  # Use the class name as the command.
        descriptor = fl.FlightDescriptor.for_command(command)  # Create a descriptor for the command.
        return descriptor


# entry point for Docker
if __name__ == "__main__":
    BallServer.start(host="0.0.0.0", port=8815)  # nosec: B104
