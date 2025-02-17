import numpy as np
import pyarrow.flight as fl
from loguru import logger

from cvx.ball.server import BallServer


def main():
    # Connect to the server
    client = fl.connect("grpc+tcp://127.0.0.1:5008")

    # Example data
    # The server is expecting a dictionary of numpy arrays
    data = {"input": np.random.randn(20, 4)}

    # The server will return a dictionary of numpy arrays
    results = BallServer.compute(client, data=data)

    logger.info(f"Results: {results.keys()}")


if __name__ == "__main__":
    main()
