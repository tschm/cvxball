import numpy as np
import pyarrow.flight as fl
from loguru import logger

from cvx.ball.server import BallServer


def compute(client, data):
    # retrieve results
    results = BallServer.compute(client, data={"input": data})
    return results


def main():
    # Connect to the server
    client = fl.connect("grpc+tcp://127.0.0.1:5008")
    logger.info("Connected to the server.")

    # Example data
    data = np.random.rand(10000, 20)  # 10000 points in 20D space
    results = compute(client, data)
    logger.info(f"Computed results: {results}")


if __name__ == "__main__":
    main()
