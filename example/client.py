"""Client module for the CVX Ball package.

This module provides a client implementation that connects to the BallServer,
sends data, and visualizes the results. It demonstrates how to use the CVX Ball
package to compute the smallest enclosing ball for a set of points in 2D space.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from np.flight import Client


def _plot_points(p: np.ndarray, p0: np.ndarray, r0: float) -> plt.Axes:
    """Plot the points, midpoint, and enclosing circle.

    Args:
        p: Array of points with shape (n, 2)
        p0: Midpoint (center) of the circle with shape (2,)
        r0: Radius of the circle

    Returns:
        The matplotlib Axes object containing the plot
    """
    _, ax = plt.subplots()
    ax.set_aspect("equal")

    # Plot the cloud of points as blue stars
    ax.plot(p[:, 0], p[:, 1], "b*")

    # Mark the midpoint as a red dot
    ax.plot(p0[0], p0[1], "r.")

    # Plot the circle with a red outline and white fill
    ax.add_patch(mpatches.Circle(p0, r0, fc="w", ec="r", lw=1.5))

    # Add grid for better visualization
    ax.grid(True)
    return ax


def main() -> None:
    """Main function to demonstrate the client usage.

    Connects to the BallServer, sends random 2D points,
    and visualizes the smallest enclosing circle.

    This function demonstrates the complete workflow:
    1. Connecting to the BallServer
    2. Generating random 2D points
    3. Sending the data to the server for computation
    4. Receiving the results (midpoint and radius)
    5. Visualizing the points and the enclosing circle
    """
    # Connect to the server using the provided URL
    # The Client class handles the gRPC connection and communication
    with Client("grpc+tls://cvxball-710171668953.us-central1.run.app:443") as client:
        # Generate random 2D points as example data
        # The server is expecting a dictionary with an 'input' key
        data: dict[str, np.ndarray] = {"input": np.random.randn(200, 2)}

        # Send the data to the server and get the results
        # The server will return a dictionary with 'radius', 'midpoint', and 'points' keys
        # - 'radius': float - The radius of the smallest enclosing circle
        # - 'midpoint': np.ndarray - The center coordinates of the circle (shape: (2,))
        # - 'points': np.ndarray - The original points sent to the server (shape: (n, 2))
        results: dict[str, float | np.ndarray] = client.compute(command="test", data=data)

        # Plot the results using the helper function
        # This will create a visualization with the points, midpoint, and enclosing circle
        ax = _plot_points(results["points"], results["midpoint"], results["radius"])
        ax.set_title("Smallest enclosing circle")
        plt.show()  # Display the plot in a new window


if __name__ == "__main__":
    # Run the main function when this script is executed directly
    main()
