"""Asymptotic performance analysis for the minimum enclosing ball algorithm.

This module provides tools to analyze the asymptotic performance of the minimum
enclosing ball algorithm by measuring execution times for different input sizes
and visualizing the results.
"""

import time
from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cvx.ball.solver import min_circle_cvx


def cvx(n: int) -> tuple[float, np.ndarray]:
    """Create random points and solve the minimum enclosing ball problem.

    Args:
        n: Number of points to generate

    Returns:
        The result of min_circle_cvx (radius and midpoint)
    """
    # Generate n random points in 5-dimensional space
    points = np.random.rand(n, 5)
    # Solve using the CLARABEL solver
    return min_circle_cvx(points, solver="CLARABEL")


def measure_execution_time(func: Callable[[int], any], n: int, num_trials: int = 3) -> float:
    """Run multiple trials and return average execution time.

    Args:
        func: Function to measure, should take n as input
        n: Input size parameter to pass to the function
        num_trials: Number of trials to run for averaging

    Returns:
        Average execution time in seconds
    """
    times = []
    for _ in range(num_trials):
        # Measure execution time
        start = time.time()
        func(n)
        times.append(time.time() - start)
    return np.mean(times)


def run_analysis() -> tuple[list[int], list[float]]:
    """Run performance analysis for different input sizes.

    Measures execution time for input sizes that are powers of 2,
    from 2^4 to 2^19.

    Returns:
        Tuple containing:
            - list of input sizes
            - list of corresponding execution times
    """
    # Test for different values of n (powers of 2)
    sequence = np.array([2**n for n in range(4, 20)])
    execution_times = []

    # Measure execution time for each input size
    for n in sequence:
        avg_time = measure_execution_time(cvx, int(n))
        execution_times.append(avg_time)
        print(f"n={n}: {avg_time:.4f} seconds")

    return sequence.tolist(), execution_times


def plot_results(sizes: list[int], times: list[float]) -> None:
    """Plot the results of the performance analysis.

    Creates a log-log plot showing the actual execution times and
    a theoretical O(n) complexity line for comparison.

    Args:
        sizes: List of input sizes
        times: List of corresponding execution times
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add actual execution times
    fig.add_trace(
        go.Scatter(
            x=sizes, y=times, name="Actual Time", mode="lines+markers", line=dict(color="blue"), marker=dict(size=8)
        )
    )

    # Add theoretical O(n) complexity line for comparison
    normalized_n = np.array(sizes) / sizes[0]
    fig.add_trace(
        go.Scatter(x=sizes, y=normalized_n * times[0], name="O(n)", line=dict(color="red", dash="dash"), mode="lines")
    )

    # Update layout with log scales for better visualization of asymptotic behavior
    fig.update_layout(
        title="Algorithm Performance Analysis",
        xaxis=dict(
            title="Input Size (n)",
            type="log",
            dtick="D1",  # Show ticks for each power of 10
        ),
        yaxis=dict(title="Execution Time (seconds)", type="log", dtick="D1"),
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99),
        plot_bgcolor="white",
    )

    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # Show the plot
    fig.show()


if __name__ == "__main__":
    # Run the analysis and plot the results
    sizes, times = run_analysis()
    plot_results(sizes, times)
