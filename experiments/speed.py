"""Performance comparison of different minimum enclosing ball implementations.

This module compares the execution speed of three different implementations
of the minimum enclosing ball algorithm:
1. The main implementation from cvx.ball.solver
2. Alternative implementation 1 using SOC constraints
3. Alternative implementation 2 using norm2 constraints
"""

import statistics
import timeit as tt

import numpy as np

from cvx.ball.solver import min_circle_cvx
from experiments.alter1 import min_circle_cvx as alter1
from experiments.alter2 import min_circle_cvx as alter2

if __name__ == "__main__":
    # Generate random test points (10000 points in 5-dimensional space)
    points = np.random.randn(10000, 5)

    # Define test functions for performance comparison
    def cvx() -> tuple[float, np.ndarray]:
        """Test the main implementation from cvx.ball.solver."""
        return min_circle_cvx(points, solver="CLARABEL")

    def alter_a() -> tuple[float, np.ndarray]:
        """Test alternative implementation 1 using SOC constraints."""
        return alter1(points, solver="CLARABEL")

    def alter_b() -> tuple[float, np.ndarray]:
        """Test alternative implementation 2 using norm2 constraints."""
        return alter2(points, solver="CLARABEL")

    # Measure performance of the main implementation
    times_clarabel = tt.repeat(cvx, number=1, repeat=5)
    print("Main implementation (cvx.ball.solver) times:")
    print(times_clarabel)
    print(f"Mean: {statistics.mean(times_clarabel):.4f} seconds")

    # Measure performance of alternative implementation 1
    times_alter1 = tt.repeat(alter_a, number=1, repeat=5)
    print("\nAlternative implementation 1 (SOC constraints) times:")
    print(times_alter1)
    print(f"Mean: {statistics.mean(times_alter1):.4f} seconds")

    # Measure performance of alternative implementation 2
    times_alter2 = tt.repeat(alter_b, number=1, repeat=5)
    print("\nAlternative implementation 2 (norm2 constraints) times:")
    print(times_alter2)
    print(f"Mean: {statistics.mean(times_alter2):.4f} seconds")
