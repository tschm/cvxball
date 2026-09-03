"""Benchmark the three smallest-enclosing-ball methods, and print the paper's tables.

Run with ``uv run --frozen python -m experiments.bench_seb``.

Two things are measured, because either alone would mislead:

- **Enclosure error**, ``max_i ||p_i - x|| / R - 1``. Positive means the returned
  ball does not actually contain the cloud. This is a deterministic property of
  the returned pair -- no timing noise, no machine dependence -- and is the
  honest way to compare an exact method with one that stops at a tolerance.
- **Time**, reported as a median over repeats. Wall clock compares
  *implementations*: the active-set method is one vectorised sweep per iteration
  while Welzl's recursion is a scalar predicate loop, and in CPython that gap
  flatters the former for reasons that have nothing to do with either algorithm.
  So Welzl's **basis count** is reported alongside -- the number of circumspheres
  it computes is a property of the algorithm and the input in any language.

The dimension sweep exists because the basis count is a high-variance random
variable: a single seed can put ``d = 10`` below ``d = 9``, and so can a mean
over five. It is reported as a median over `SEEDS` seeds with the worst seed
beside it, because the spread is part of the finding rather than noise to be
averaged away.
"""

import statistics
import time
from collections.abc import Callable

import numpy as np

from cvxball import min_circle_active_set
from experiments.clarabel_ball import min_circle_clarabel
from experiments.welzl import ball_with_counts, min_circle_welzl

# Grids small enough to finish in a few minutes, wide enough to show the trends.
# (n, d, run_welzl). Welzl is skipped where its basis count makes it hopeless in
# CPython; the dimension sweep below is where that limit is quantified rather
# than asserted, so the skipped rows point at it instead of guessing a number.
MAIN_GRID = [
    (1000, 2, True),
    (1000, 5, True),
    (10000, 3, True),
    (10000, 20, False),
    (100000, 2, True),
    (100000, 3, True),
    (100000, 10, False),
]
SWEEP_DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SWEEP_POINTS = 1000
REPEATS = 3
SEEDS = 15


def enclosure_error(points: np.ndarray, radius: float, centre: np.ndarray) -> float:
    """Compute ``max_i ||p_i - x|| / R - 1``, the relative failure to enclose.

    Args:
        points: The ``(n, d)`` cloud.
        radius: The returned radius.
        centre: The returned centre.

    Returns:
        The relative error; positive values mean points lie outside the ball.
    """
    distances = np.linalg.norm(points - centre, axis=1)
    return float(distances.max() / radius - 1.0)


def timed(
    solver: Callable[[np.ndarray], tuple[float, np.ndarray]],
    points: np.ndarray,
    repeats: int = REPEATS,
) -> tuple[float, float, np.ndarray]:
    """Run ``solver`` on ``points`` several times and keep the median time.

    Args:
        solver: A callable with this package's ``(points) -> (radius, centre)`` shape.
        points: The cloud.
        repeats: How many runs to time.

    Returns:
        The ``(median_seconds, radius, centre)`` triple.
    """
    times = []
    radius, centre = 0.0, np.zeros(points.shape[1])
    for _ in range(repeats):
        start = time.perf_counter()
        radius, centre = solver(points)
        times.append(time.perf_counter() - start)
    return statistics.median(times), radius, centre


def main_table() -> None:
    """Print the enclosure error and timing of all three methods over `MAIN_GRID`."""
    rng = np.random.default_rng(0)
    errors = f"{'err_as':>10} {'err_cl':>10} {'err_wz':>10}"
    seconds = f"{'t_as':>8} {'t_cl':>8} {'t_wz':>8}"
    print(f"{'n':>7} {'d':>3} | {errors} | {seconds} | {'bases':>7}")
    for n, d, run_welzl in MAIN_GRID:
        points = rng.normal(size=(n, d))
        t_as, r_as, c_as = timed(min_circle_active_set, points)
        t_cl, r_cl, c_cl = timed(min_circle_clarabel, points)
        row = (
            f"{n:>7} {d:>3} | "
            f"{enclosure_error(points, r_as, c_as):>10.2e} "
            f"{enclosure_error(points, r_cl, c_cl):>10.2e} "
        )
        if run_welzl:
            t_wz, r_wz, c_wz = timed(min_circle_welzl, points)
            bases = ball_with_counts(points).bases
            row += f"{enclosure_error(points, r_wz, c_wz):>10.2e} | {t_as:>8.3f} {t_cl:>8.3f} {t_wz:>8.3f} | {bases:>7}"
        else:
            row += f"{'-':>10} | {t_as:>8.3f} {t_cl:>8.3f} {'-':>8} | {'-':>7}"
        print(row)


def dimension_sweep() -> None:
    """Print how Welzl's basis count and time grow with `d`, as medians over seeds."""
    print(f"\n{'d':>3} | {'bases (med)':>12} {'bases (max)':>12} {'t_wz (med)':>11} {'t_as (med)':>11}")
    for d in SWEEP_DIMENSIONS:
        bases, welzl_times, active_times = [], [], []
        for seed in range(SEEDS):
            points = np.random.default_rng(100 + seed).normal(size=(SWEEP_POINTS, d))
            start = time.perf_counter()
            ball = ball_with_counts(points, seed=seed)
            welzl_times.append(time.perf_counter() - start)
            bases.append(ball.bases)
            start = time.perf_counter()
            min_circle_active_set(points)
            active_times.append(time.perf_counter() - start)
        print(
            f"{d:>3} | {statistics.median(bases):>12.0f} {max(bases):>12d} "
            f"{statistics.median(welzl_times):>11.3f} {statistics.median(active_times):>11.4f}"
        )


if __name__ == "__main__":
    main_table()
    dimension_sweep()
