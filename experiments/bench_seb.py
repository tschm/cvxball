"""Benchmark the four smallest-enclosing-ball methods, and print the paper's tables.

Run with ``uv run --frozen python -m experiments.bench_seb``.

Two things are measured, because either alone would mislead:

- **Enclosure error**, ``max_i ||p_i - x|| / R - 1``. Positive means the returned
  ball does not actually contain the cloud. This is a deterministic property of
  the returned pair -- no timing noise, no machine dependence -- and is the
  honest way to compare an exact method with one that stops at a tolerance.
- **Time**, reported as a median over repeats. Wall clock compares
  *implementations*: the active-set and pivoting methods each do one vectorised
  sweep per iteration while Welzl's recursion is a scalar predicate loop, and in
  CPython that gap flatters the former pair for reasons that have nothing to do
  with any of the algorithms. So a **step count** is reported alongside each of
  the two combinatorial methods -- the circumspheres Welzl computes, and the
  pivot steps the Fischer-Gärtner-Kutz method takes. Both are properties of the
  algorithm and the input in any language.

Note what the two step counts do *not* license: comparing them to each other. A
Welzl basis is a circumsphere solve over at most ``d + 1`` points; a pivot step is
a projection plus a sweep over all ``n``. They are each the right way to track one
method against itself as ``d`` grows, and the wrong way to rank one method against
the other -- which is what the wall clocks beside them are for.

The high-dimension table is a different comparison again, and a narrower one:
only the active-set and pivoting methods, at ``n = 1000`` and ``d`` from ``1000``
to ``16000``. Welzl and Clarabel are not run there because neither finishes, and
what is left are two methods that both repair the factorisation of their support
instead of rebuilding it and both sweep the cloud once per iteration -- so here
the two step counts *are* comparable, and the wall clocks measure the same shape
of program.

The dimension sweep exists because the basis count is a high-variance random
variable: a single seed can put ``d = 10`` below ``d = 9``, and so can a mean
over five. It is reported as a median over `SEEDS` seeds with the worst seed
beside it, because the spread is part of the finding rather than noise to be
averaged away.
"""

import contextlib
import io
import statistics
import time
from collections.abc import Callable

import numpy as np

from cvxball import min_circle_active_set
from experiments.clarabel_ball import min_circle_clarabel
from experiments.fischer_gaertner_kutz import ball_with_counts as fgk_ball
from experiments.fischer_gaertner_kutz import min_circle_fgk
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
# The high-dimension sweep, where both methods carry their support's factorisation
# and repair it: `cvxball._frame` above `_MAINTAIN_MIN_DIM`, and section 4 of the
# pivoting paper. Welzl and Clarabel are absent because neither finishes here.
HIGH_DIMENSIONS = [1000, 2000, 4000, 8000, 16000]
HIGH_POINTS = 1000
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
    """Print the enclosure error and timing of all four methods over `MAIN_GRID`.

    The pivoting method runs on every row: unlike Welzl's recursion it does one
    vectorised sweep per iteration, so nothing about it degrades with `n` the way
    a scalar predicate loop does, and its iteration count stays in the tens.
    """
    rng = np.random.default_rng(0)
    errors = f"{'err_as':>10} {'err_fgk':>10} {'err_cl':>10} {'err_wz':>10}"
    seconds = f"{'t_as':>8} {'t_fgk':>8} {'t_cl':>8} {'t_wz':>8}"
    print(f"{'n':>7} {'d':>3} | {errors} | {seconds} | {'iters':>6} {'bases':>7}")
    for n, d, run_welzl in MAIN_GRID:
        points = rng.normal(size=(n, d))
        t_as, r_as, c_as = timed(min_circle_active_set, points)
        t_fg, r_fg, c_fg = timed(min_circle_fgk, points)
        t_cl, r_cl, c_cl = timed(min_circle_clarabel, points)
        iterations = fgk_ball(points).iterations
        row = (
            f"{n:>7} {d:>3} | "
            f"{enclosure_error(points, r_as, c_as):>10.2e} "
            f"{enclosure_error(points, r_fg, c_fg):>10.2e} "
            f"{enclosure_error(points, r_cl, c_cl):>10.2e} "
        )
        if run_welzl:
            t_wz, r_wz, c_wz = timed(min_circle_welzl, points)
            bases = ball_with_counts(points).bases
            row += (
                f"{enclosure_error(points, r_wz, c_wz):>10.2e} | "
                f"{t_as:>8.3f} {t_fg:>8.3f} {t_cl:>8.3f} {t_wz:>8.3f} | {iterations:>6} {bases:>7}"
            )
        else:
            row += f"{'-':>10} | {t_as:>8.3f} {t_fg:>8.3f} {t_cl:>8.3f} {'-':>8} | {iterations:>6} {'-':>7}"
        print(row)


def dimension_sweep() -> None:
    """Print how the work of each combinatorial method grows with `d`, over seeds.

    Welzl's basis count and the pivoting method's iteration count are the two
    language-independent measures here, and they are what makes the comparison
    mean anything: both count the algorithm's own steps rather than how fast
    CPython walks them. The two wall clocks are reported beside them so the gap
    between a step count and a running time stays visible.
    """
    header = f"{'bases (med)':>12} {'bases (max)':>12} {'fgk_it (med)':>13} {'fgk_it (max)':>13}"
    print(f"\n{'d':>3} | {header} | {'t_wz (med)':>11} {'t_fgk (med)':>12} {'t_as (med)':>11}")
    for d in SWEEP_DIMENSIONS:
        bases, welzl_times, active_times = [], [], []
        iterations, fgk_times = [], []
        for seed in range(SEEDS):
            points = np.random.default_rng(100 + seed).normal(size=(SWEEP_POINTS, d))
            start = time.perf_counter()
            ball = ball_with_counts(points, seed=seed)
            welzl_times.append(time.perf_counter() - start)
            bases.append(ball.bases)
            start = time.perf_counter()
            min_circle_active_set(points)
            active_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            min_circle_fgk(points)
            fgk_times.append(time.perf_counter() - start)
            iterations.append(fgk_ball(points).iterations)
        print(
            f"{d:>3} | {statistics.median(bases):>12.0f} {max(bases):>12d} "
            f"{statistics.median(iterations):>13.0f} {max(iterations):>13d} | "
            f"{statistics.median(welzl_times):>11.3f} {statistics.median(fgk_times):>12.4f} "
            f"{statistics.median(active_times):>11.4f}"
        )


def active_set_steps(points: np.ndarray) -> tuple[int, int]:
    """Count the active-set method's iterations, and the size of its final support.

    The solver returns ``(radius, centre)`` and nothing else -- the weights are its
    certificate, not its instrumentation -- so the count is read off its ``verbose``
    trace, which prints one line per iteration. That costs a second solve, which is
    why this is a separate call rather than folded into the timed one: printing
    inside the loop would land in the wall clock it is reported beside.

    Args:
        points: The ``(n, d)`` cloud.

    Returns:
        The ``(iterations, support_size)`` pair, the support being the one carried
        into the final iteration -- the set whose circumsphere is the answer.
    """
    trace = io.StringIO()
    with contextlib.redirect_stdout(trace):
        min_circle_active_set(points, verbose=True)
    lines = trace.getvalue().splitlines()
    return len(lines), int(lines[-1].split("support=")[1].split()[0])


def high_dimension_table() -> None:
    """Print the active-set method against the pivoting one over `HIGH_DIMENSIONS`.

    This is the regime the pivoting paper was written for and the one the other two
    references cannot enter: Welzl's basis count is hopeless past ``d ~ 11`` and
    Clarabel is asked to factor an ``n(d + 1)``-row system, so both are omitted.
    What is left is the comparison that has something at stake, since both methods
    here repair the factorisation of their support rather than rebuild it, both
    spend the rest of each iteration in one ``O(nd)`` sweep, and they approach the
    answer from opposite sides -- dual-feasible ascent against primal-feasible
    deflation.

    The step counts are comparable in this table in a way the two in
    `dimension_sweep` are not: an active-set iteration and a pivot step each cost a
    solve or projection of order ``r`` plus a sweep over all ``n``.
    """
    rng = np.random.default_rng(0)
    steps = f"{'supp':>5} {'it_as':>6} {'pivots':>7}"
    errors = f"{'err_as':>10} {'err_fgk':>10}"
    print(f"\n{'d':>6} {steps} | {errors} | {'t_as':>8} {'t_fgk':>8} {'ratio':>6}")
    for d in HIGH_DIMENSIONS:
        points = rng.normal(size=(HIGH_POINTS, d))
        t_as, r_as, c_as = timed(min_circle_active_set, points)
        t_fg, r_fg, c_fg = timed(min_circle_fgk, points)
        iterations, support = active_set_steps(points)
        pivots = fgk_ball(points).iterations
        print(
            f"{d:>6} {support:>5} {iterations:>6} {pivots:>7} | "
            f"{enclosure_error(points, r_as, c_as):>10.2e} "
            f"{enclosure_error(points, r_fg, c_fg):>10.2e} | "
            f"{t_as:>8.3f} {t_fg:>8.3f} {t_fg / t_as:>6.2f}"
        )


if __name__ == "__main__":
    main_table()
    dimension_sweep()
    high_dimension_table()
