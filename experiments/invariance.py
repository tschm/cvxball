"""Measure what each invariance guard is worth.

The active-set method of ``cvxball.solver`` is scale- and origin-invariant by
construction: it recentres the cloud, rescales it by an exact power of two, forms
squared distances by differencing rather than expanding, and sizes its feasibility
slack off the cloud's extent.  This module runs the *same* method with those
guards switchable, so the cost of dropping one can be measured rather than argued.

``_solve`` below mirrors ``cvxball.solver.min_circle_active_set`` step for step;
with every flag set it computes the same ball.  The differences are exactly the
four guards, so a row of the table this produces is the same search with one
piece of numerical care removed.

Run as ``python -m experiments.invariance`` from the repository root.
"""

from __future__ import annotations

import numpy as np

from cvxball.solver import (
    _DROP_TOL,
    _FEAS_NOISE,
    _FEAS_RTOL,
    _MAX_ITER_PER_POINT,
    _MaintainedFace,
    _RebuiltFace,
    _shrink,
    _validate,
)

_EPS = float(np.finfo(np.float64).eps)


def _sq_dist_differenced(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Squared distances formed by differencing before squaring."""
    offsets = points - centre
    out: np.ndarray = np.einsum("ij,ij->i", offsets, offsets)
    return out


def _sq_dist_expanded(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Squared distances formed as ``||p||^2 - 2 p'x + ||x||^2``."""
    out: np.ndarray = np.einsum("ij,ij->i", points, points) - 2.0 * (points @ centre) + float(centre @ centre)
    return out


def _solve(
    points: np.ndarray,
    *,
    recentre: bool = True,
    rescale: str = "pow2",
    differencing: bool = True,
    scaled_tol: bool = True,
) -> tuple[float, np.ndarray]:
    """Run the active-set method with each invariance guard switchable.

    Args:
        points: The ``(n, d)`` cloud.
        recentre: Subtract the cloud mean before iterating.
        rescale: ``"pow2"`` for the exact power-of-two normalisation, ``"exact"``
            to divide by the largest coordinate (a rounding factor), ``"none"``
            to leave the magnitude alone.
        differencing: Form squared distances by differencing rather than expanding.
        scaled_tol: Size the feasibility slack and the null-space direction off the
            cloud's extent rather than off magnitude one.

    Returns:
        The radius and centre of the ball the method arrives at.
    """
    sq_dist = _sq_dist_differenced if differencing else _sq_dist_expanded
    pts = _validate(points)
    n = pts.shape[0]

    shift = pts.mean(axis=0) if recentre else np.zeros(pts.shape[1])
    pts = pts - shift

    exponent = 0
    largest = float(np.abs(pts).max())
    if largest == 0.0:
        return 0.0, shift
    divisor = 1.0
    if rescale == "pow2":
        exponent = int(np.frexp(largest)[1])
        pts = np.ldexp(pts, -exponent)
    elif rescale == "exact":
        divisor = largest
        pts = pts / divisor

    sq_norms: np.ndarray = np.einsum("ij,ij->i", pts, pts)
    seed = int(np.argmax(sq_dist(pts, pts[0])))
    floor_scale = float(sq_norms.max()) if scaled_tol else 1.0
    noise_floor = _FEAS_NOISE * _EPS * floor_scale

    support = _MaintainedFace(pts, seed) if pts.shape[1] >= 100 else _RebuiltFace(pts, seed)
    free = support.support
    weights = np.ones(1)

    for _ in range(_MAX_ITER_PER_POINT * (n + 1)):
        face = support.face
        null_space = support.null_space()

        if null_space.size:
            centre = face.T @ weights
            gradient = -sq_dist(face, centre)
            descent = -(null_space @ (null_space.T @ gradient))
            if scaled_tol:
                biggest = float(np.abs(descent).max())
                step = descent / biggest if biggest > 0.0 else descent
            else:
                step = descent
        else:
            target = support.circumcentre_weights()
            if target.min() >= -_DROP_TOL:
                weights = np.maximum(target, 0.0)
                total = weights.sum()
                if not np.isfinite(total) or total <= 0.0:
                    raise FloatingPointError("weights are not finite")  # noqa: TRY003
                weights /= total
                centre = face.T @ weights
                dist_sq = sq_dist(pts, centre)
                radius_sq = float(dist_sq[free].max())
                radius = float(np.sqrt(max(radius_sq, 0.0)))
                worst = int(np.argmax(dist_sq))
                if dist_sq[worst] <= radius_sq * (1.0 + _FEAS_RTOL) + noise_floor:
                    return (
                        float(np.ldexp(radius, exponent)) * divisor,
                        np.ldexp(centre, exponent) * divisor + shift,
                    )
                keep = weights > _DROP_TOL
                _shrink(support, keep)
                support.insert(worst)
                weights = np.append(weights[keep], 0.0)
                continue
            step = target - weights

        binding = step < -_DROP_TOL
        ratios = np.where(binding, -weights / np.where(binding, step, -1.0), np.inf)
        alpha = float(ratios.min())
        if not np.isfinite(alpha):
            raise FloatingPointError("no support point blocks the step")  # noqa: TRY003
        weights = np.maximum(weights + alpha * step, 0.0)
        keep = weights > _DROP_TOL
        _shrink(support, keep)
        weights = weights[keep]
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("weights are not finite")  # noqa: TRY003
        weights /= total

    raise FloatingPointError("did not converge")  # noqa: TRY003


def enclosure_error(points: np.ndarray, radius: float, centre: np.ndarray) -> float:
    """Relative enclosure error ``max_i ||p_i - x|| / R - 1``.

    The measurement is itself recentred and power-of-two rescaled before any
    squaring.  Without that it underflows at extent ``1e-160`` and overflows at
    ``1e+160``, and at offset ``1e9`` its own cancellation floor sits at ``1e-7``
    -- so an unguarded metric reports the guards failing when what failed was the
    metric.

    Args:
        points: The ``(n, d)`` cloud the ball was computed for.
        radius: The returned radius.
        centre: The returned centre.

    Returns:
        The relative error; positive means the ball does not contain the cloud.
    """
    if not np.isfinite(radius) or radius <= 0.0 or not np.all(np.isfinite(centre)):
        return float("nan")
    shift = points.mean(axis=0)
    pts = points - shift
    ctr = centre - shift
    largest = max(float(np.abs(pts).max()), float(np.abs(ctr).max()))
    if largest == 0.0:
        return float("nan")
    exponent = int(np.frexp(largest)[1])
    pts = np.ldexp(pts, -exponent)
    ctr = np.ldexp(ctr, -exponent)
    scaled_radius = float(np.ldexp(radius, -exponent))
    if scaled_radius <= 0.0 or not np.isfinite(scaled_radius):
        return float("nan")
    far = float(np.sqrt(np.max(np.einsum("ij,ij->i", pts - ctr, pts - ctr))))
    return far / scaled_radius - 1.0


# Cloud extent and offset from the origin, one column of the table each.
SCENARIOS: list[tuple[str, float, float]] = [
    ("1", 1.0, 0.0),
    ("1", 1.0, 1e6),
    ("1", 1.0, 1e9),
    ("1e-20", 1e-20, 0.0),
    ("1e-160", 1e-160, 0.0),
    ("1e+160", 1e160, 0.0),
]

# Guard configurations, one row of the table each.
CONFIGS: list[tuple[str, dict[str, object]]] = [
    ("all guards", {}),
    ("expanded distances", {"differencing": False}),
    ("no recentring", {"recentre": False}),
    ("no recentring, expanded", {"recentre": False, "differencing": False}),
    ("no rescaling", {"rescale": "none"}),
    ("no rescaling, unit tol.", {"rescale": "none", "scaled_tol": False}),
    ("rescale by extent", {"rescale": "exact"}),
    (
        "no guards",
        {"recentre": False, "differencing": False, "rescale": "none", "scaled_tol": False},
    ),
]


def run(n: int = 500, d: int = 10, seed: int = 0) -> list[tuple[str, list[str]]]:
    """Run every configuration against every scenario.

    Args:
        n: Number of points per cloud.
        d: Ambient dimension.
        seed: Seed for the standard normal cloud, shared by every cell.

    Returns:
        One ``(label, cells)`` pair per configuration.
    """
    base = np.random.default_rng(seed).standard_normal((n, d))
    base = base / float(np.abs(base).max())

    rows: list[tuple[str, list[str]]] = []
    for label, kwargs in CONFIGS:
        cells: list[str] = []
        for _, extent, offset in SCENARIOS:
            cloud = base * extent + offset
            with np.errstate(all="ignore"):
                try:
                    radius, centre = _solve(cloud, **kwargs)  # type: ignore[arg-type]
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    cells.append("fail")
                    continue
            if not np.isfinite(radius):
                cells.append("inf")
            elif radius == 0.0:
                cells.append("R=0")
            else:
                err = enclosure_error(cloud, radius, centre)
                cells.append("nan" if np.isnan(err) else f"{err:.1e}")
        rows.append((label, cells))
    return rows


if __name__ == "__main__":
    header = [f"{e}@{o:g}" for _, e, o in SCENARIOS]
    print(f"{'':28s}" + "".join(f"{h:>14s}" for h in header))
    for label, cells in run():
        print(f"{label:28s}" + "".join(f"{c:>14s}" for c in cells))
