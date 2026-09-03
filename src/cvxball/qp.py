"""A dense active-set method for small convex quadratic programs.

This is the numerical core behind :mod:`cvxball.active_set`, kept separate from
the Clarabel-shaped front end so that it can be read, tested and reused as what
it is: a function over plain NumPy arrays.  It solves

    minimise   (1/2) x' P x + q' x
    subject to E x == f,  G x <= h

with ``P`` symmetric positive definite, by the dual active-set method of
Goldfarb and Idnani (*A numerically stable dual method for solving strictly
convex quadratic programs*, Mathematical Programming 27, 1983).

Why a dual method.  A *primal* active-set method needs a feasible starting point
and so needs a phase-1 problem to find one; the dual method starts at the
unconstrained minimiser ``x = -P⁻¹q``, which is always available, and adds
constraints one at a time while keeping the multipliers non-negative.  Every
iterate is the exact solution of the problem restricted to the current active
set, so the run ends *at* the optimal face rather than approaching it: on a
problem whose solution is determined by which constraints are tight -- a
portfolio at its box bounds, say -- the answer comes back to machine precision,
and the active set itself is exact rather than a set of near-zero slacks that
have to be rounded.

The linear algebra here is deliberately dense and recomputed from scratch each
iteration: one Cholesky factorisation of ``P`` up front, then per iteration a
``k x k`` solve with ``k`` the size of the active set.  That is the right trade
for the problems this is aimed at -- tens to a few hundred variables, dense
covariance-shaped ``P``, dense constraint rows -- where the factorisation
updates that make Goldfarb-Idnani asymptotically cheaper cost more in complexity
than they save in time.  It is the wrong trade for large sparse cone programs,
which is what Clarabel is for.
"""

from typing import Any, NamedTuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Default feasibility tolerance, matching Clarabel's `tol_feas`.  It is applied to
# a *relative* violation (see `_violations`), so it carries no unit and no implied
# problem scale.
FEAS_TOL = 1e-8
# A candidate constraint offers no primal direction once the curvature left after
# projecting out the active set falls to this fraction of the unprojected
# curvature.  Both sides are quadratic forms in the same row, so the ratio is
# dimensionless and the test is invariant to how the row is scaled.
_CURVATURE_TOL = 1e-11
# A multiplier decreasing more slowly than this (relative to the largest rate in
# the same step) is not treated as blocking: it would otherwise contribute a step
# length of ~1e16 and swamp the real minimum with rounding noise.
_RATE_TOL = 1e-12
# An eigenvalue this small, relative to the largest, counts as zero when deciding
# whether P is definite, singular, or indefinite.
_DEFINITENESS_TOL = 1e-12
# Safety net.  The method is finite in exact arithmetic, so the budget only has to
# be generous enough that hitting it means degeneracy rather than a hard problem.
_MAX_PIVOTS_PER_ROW = 20


class QPSolution(NamedTuple):
    """The outcome of one run of :func:`solve_qp`.

    The multipliers are given over *all* rows, not just the active ones: ``y_in``
    is zero on every inactive inequality, which is exactly the complementarity
    condition, and makes the vector directly usable as a conic dual variable.
    """

    x: np.ndarray
    y_eq: np.ndarray
    y_in: np.ndarray
    iterations: int
    status: str


def symmetrise(p_mat: np.ndarray) -> np.ndarray:
    """Return the symmetric matrix that ``p_mat`` describes.

    Clarabel takes the *upper triangle* of ``P`` and ignores the rest, so callers
    reasonably pass either the upper triangle alone or the full symmetric matrix.
    Both have to mean the same thing here, and a caller who passes the triangle to
    a solver that reads it whole would otherwise get an answer to a problem with
    half the off-diagonal curvature -- a wrong answer with no error.

    Args:
        p_mat: A dense ``(n, n)`` array holding either the full symmetric matrix
               or only its upper triangle.

    Returns:
        The dense symmetric ``(n, n)`` matrix.
    """
    lower = np.tril(p_mat, -1)
    if not lower.any():
        upper = np.triu(p_mat, 1)
        return np.asarray(np.diag(np.diag(p_mat)) + upper + upper.T)
    return np.asarray(0.5 * (p_mat + p_mat.T))


def _factorise(p_mat: np.ndarray, regularization: float) -> tuple[Any, np.ndarray]:
    """Factorise ``P``, shifting it first if -- and only if -- it is not definite.

    The dual method starts from the unconstrained minimiser ``-P⁻¹q``, so it needs
    a definite ``P``.  A merely semidefinite one (a factor-model covariance with
    fewer factors than assets, or the zero matrix of a linear program) has no such
    point, and Clarabel meets the same obstacle in its KKT matrix and answers it
    with static regularization; this does the same, with the shift taken from the
    same settings.

    The decision is made on the eigenvalues rather than by trying a Cholesky and
    seeing whether it fails, because those two failures need opposite answers: a
    *singular* ``P`` is a convex problem that a small ridge solves, while an
    *indefinite* ``P`` is not a convex problem at all and no ridge should be
    allowed to hide it -- shifting until the factorisation succeeds would silently
    answer a different, convexified question.  Deciding on the spectrum costs a
    symmetric eigendecomposition, a few times the price of the Cholesky it
    replaces, which at these problem sizes is worth an unambiguous answer.

    Args:
        p_mat: The dense symmetric ``(n, n)`` objective matrix.
        regularization: The relative diagonal shift available to make a singular
                        ``P`` definite; ``0.0`` refuses to shift.

    Returns:
        A tuple ``(factor, p_used)`` of the Cholesky factor as
        :func:`scipy.linalg.cho_factor` returns it, and the matrix it factorised
        -- which is ``p_mat`` itself unless a shift was needed.  Both are returned
        because everything downstream has to use the *same* matrix: solving one
        subproblem with a shifted ``P`` and another with the original produces a
        primal-dual pair that solves neither problem.

    Raises:
        ValueError: If ``P`` is indefinite, or is singular while regularization is
                    switched off.
    """
    eigenvalues = np.linalg.eigvalsh(p_mat)
    smallest = float(eigenvalues[0])
    scale = max(1.0, float(np.abs(eigenvalues).max(initial=0.0)))

    if smallest < -_DEFINITENESS_TOL * scale:
        raise ValueError(  # noqa: TRY003
            f"P has a negative eigenvalue ({smallest:.3g}), so the objective is not convex "
            f"and its minimum is not where any solver's optimality conditions say it is."
        )

    if smallest > _DEFINITENESS_TOL * scale:
        return cho_factor(p_mat, lower=True), p_mat

    if regularization <= 0.0:
        raise ValueError(  # noqa: TRY003
            "P is positive semidefinite but not definite, and static regularization is "
            "disabled. This dual active-set method starts from the unconstrained "
            "minimiser, which a singular P does not have. Add a small ridge to P, or "
            "enable settings.static_regularization_enable."
        )

    # Lift the spectrum clear of zero: the requested ridge, plus whatever it takes
    # to undo a slightly negative eigenvalue that is rounding noise on a
    # semidefinite matrix rather than genuine non-convexity.
    delta = regularization * scale - min(smallest, 0.0) + _DEFINITENESS_TOL * scale
    shifted = p_mat + delta * np.eye(p_mat.shape[0])
    return cho_factor(shifted, lower=True), shifted


def _solve_active(m_mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``m_mat @ y == rhs`` for the (positive semidefinite) active-set system.

    ``m_mat`` is ``R P⁻¹ R'`` for the active constraint rows ``R``, so it is
    definite exactly when those rows are linearly independent.  Dependent rows are
    a degenerate active set, not an error -- three box bounds meeting at a corner
    of a two-dimensional face, say -- so the singular case falls back to the
    minimum-norm least-squares solution and lets the method keep pivoting.

    Args:
        m_mat: The ``(k, k)`` positive semidefinite active-set matrix.
        rhs: The ``(k,)`` right-hand side.

    Returns:
        The ``(k,)`` solution.
    """
    try:
        return np.asarray(cho_solve(cho_factor(m_mat, lower=True), rhs))
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(m_mat, rhs, rcond=None)[0])


def _violations(a_in: np.ndarray, b_in: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Measure how far ``x`` breaks each inequality, relative to the row's own size.

    The absolute residual ``g_i'x - h_i`` is not comparable across rows: a budget
    row summing weights and a row bounding a notional in currency units differ by
    whatever the units differ by, and the largest absolute violation would then
    just name the row with the biggest numbers.  Dividing by the magnitude of the
    terms that went into the row -- ``sum_j |g_ij x_j|`` and ``|h_i|`` -- both makes
    the comparison meaningful and ties the tolerance to the row's own rounding
    error rather than to an assumed problem scale.

    Args:
        a_in: The ``(m, n)`` inequality matrix ``G``.
        b_in: The ``(m,)`` right-hand side ``h``.
        x: The current ``(n,)`` iterate.

    Returns:
        The ``(m,)`` relative violations: positive where the row is broken.
    """
    residual = a_in @ x - b_in
    scale = np.maximum(np.abs(a_in) @ np.abs(x), np.abs(b_in))
    return np.asarray(residual / np.where(scale > 0.0, scale, 1.0))


def _start_point(chol: Any, q: np.ndarray, a_eq: np.ndarray, b_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the objective over the equality constraints alone.

    This is where the dual method begins, and every equality is in the active set
    from here on: an equality multiplier has no sign to lose, so no step can block
    on it and no drop can remove it.  Eliminating ``x`` from ``P x + q + E'nu = 0``
    and ``E x = f`` leaves ``(E P⁻¹ E') nu = -(E P⁻¹ q + f)`` -- the same small
    system every later pivot solves, so the start needs no separate machinery.

    Args:
        chol: The Cholesky factor of ``P`` from :func:`_factorise`.
        q: The ``(n,)`` linear objective term.
        a_eq: The ``(m_eq, n)`` equality matrix.
        b_eq: The ``(m_eq,)`` equality right-hand side.

    Returns:
        The tuple ``(x, nu)`` of the minimiser and the equality multipliers.  If
        the equalities contradict each other there is no minimiser, and ``x`` is
        then the least-squares compromise the caller must detect and reject.
    """
    pinv_q = cho_solve(chol, q)
    if a_eq.shape[0] == 0:
        return -pinv_q, np.zeros(0)

    pinv_eq_t = cho_solve(chol, a_eq.T)
    nu = _solve_active(a_eq @ pinv_eq_t, -(a_eq @ pinv_q + b_eq))
    return -(pinv_q + pinv_eq_t @ nu), nu


def _next_candidate(a_in: np.ndarray, b_in: np.ndarray, x: np.ndarray, active: np.ndarray, tol: float) -> int:
    """Pick the inactive inequality that ``x`` breaks worst.

    Choosing the *worst* violation rather than the first is what keeps the pivot
    count near the number of constraints that end up binding: it is the same
    greedy choice a simplex method makes with the most negative reduced cost.

    Args:
        a_in: The ``(m, n)`` inequality matrix.
        b_in: The ``(m,)`` inequality right-hand side.
        x: The current iterate.
        active: The boolean mask of currently active rows.
        tol: The relative violation below which a row counts as satisfied.

    Returns:
        The row index, or ``-1`` when every inactive row is satisfied -- which,
        the iterate being dual feasible throughout, means optimal.
    """
    if a_in.shape[0] == 0:
        return -1

    relative = _violations(a_in, b_in, x)
    relative[active] = -np.inf
    worst = int(np.argmax(relative))
    return worst if relative[worst] > tol else -1


def _direction(chol: Any, rows: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find how ``x`` and the multipliers move as one constraint's multiplier grows.

    Differentiating the active-set KKT system -- ``P x + q + R' lambda = 0`` with
    ``R x`` held fixed -- with respect to the candidate's own multiplier gives
    ``M dlambda = -R P⁻¹ g`` with ``M = R P⁻¹ R'``, and then
    ``dx = -P⁻¹ (g + R' dlambda)``.

    Args:
        chol: The Cholesky factor of ``P`` from :func:`_factorise`.
        rows: The ``(k, n)`` currently active rows, equalities first.
        row: The ``(n,)`` candidate row ``g``.

    Returns:
        The tuple ``(direction, d_lambda, rate, curvature)``: the primal
        direction, the multiplier direction, the rate at which the candidate's own
        violation falls along it, and the curvature of ``P`` along the candidate
        row before the active set is projected out.  ``rate`` is minus the
        curvature that survives that projection, so it is never positive, and it
        is zero exactly when the candidate row is a combination of the active
        rows.  Comparing the two is a scale-free test for "no primal direction
        left", which is what the caller needs it for.
    """
    pinv_row = cho_solve(chol, row)
    curvature = float(row @ pinv_row)

    if rows.shape[0] == 0:
        direction = -pinv_row
        return direction, np.zeros(0), float(row @ direction), curvature

    pinv_rows_t = cho_solve(chol, rows.T)
    d_lambda = -_solve_active(rows @ pinv_rows_t, rows @ pinv_row)
    direction = -(pinv_row + pinv_rows_t @ d_lambda)
    return direction, d_lambda, float(row @ direction), curvature


def solve_qp(
    p_mat: np.ndarray,
    q: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_in: np.ndarray,
    b_in: np.ndarray,
    *,
    tol_feas: float = FEAS_TOL,
    regularization: float = 0.0,
    verbose: bool = False,
) -> QPSolution:
    """Minimise a convex quadratic subject to equalities and inequalities.

    The method (Goldfarb-Idnani; see the module docstring) walks the dual feasible
    set.  It starts at the minimiser over the equalities alone and then repeatedly

    1. picks the worst-violated inequality ``p`` (:func:`_violations`),
    2. computes the direction in which ``x`` and the multipliers move as ``p``'s own
       multiplier is raised from zero -- one ``k x k`` solve, where ``k`` counts the
       currently active rows,
    3. moves as far as it can: either far enough to satisfy ``p``, which activates
       it, or until an active constraint's multiplier reaches zero, which
       deactivates *that* one and leaves ``p`` still pending.

    Every iterate satisfies the KKT stationarity condition and has non-negative
    multipliers; what the loop is chasing is primal feasibility.  So the invariant
    is the opposite of an interior-point method's, and the run stops at an exact
    optimal active set instead of at a tolerance.

    Args:
        p_mat: The dense symmetric ``(n, n)`` objective matrix, positive definite
               (or made so by ``regularization``).  Passing only its upper
               triangle is accepted, as Clarabel accepts it.
        q: The ``(n,)`` linear objective term.
        a_eq: The ``(m_eq, n)`` equality matrix ``E``.  May have zero rows.
        b_eq: The ``(m_eq,)`` equality right-hand side ``f``.
        a_in: The ``(m_in, n)`` inequality matrix ``G``.  May have zero rows.
        b_in: The ``(m_in,)`` inequality right-hand side ``h``.
        tol_feas: The relative violation below which a row counts as satisfied.
        regularization: The diagonal shift to fall back on if ``P`` is singular;
                        ``0.0`` refuses to shift.  See :func:`_factorise`.
        verbose: If ``True``, print one line per pivot.

    Returns:
        The :class:`QPSolution`.  Its ``status`` is ``"Solved"``,
        ``"PrimalInfeasible"`` if a constraint is found that no move can satisfy,
        or ``"MaxIterations"`` if the pivot budget runs out.

    Raises:
        ValueError: If ``P`` cannot be factorised even with regularisation (see
                    :func:`_factorise`).

    Example:
        The projection of the origin onto the simplex-like set ``x1 + x2 == 1``,
        ``x >= 0`` -- a miniature portfolio problem -- is the point ``(0.5, 0.5)``,
        and this returns it exactly rather than to a tolerance.

        >>> import numpy as np
        >>> from cvxball.qp import solve_qp
        >>> solution = solve_qp(
        ...     np.eye(2), np.zeros(2),
        ...     np.array([[1.0, 1.0]]), np.array([1.0]),
        ...     -np.eye(2), np.zeros(2),
        ... )
        >>> solution.status
        'Solved'
        >>> solution.x
        array([0.5, 0.5])
    """
    chol, _ = _factorise(symmetrise(p_mat), regularization)
    m_eq, m_in = a_eq.shape[0], a_in.shape[0]
    budget = _MAX_PIVOTS_PER_ROW * (m_in + m_eq + q.size + 1)

    x, nu = _start_point(chol, q, a_eq, b_eq)
    if m_eq and not np.allclose(a_eq @ x, b_eq, rtol=tol_feas, atol=tol_feas * float(np.abs(b_eq).max(initial=1.0))):
        # The equalities contradict each other, so `_start_point` returned the
        # least-squares compromise rather than a minimiser, and there is nothing
        # for the loop below to improve on.
        return QPSolution(x, nu, np.zeros(m_in), 0, "PrimalInfeasible")

    mu = np.zeros(m_in)
    active = np.zeros(m_in, dtype=bool)
    candidate = -1

    for iteration in range(budget):
        if candidate < 0:
            candidate = _next_candidate(a_in, b_in, x, active, tol_feas)
            if candidate < 0:
                # Every row is satisfied, and every iterate of this method is
                # already dual feasible: that pair of facts is optimality.
                if verbose:
                    print(f"[{iteration:4d}] optimal: active={int(active.sum()):3d} of {m_in}")
                return QPSolution(x, nu, mu, iteration, "Solved")

        row = a_in[candidate]
        direction, d_lambda, rate, curvature = _direction(chol, np.vstack([a_eq, a_in[active]]), row)
        exhausted = -rate <= _CURVATURE_TOL * curvature

        # --- Step length: as far as dual feasibility and the candidate allow ----
        # Raising mu[candidate] drives the active multipliers along `d_mu`; the ones
        # that decrease bound the step, since a multiplier may not go negative.
        d_mu = d_lambda[m_eq:]
        largest_rate = float(np.abs(d_mu).max(initial=0.0))
        blocking = d_mu < -_RATE_TOL * max(1.0, largest_rate)
        ratios = np.where(blocking, mu[active] / np.where(blocking, -d_mu, 1.0), np.inf)
        drop_step = float(ratios.min(initial=np.inf))

        if exhausted and not np.isfinite(drop_step):
            # The candidate row is a combination of the active rows, so no move can
            # reduce its violation, and no multiplier stands in the way of raising
            # its own without bound: the dual is unbounded, so the primal is
            # infeasible. This is where the method *proves* infeasibility rather
            # than running out of patience -- an interior-point method infers it
            # from a residual that stops improving.
            if verbose:
                print(f"[{iteration:4d}] primal infeasible: row {candidate} cannot be satisfied")
            return QPSolution(x, nu, mu, iteration, "PrimalInfeasible")

        # The candidate's violation falls at `rate` per unit of its own multiplier,
        # so this is the step that closes it exactly.
        residual = float(row @ x - b_in[candidate])
        full_step = np.inf if exhausted else residual / -rate
        step = min(drop_step, full_step)

        x = x + step * direction
        nu = nu + step * d_lambda[:m_eq]
        mu[active] = np.maximum(mu[active] + step * d_mu, 0.0)
        mu[candidate] += step

        if full_step <= drop_step:
            active[candidate] = True
            if verbose:
                print(f"[{iteration:4d}] add row {candidate:4d} step={step:.6g} active={int(active.sum()):3d}")
            candidate = -1
        else:
            # A multiplier reached zero first: that constraint leaves the active set
            # and the candidate stays pending, with its multiplier already partly
            # raised. The active set shrinks here, which is what keeps the method
            # finite -- it never revisits an active set it has already left.
            dropped = int(np.flatnonzero(active)[int(np.argmin(ratios))])
            active[dropped] = False
            mu[dropped] = 0.0
            if verbose:
                print(f"[{iteration:4d}] drop row {dropped:4d} step={step:.6g} active={int(active.sum()):3d}")

    return QPSolution(x, nu, mu, budget, "MaxIterations")
