"""Tests for the dense active-set method for convex quadratic programs.

Clarabel is the oracle throughout: it solves the same programs by a completely
different method, so agreement between the two is evidence about the answer
rather than about a shared implementation.  Where the two are expected to
*disagree* -- the active-set method returning the optimal face exactly where an
interior-point method approaches it -- the test says so and pins the exact value.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import cho_factor

import cvxball.qp as qp_module
from cvxball.qp import solve_qp, symmetrise


def _clarabel_qp(p_mat, q, a_eq, b_eq, a_in, b_in):
    """Solve the same QP with Clarabel and return its solution object."""
    import clarabel

    a_mat = sp.csc_matrix(np.vstack([a_eq, a_in]))
    b = np.concatenate([b_eq, b_in])
    cones = [clarabel.ZeroConeT(a_eq.shape[0]), clarabel.NonnegativeConeT(a_in.shape[0])]
    settings = clarabel.DefaultSettings.default()
    settings.verbose = False
    return clarabel.DefaultSolver(sp.csc_matrix(np.triu(p_mat)), q, a_mat, b, cones, settings).solve()


def _assert_kkt(p_mat, q, a_eq, b_eq, a_in, b_in, solution, tol=1e-9):
    """Assert the solution satisfies the KKT conditions of the QP.

    This is the check that does not depend on another solver: stationarity,
    primal feasibility, dual feasibility and complementary slackness together
    certify optimality for a convex program, so a solution that passes is optimal
    whatever Clarabel says.
    """
    x = solution.x
    scale = max(1.0, float(np.abs(q).max(initial=0.0)), float(np.abs(b_in).max(initial=0.0)))

    stationarity = p_mat @ x + q + a_eq.T @ solution.y_eq + a_in.T @ solution.y_in
    assert np.abs(stationarity).max(initial=0.0) <= tol * scale, "stationarity violated"
    assert np.abs(a_eq @ x - b_eq).max(initial=0.0) <= tol * scale, "equality violated"
    assert (a_in @ x - b_in).max(initial=0.0) <= tol * scale, "inequality violated"
    assert solution.y_in.min(initial=0.0) >= -tol, "negative inequality multiplier"
    slack = b_in - a_in @ x
    assert np.abs(solution.y_in * slack).max(initial=0.0) <= tol * scale, "complementarity violated"


def _random_qp(seed):
    """Draw a random, guaranteed-feasible dense QP of the shape finance produces."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 15))
    factors = rng.normal(size=(n, 10))
    p_mat = factors @ factors.T / 10 + np.diag(rng.random(n) * 0.5 + 0.05)
    q = rng.normal(size=n)

    interior = rng.normal(size=n)
    a_eq = rng.normal(size=(int(rng.integers(0, 4)), n))
    a_in = rng.normal(size=(int(rng.integers(1, 15)), n))
    # Right-hand sides measured from a point that satisfies them all, with some
    # inequalities exactly tight there and the rest slack, so the optimum has an
    # active set that has to be found rather than guessed.
    slack = rng.random(a_in.shape[0]) * rng.integers(0, 2, a_in.shape[0])
    return p_mat, q, a_eq, a_eq @ interior, a_in, a_in @ interior + slack


# --- Agreement with Clarabel ---------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_matches_clarabel_on_random_dense_qps(seed) -> None:
    """On random feasible QPs the two solvers agree, and the KKT check passes."""
    p_mat, q, a_eq, b_eq, a_in, b_in = _random_qp(seed)
    solution = solve_qp(p_mat, q, a_eq, b_eq, a_in, b_in)
    reference = _clarabel_qp(p_mat, q, a_eq, b_eq, a_in, b_in)

    assert solution.status == "Solved"
    _assert_kkt(p_mat, q, a_eq, b_eq, a_in, b_in, solution)

    objective = 0.5 * solution.x @ p_mat @ solution.x + q @ solution.x
    # The active-set objective may come out *below* Clarabel's, which is not a
    # disagreement: an interior-point iterate stops just short of the boundary.
    assert objective <= reference.obj_val + 1e-7
    assert objective == pytest.approx(reference.obj_val, abs=1e-7)
    assert solution.x == pytest.approx(np.asarray(reference.x), abs=1e-5)


def test_drop_step_is_exercised_and_still_optimal(capsys) -> None:
    """A constraint that becomes active and later leaves does not spoil the answer.

    The interesting iterations are the ones where a multiplier reaches zero before
    the pending constraint is satisfied: the active set shrinks, the pending
    constraint stays pending, and only the dual has moved.  This 3-variable
    problem takes that path (seed found by search), so the log is asserted to
    contain a drop -- otherwise the test could pass while silently covering only
    the easy path.
    """
    p_mat, q, a_eq, b_eq, a_in, b_in = _random_qp(4)
    p_mat = np.array([[3.3233, -0.1515, -2.2952], [-0.1515, 3.2287, -0.6466], [-2.2952, -0.6466, 3.0971]])
    q = np.array([0.2418, 0.2354, 1.5756])
    a_eq, b_eq = np.zeros((0, 3)), np.zeros(0)
    a_in = np.array(
        [
            [0.3166, 0.5105, -1.4931],
            [2.2527, -1.9156, 1.1018],
            [-0.3299, -0.8806, -0.6563],
            [-0.672, 0.3802, -0.1101],
            [1.4826, -1.8296, -0.0031],
        ]
    )
    b_in = np.array([-0.8921, 0.7759, -2.1181, -0.3437, 0.2102])

    solution = solve_qp(p_mat, q, a_eq, b_eq, a_in, b_in, verbose=True)
    log = capsys.readouterr().out

    assert "drop" in log, "this problem is supposed to exercise the drop path"
    assert "add" in log
    assert solution.status == "Solved"
    _assert_kkt(p_mat, q, a_eq, b_eq, a_in, b_in, solution)
    assert solution.x == pytest.approx(np.asarray(_clarabel_qp(p_mat, q, a_eq, b_eq, a_in, b_in).x), abs=1e-6)


# --- What an exact active set buys --------------------------------------------


def test_projection_onto_the_simplex_is_exact() -> None:
    """The fully-invested, long-only minimum-variance point lands to machine precision.

    Two symmetric assets: the answer is the equally weighted portfolio. What is
    asserted is the *precision*, at 1e-15 rather than the 1e-8 an interior-point
    method is entitled to -- this method solves the equality-constrained
    subproblem on the optimal face, so the only error left is the rounding of one
    small linear solve.
    """
    solution = solve_qp(
        np.array([[0.04, 0.01], [0.01, 0.04]]),
        np.zeros(2),
        np.ones((1, 2)),
        np.array([1.0]),
        -np.eye(2),
        np.zeros(2),
    )
    assert solution.status == "Solved"
    assert solution.x == pytest.approx([0.5, 0.5], abs=1e-15)


def test_binding_box_bounds_are_hit_exactly() -> None:
    """Weights pinned to a position cap sit on the cap, and are known to be on it.

    This is the property an active-set method has and an interior-point method
    does not: the *set* of binding constraints is a discrete answer, returned
    exactly, so "which names are at the cap" needs no threshold. Here it is
    checked two ways that must agree -- the weights that equal the cap, and the
    cap rows carrying a positive multiplier.
    """
    n, cap = 6, 1.0 / 3.0
    rng = np.random.default_rng(11)
    factors = rng.normal(size=(n, 3))
    p_mat = factors @ factors.T / 3 + np.diag(rng.random(n) * 0.1 + 0.02)
    expected_return = np.arange(n, dtype=float) / n

    solution = solve_qp(
        p_mat,
        -expected_return,
        np.ones((1, n)),
        np.array([1.0]),
        np.vstack([np.eye(n), -np.eye(n)]),
        np.concatenate([np.full(n, cap), np.zeros(n)]),
    )

    assert solution.status == "Solved"
    at_cap = np.isclose(solution.x, cap, rtol=1e-14, atol=0.0)
    assert at_cap.any(), "this problem is supposed to push weights onto the cap"
    # The first n inequality rows are the caps, the rest the long-only bounds.
    assert (solution.y_in[:n] > 0.0).tolist() == at_cap.tolist()
    assert solution.x.sum() == pytest.approx(1.0, abs=1e-14)
    # The long-only bounds hold to machine precision: a weight held at zero is a
    # linear solve away from zero, not a substituted constant.
    assert solution.x.min() >= -1e-14


def test_equality_only_problem_matches_the_closed_form() -> None:
    """With only a budget row, the minimum-norm solution is the equal weighting."""
    n = 5
    solution = solve_qp(np.eye(n), np.zeros(n), np.ones((1, n)), np.array([1.0]), np.zeros((0, n)), np.zeros(0))
    assert solution.status == "Solved"
    assert solution.x == pytest.approx([1.0 / n] * n, abs=1e-15)
    assert solution.iterations == 0


def test_unconstrained_problem_is_the_newton_step() -> None:
    """With no constraints at all the answer is -P^-1 q, in zero pivots."""
    p_mat = np.array([[2.0, 0.5], [0.5, 1.0]])
    q = np.array([1.0, -2.0])
    solution = solve_qp(p_mat, q, np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), np.zeros(0))

    assert solution.status == "Solved"
    assert solution.x == pytest.approx(np.linalg.solve(p_mat, -q), abs=1e-14)
    assert solution.iterations == 0


# --- Infeasibility: proved, not guessed ---------------------------------------


def test_contradictory_inequalities_are_primal_infeasible() -> None:
    """``x >= 1`` together with ``x <= -1`` is reported as primal infeasible."""
    solution = solve_qp(
        np.eye(1),
        np.zeros(1),
        np.zeros((0, 1)),
        np.zeros(0),
        np.array([[1.0], [-1.0]]),
        np.array([-1.0, -1.0]),
    )
    assert solution.status == "PrimalInfeasible"


def test_inconsistent_equalities_are_primal_infeasible() -> None:
    """Two copies of one row with different right-hand sides cannot be satisfied."""
    solution = solve_qp(
        np.eye(2),
        np.zeros(2),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([1.0, 2.0]),
        np.zeros((0, 2)),
        np.zeros(0),
    )
    assert solution.status == "PrimalInfeasible"


def test_unsatisfiable_zero_row_is_primal_infeasible(capsys) -> None:
    """A row of zeros bounded below zero is infeasible, and has no direction to move.

    It is the degenerate case of the infeasibility test: the row offers no
    curvature at all, so the branch that decides between "move" and "prove
    infeasible" is entered with nothing to move along. The log is asserted too --
    it is the only place a run says *which* row it could not satisfy, which is the
    first thing anyone debugging an infeasible model wants to know.
    """
    solution = solve_qp(
        np.eye(2), np.zeros(2), np.zeros((0, 2)), np.zeros(0), np.zeros((1, 2)), np.array([-1.0]), verbose=True
    )
    assert solution.status == "PrimalInfeasible"
    assert "primal infeasible: row 0" in capsys.readouterr().out


def test_the_active_set_solve_falls_back_when_cholesky_refuses() -> None:
    """A rank-deficient active-set system is answered, not raised on.

    In practice LAPACK factorises a *semidefinite* system anyway -- the last pivot
    comes out as rounding noise rather than as a failure -- so the fallback is
    reached only when the system is worse than that. It still has to be there: the
    alternative is a `LinAlgError` escaping from the middle of a pivot, which tells
    a caller nothing about their program. This calls it directly, since the
    condition that triggers it cannot be produced through `solve_qp`.
    """
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        cho_factor(indefinite, lower=True)

    rhs = np.array([1.0, 1.0])
    solution = qp_module._solve_active(indefinite, rhs)
    assert indefinite @ solution == pytest.approx(rhs, abs=1e-12)


def test_duplicate_equality_rows_are_tolerated() -> None:
    """A redundant equality row leaves the active-set system singular, not wrong.

    Repeating a constraint is a plausible way to build a model, and it makes both
    the initial KKT matrix and the active-set matrix singular. Both fall back to a
    minimum-norm solve, which is a genuine solution here: the two multipliers
    share the one shadow price between them.
    """
    a_eq = np.array([[1.0, 1.0], [1.0, 1.0]])
    solution = solve_qp(
        np.eye(2),
        np.array([-1.0, 0.0]),
        a_eq,
        np.array([1.0, 1.0]),
        -np.eye(2),
        np.array([0.0, -0.3]),
    )
    assert solution.status == "Solved"
    assert solution.x == pytest.approx([0.7, 0.3], abs=1e-12)
    _assert_kkt(
        np.eye(2), np.array([-1.0, 0.0]), a_eq, np.array([1.0, 1.0]), -np.eye(2), np.array([0.0, -0.3]), solution
    )


# --- The objective matrix ------------------------------------------------------


def test_upper_triangle_and_full_matrix_mean_the_same_thing() -> None:
    """Passing only P's upper triangle gives the same answer as passing all of it.

    Clarabel reads the upper triangle and ignores the rest, so callers write
    either; a solver that read the triangle as a full matrix would silently halve
    every off-diagonal covariance.
    """
    p_mat = np.array([[0.04, 0.01], [0.01, 0.09]])
    q = np.array([-0.1, -0.05])
    args = (np.ones((1, 2)), np.array([1.0]), -np.eye(2), np.zeros(2))

    full = solve_qp(p_mat, q, *args)
    triangular = solve_qp(np.triu(p_mat), q, *args)
    assert triangular.x.tolist() == full.x.tolist()
    assert symmetrise(np.triu(p_mat)).tolist() == p_mat.tolist()


def test_singular_p_is_refused_unless_regularized() -> None:
    """A semidefinite P has no unconstrained minimiser, and the method says so.

    With regularisation switched off this is an error rather than a status: the
    method never starts. With a shift it runs, and the shift is small enough that
    the answer is the one the unregularised problem would have had.
    """
    p_mat = np.diag([1.0, 0.0])
    q = np.array([0.0, -1.0])
    args = (np.zeros((0, 2)), np.zeros(0), np.eye(2), np.array([1.0, 1.0]))

    with pytest.raises(ValueError, match="semidefinite but not definite"):
        solve_qp(p_mat, q, *args)

    solution = solve_qp(p_mat, q, *args, regularization=1e-8)
    assert solution.status == "Solved"
    assert solution.x == pytest.approx([0.0, 1.0], abs=1e-6)


def test_indefinite_p_is_refused_even_with_regularization() -> None:
    """A non-convex objective is refused rather than quietly convexified.

    Regularisation exists to make a *singular* P factorisable. Applying it until
    an indefinite P factorises would answer a different question, so the two cases
    are separated on the spectrum and only the first one gets a shift.
    """
    with pytest.raises(ValueError, match="not convex"):
        solve_qp(
            np.diag([1.0, -1.0]),
            np.zeros(2),
            np.zeros((0, 2)),
            np.zeros(0),
            np.eye(2),
            np.ones(2),
            regularization=1e-8,
        )


# --- Scale invariance and the safety net --------------------------------------


def test_row_scaling_does_not_change_the_answer() -> None:
    """Multiplying a constraint row by 1e10 leaves the solution unchanged.

    The violation test is relative to the size of the terms in the row, so the
    row with the largest numbers does not automatically become the one the method
    chases. A solver comparing absolute residuals across rows would pick its
    pivots by units.
    """
    p_mat = np.eye(3)
    q = np.array([-1.0, -2.0, -3.0])
    a_in = np.vstack([np.eye(3), np.ones((1, 3))])
    b_in = np.array([0.5, 0.5, 0.5, 1.0])

    plain = solve_qp(p_mat, q, np.zeros((0, 3)), np.zeros(0), a_in, b_in)
    scaled_rows = a_in.copy()
    scaled_rhs = b_in.copy()
    scaled_rows[3] *= 1e10
    scaled_rhs[3] *= 1e10
    scaled = solve_qp(p_mat, q, np.zeros((0, 3)), np.zeros(0), scaled_rows, scaled_rhs)

    assert plain.status == scaled.status == "Solved"
    assert scaled.x == pytest.approx(plain.x, abs=1e-12)


def test_exhausted_pivot_budget_reports_max_iterations(monkeypatch) -> None:
    """Running out of pivots is a status, not a wrong answer dressed up as one."""
    monkeypatch.setattr(qp_module, "_MAX_PIVOTS_PER_ROW", 0)
    solution = solve_qp(np.eye(2), np.array([-1.0, -1.0]), np.zeros((0, 2)), np.zeros(0), np.eye(2), np.zeros(2))
    assert solution.status == "MaxIterations"
