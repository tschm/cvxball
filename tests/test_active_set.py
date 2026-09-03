"""Tests for the Clarabel-shaped front end in :mod:`cvxball.active_set`.

The claim this module makes is *substitutability*: code written against
``clarabel.DefaultSolver`` should run against ``active_set.DefaultSolver``
unchanged.  So the central tests here parametrise over the two modules and run
the same lines against both -- if the two ever disagree about the shape of a
program, a settings object, a cone, or a status, one of these fails.
"""

import doctest

import clarabel
import numpy as np
import pytest
import scipy.sparse as sp

import cvxball.active_set as front_end
import cvxball.qp as qp_module
import cvxball.solver as solver_module
from cvxball import active_set
from cvxball.solver import _build_soc_program, min_circle_active_set

# The two interchangeable back ends. Every test parametrised over this runs the
# identical call sequence against both.
BACKENDS = [clarabel, active_set]
BACKEND_IDS = ["clarabel", "active_set"]

# A point cloud whose smallest enclosing circle is the one on the hypotenuse.
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _portfolio_program(n=6, cap=0.4):
    """Build a long-only, capped minimum-variance program in Clarabel's form.

    Returns the ``(P, q, A, b, cone_dims)`` of the program, with the cone
    dimensions given as counts rather than cone objects so that each backend can
    construct its own.
    """
    rng = np.random.default_rng(5)
    factors = rng.normal(size=(n, 3))
    covariance = factors @ factors.T / 3 + np.diag(rng.random(n) * 0.1 + 0.02)
    q = -np.arange(n, dtype=float) / n

    a_mat = np.vstack([np.ones((1, n)), np.eye(n), -np.eye(n)])
    b = np.concatenate([[1.0], np.full(n, cap), np.zeros(n)])
    return sp.csc_matrix(np.triu(covariance)), q, sp.csc_matrix(a_mat), b, (1, 2 * n)


def _solve_with(backend, p_mat, q, a_mat, b, cones):
    """Run one program through one backend, with logging off."""
    settings = backend.DefaultSettings.default()
    settings.verbose = False
    return backend.DefaultSolver(p_mat, q, a_mat, b, cones, settings).solve()


# --- Substitutability ----------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_the_enclosing_ball_program_solves_the_same_either_way(backend) -> None:
    """The cone program of `min_circle_clarabel` runs unchanged on both backends."""
    p_mat, q, a_mat, b, cones = _build_soc_program(TRIANGLE)
    solution = _solve_with(backend, p_mat, q, a_mat, b, cones)

    assert solution.status == backend.SolverStatus.Solved
    assert solution.obj_val == pytest.approx(2**0.5 / 2, abs=1e-6)
    assert solution.x[1:] == pytest.approx([0.5, 0.5], abs=1e-4)
    # The dual objective agrees with the primal, which is the duality gap closing.
    assert solution.obj_val_dual == pytest.approx(solution.obj_val, abs=1e-6)
    assert solution.solve_time >= 0.0
    assert solution.iterations >= 0
    assert len(solution.s) == len(b)
    assert len(solution.z) == len(b)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_the_portfolio_program_solves_the_same_either_way(backend) -> None:
    """A capped long-only minimum-variance QP agrees between the two backends."""
    p_mat, q, a_mat, b, (n_eq, n_in) = _portfolio_program()
    cones = [backend.ZeroConeT(n_eq), backend.NonnegativeConeT(n_in)]
    solution = _solve_with(backend, p_mat, q, a_mat, b, cones)

    assert solution.status == backend.SolverStatus.Solved
    assert np.sum(solution.x) == pytest.approx(1.0, abs=1e-7)
    assert np.min(solution.x) >= -1e-7
    assert np.max(solution.x) <= 0.4 + 1e-7
    assert solution.obj_val == pytest.approx(-0.5202892312, abs=1e-6)
    assert solution.obj_val_dual == pytest.approx(solution.obj_val, abs=1e-6)


def test_the_active_set_answer_is_the_sharper_one() -> None:
    """Where the two backends differ on the ball program, this one is exact.

    Clarabel stops when its residuals are small; the support-set method stops at
    the vertex.  On a cloud whose answer is exactly representable the difference
    is visible in the residuals the two report -- and it is the whole reason for
    this front end to exist, so it is asserted rather than left implied.
    """
    program = _build_soc_program(TRIANGLE)
    exact = _solve_with(active_set, *program)
    approximate = _solve_with(clarabel, *program)

    assert exact.obj_val == 2**0.5 / 2
    assert exact.r_prim == 0.0
    assert exact.r_dual == 0.0
    assert approximate.r_prim > 0.0
    assert abs(approximate.obj_val - 2**0.5 / 2) > abs(exact.obj_val - 2**0.5 / 2)


def test_solution_attributes_match_clarabels() -> None:
    """The solution object carries every attribute Clarabel's does.

    Substitutability is about attribute names as much as numbers: a caller reading
    `solution.r_prim` must not get an AttributeError from the replacement.
    """
    program = _build_soc_program(TRIANGLE)
    reference = _solve_with(clarabel, *program)
    mine = _solve_with(active_set, *program)

    expected = {name for name in dir(reference) if not name.startswith("_")}
    assert expected <= {name for name in dir(mine) if not name.startswith("_")}


# --- The dual solution ---------------------------------------------------------


def test_the_ball_dual_is_the_support_weights() -> None:
    """The conic dual of the ball program is the support set, arranged for cones.

    Each point's block is ``(u_i, -u_i (p_i - x) / R)``: zero off the support, and
    on the boundary of the second-order cone on it.  Both halves are checked here,
    because that is what makes the returned ``z`` an actual certificate and not
    just a vector of the right length.
    """
    points = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [2.0, 1.0]])
    solution = _solve_with(active_set, *_build_soc_program(points))
    radius, centre = min_circle_active_set(points)

    blocks = solution.z.reshape(len(points), 3)
    weights = blocks[:, 0]
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert weights.min() >= 0.0
    # The centre is the weighted average of the points: the KKT certificate.
    assert weights @ points == pytest.approx(centre, abs=1e-12)
    # On the support, z sits exactly on the cone boundary; off it, z is zero.
    support = weights > 0.0
    assert np.linalg.norm(blocks[support, 1:], axis=1) == pytest.approx(weights[support], abs=1e-12)
    assert not blocks[~support].any()
    assert solution.x[0] == pytest.approx(radius, abs=1e-12)


def test_the_qp_dual_respects_its_cones() -> None:
    """Inequality multipliers are non-negative and complementary; equalities are free."""
    p_mat, q, a_mat, b, (n_eq, n_in) = _portfolio_program()
    cones = [active_set.ZeroConeT(n_eq), active_set.NonnegativeConeT(n_in)]
    solution = _solve_with(active_set, p_mat, q, a_mat, b, cones)

    assert solution.z[n_eq:].min() >= 0.0
    assert np.abs(solution.z[n_eq:] * solution.s[n_eq:]).max() <= 1e-12
    assert abs(solution.s[0]) <= 1e-14
    assert solution.r_prim <= 1e-14
    assert solution.r_dual <= 1e-12


# --- Programs this solver declines ---------------------------------------------


def test_an_unsupported_cone_is_refused_at_construction() -> None:
    """A cone family this method has no method for is refused, and says which."""
    with pytest.raises(ValueError, match="unsupported cone combination"):
        active_set.DefaultSolver(
            None,
            np.zeros(3),
            sp.csc_matrix(np.eye(3)),
            np.zeros(3),
            [clarabel.PSDTriangleConeT(3)],
        )


def test_mixing_cone_families_is_refused() -> None:
    """A program that is part QP and part second-order cone is not either family."""
    with pytest.raises(ValueError, match="unsupported cone combination"):
        active_set.DefaultSolver(
            None,
            np.zeros(3),
            sp.csc_matrix(np.eye(4, 3)),
            np.zeros(4),
            [active_set.NonnegativeConeT(1), active_set.SecondOrderConeT(3)],
        )


def test_unequal_second_order_cones_are_refused() -> None:
    """Second-order cones of different sizes cannot be one point cloud."""
    with pytest.raises(ValueError, match="unsupported cone combination"):
        active_set.DefaultSolver(
            None,
            np.zeros(3),
            sp.csc_matrix(np.eye(5, 3)),
            np.zeros(5),
            [active_set.SecondOrderConeT(2), active_set.SecondOrderConeT(3)],
        )


def test_a_general_second_order_cone_program_is_refused() -> None:
    """An SOCP that is not the enclosing-ball program is refused, not approximated.

    The cone structure alone does not identify the problem: the front end rebuilds
    the ball program from the right-hand side and compares, so a program with the
    same cones but a different matrix is caught rather than answered.
    """
    p_mat, q, a_mat, b, cones = _build_soc_program(TRIANGLE)
    tampered = a_mat.toarray()
    tampered[1, 1] = 2.0
    with pytest.raises(ValueError, match="not the smallest-enclosing-ball program"):
        active_set.DefaultSolver(p_mat, q, sp.csc_matrix(tampered), b, cones)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cones": []}, "cones is empty"),
        ({"cones": [object()]}, "no positive integer"),
        ({"cones": [clarabel.NonnegativeConeT(2)]}, "cone dimensions sum to 2"),
        ({"a_mat": np.eye(2)}, r"A has shape \(2, 2\)"),
        ({"p_mat": np.eye(2)}, r"P has shape \(2, 2\)"),
    ],
    ids=["no-cones", "cone-without-dim", "wrong-cone-total", "wrong-a-shape", "wrong-p-shape"],
)
def test_inconsistent_arguments_are_refused(kwargs, match) -> None:
    """Shape and cone bookkeeping is checked up front, with the mismatch named."""
    program = {
        "p_mat": None,
        "q": np.zeros(3),
        "a_mat": sp.csc_matrix(np.eye(4, 3)),
        "b": np.zeros(4),
        "cones": [clarabel.NonnegativeConeT(4)],
    }
    with pytest.raises(ValueError, match=match):
        active_set.DefaultSolver(**(program | kwargs))


def test_a_degenerate_point_cloud_is_refused_with_its_own_message() -> None:
    """A ball program whose points are not finite fails as an input problem."""
    p_mat, q, a_mat, b, cones = _build_soc_program(TRIANGLE)
    broken = b.copy()
    broken[1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        active_set.DefaultSolver(p_mat, q, a_mat, broken, cones)


# --- Statuses other than Solved ------------------------------------------------


def test_an_infeasible_qp_reports_primal_infeasible() -> None:
    """``x >= 1`` and ``x <= -1`` comes back as a status, in Clarabel's vocabulary."""
    solution = _solve_with(
        active_set,
        sp.csc_matrix(np.eye(1)),
        np.zeros(1),
        sp.csc_matrix(np.array([[1.0], [-1.0]])),
        np.array([-1.0, -1.0]),
        [active_set.NonnegativeConeT(2)],
    )
    assert solution.status == active_set.SolverStatus.PrimalInfeasible


def test_a_ball_run_that_exhausts_its_budget_reports_max_iterations(monkeypatch) -> None:
    """A support-set run that cannot finish reports a status, not a plausible ball.

    The public `min_circle_active_set` raises here, because a caller asking for a
    radius has no use for a half-finished one. Behind Clarabel's interface the
    same event has to be a status instead, and the iterate is deliberately `nan`:
    an active-set iterate that never reached optimality is not an answer.
    """
    monkeypatch.setattr(solver_module, "_MAX_ITER_PER_POINT", 0)
    solution = _solve_with(active_set, *_build_soc_program(TRIANGLE))

    assert solution.status == active_set.SolverStatus.MaxIterations
    assert np.isnan(solution.x).all()
    assert np.isnan(solution.z).all()


def test_a_stalled_ball_run_reports_a_numerical_error(monkeypatch) -> None:
    """The other support-set failure maps to `NumericalError` rather than to a raise."""
    monkeypatch.setattr(solver_module, "_affine_null_space", lambda face: np.zeros((face.shape[0], 1)))
    solution = _solve_with(active_set, *_build_soc_program(TRIANGLE))
    assert solution.status == active_set.SolverStatus.NumericalError


def test_a_qp_that_runs_out_of_pivots_reports_max_iterations(monkeypatch) -> None:
    """The QP path's budget maps to `MaxIterations` too."""
    monkeypatch.setattr(qp_module, "_MAX_PIVOTS_PER_ROW", 0)
    p_mat, q, a_mat, b, (n_eq, n_in) = _portfolio_program()
    solution = _solve_with(
        active_set, p_mat, q, a_mat, b, [active_set.ZeroConeT(n_eq), active_set.NonnegativeConeT(n_in)]
    )
    assert solution.status == active_set.SolverStatus.MaxIterations


# --- Settings ------------------------------------------------------------------


def test_settings_are_optional() -> None:
    """Omitting the settings argument takes the defaults, as Clarabel's does."""
    solution = active_set.DefaultSolver(*_build_soc_program(TRIANGLE)).solve()
    assert solution.status == active_set.SolverStatus.Solved


def test_verbose_settings_reach_both_methods(capsys) -> None:
    """`settings.verbose` turns on the pivot log for either family of program."""
    settings = active_set.DefaultSettings.default()
    settings.verbose = True

    p_mat, q, a_mat, b, cones = _build_soc_program(TRIANGLE)
    active_set.DefaultSolver(p_mat, q, a_mat, b, cones, settings).solve()
    assert "support=" in capsys.readouterr().out

    p_mat, q, a_mat, b, (n_eq, n_in) = _portfolio_program()
    cones = [active_set.ZeroConeT(n_eq), active_set.NonnegativeConeT(n_in)]
    active_set.DefaultSolver(p_mat, q, a_mat, b, cones, settings).solve()
    assert "active=" in capsys.readouterr().out


def test_a_singular_objective_needs_clarabels_regularization_setting() -> None:
    """The static-regularization settings are read, and they decide this program.

    ``P`` here is singular, so the method has no starting point unless it is
    allowed to shift the diagonal. Clarabel has settings for exactly that, and
    honouring them is what lets the same settings object drive both solvers.
    """
    p_mat = sp.csc_matrix(np.diag([1.0, 0.0]))
    program = (p_mat, np.array([0.0, -1.0]), sp.csc_matrix(np.eye(2)), np.array([1.0, 1.0]))
    cones = [active_set.NonnegativeConeT(2)]

    settings = active_set.DefaultSettings.default()
    settings.verbose = False
    settings.static_regularization_enable = False
    with pytest.raises(ValueError, match="semidefinite but not definite"):
        active_set.DefaultSolver(*program, cones, settings).solve()

    settings.static_regularization_enable = True
    solution = active_set.DefaultSolver(*program, cones, settings).solve()
    assert solution.status == active_set.SolverStatus.Solved
    assert solution.x == pytest.approx([0.0, 1.0], abs=1e-6)


def test_a_settings_object_may_be_any_duck() -> None:
    """Any object with the right attributes works, so callers need not import clarabel."""

    class Settings:
        """The three fields this front end reads, and nothing else."""

        verbose = False
        tol_feas = 1e-10
        static_regularization_enable = False

    solution = active_set.DefaultSolver(*_build_soc_program(TRIANGLE), Settings()).solve()
    assert solution.status == active_set.SolverStatus.Solved


# --- The documented examples ---------------------------------------------------


@pytest.mark.parametrize("module", [front_end, qp_module, solver_module], ids=["active_set", "qp", "solver"])
def test_docstring_examples_are_true(module) -> None:
    """Every `>>>` example in the source actually produces what it claims.

    The examples pin exact values on purpose, so they are only worth writing if
    something runs them.
    """
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
