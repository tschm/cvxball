# 🔵 [cvxball](https://www.cvxgrp.org/cvxball)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

We compute the smallest enclosing circle/ball for a set of points.

![Smallest enclosing circle for $50$ random points](example.png)

```python
import numpy as np
from cvxball import min_circle_clarabel

# create a numpy array where each row corresponds to a point
points = np.array([[2.0, 4.0], [0, 0], [2.5, 2.0]])

# compute the smallest enclosing circle
radius, centre = min_circle_clarabel(points)
```

### 🔧 The solvers

Two solvers share one interface — both take `(points, verbose=False)` and
return `(radius, center)` — so they are interchangeable:

| Solver | Approach |
|---|---|
| **`min_circle_clarabel`** | Assembles the second-order cone program directly and calls [Clarabel](https://clarabel.org), with no modelling-layer canonicalisation in between. |
| **`min_circle_active_set`** | Runs an active-set QP method on the *dual* of the same problem. Pure NumPy, no solver dependency. |

```python
from cvxball.solver import min_circle_active_set

radius, centre = min_circle_active_set(points)  # same call, same result
```

The active-set method keeps a *support set* of points held on the ball's
boundary and repeatedly re-centres the ball on that set, dropping a support
point whose dual weight would turn negative and adding the farthest point still
left outside. Each iteration is one small dense solve of size at most the
ambient dimension, so its cost is driven by $d$ rather than by $n$.

Two consequences are worth knowing when picking between them. The active-set
method stops at an *exact vertex* of the dual feasible set instead of at an
interior-point tolerance, so on inputs where the answer is determined exactly —
points already lying on a common sphere, say — it returns the radius to machine
precision where an interior-point solver lands within its own tolerance. And
because it touches all $n$ points only to find the next farthest one, it scales
much better in $n$: on $20\,000$ points in $\mathbb{R}^{10}$ it is roughly
three orders of magnitude faster here. The cone program, in exchange, is the
more familiar formulation and the one to reach for if you want to extend the
model with further conic constraints.

## 🧰 A Clarabel-shaped active-set solver

`cvxball.active_set` is a solver front end that can be *substituted* for the
`clarabel` module. Code written against Clarabel's Python API keeps working when
the import changes:

```python
import numpy as np
import scipy.sparse as sp
from cvxball import active_set as backend  # or: import clarabel as backend

# minimum-variance, fully invested, long-only
covariance = sp.csc_matrix(np.array([[0.04, 0.01], [0.01, 0.04]]))
a_mat = sp.csc_matrix(np.vstack([np.ones((1, 2)), -np.eye(2)]))
b = np.array([1.0, 0.0, 0.0])
cones = [backend.ZeroConeT(1), backend.NonnegativeConeT(2)]

settings = backend.DefaultSettings.default()
settings.verbose = False
solver = backend.DefaultSolver(covariance, np.zeros(2), a_mat, b, cones, settings)
solution = solver.solve()  # solution.x, .z, .s, .status, .obj_val, ...
```

The same standard form — minimise $\tfrac12 x'Px + q'x$ subject to
$Ax + s = b$, $s \in \mathcal{K}$ — the same solution attributes, and the same
`clarabel.SolverStatus` values, which are re-exported so that
`solution.status == clarabel.SolverStatus.Solved` holds either way.

### What it handles

| Cones | Method |
|---|---|
| `ZeroConeT` and `NonnegativeConeT` blocks, in any order | `cvxball.qp.solve_qp`, a dense dual active-set method for the resulting equality- and inequality-constrained QP (Goldfarb–Idnani, 1983) |
| Equally sized `SecondOrderConeT` blocks in the enclosing-ball shape | the support-set method above |
| anything else | refused at construction, with a message naming what it saw |

`P` must be positive definite, or made so by Clarabel's own
`static_regularization_*` settings; an *indefinite* `P` is refused rather than
quietly convexified. Of the remaining settings, `verbose` and `tol_feas` are
read and the interior-point controls are ignored on purpose — `max_iter` counts
a different kind of iteration, and `tol_gap_*` describe a duality gap this
method closes exactly rather than approaches.

### Why bother, when Clarabel is right there

Not for speed. Clarabel is compiled, this is NumPy, and a pivot loop in Python
pays ~10–20 µs per pivot; on a capped long-only portfolio of $n$ names it needs
roughly one pivot per binding bound, so it loses by an order of magnitude:

| $n$ | active-set | Clarabel | pivots |
|---:|---:|---:|---:|
| 10 | 0.18 ms | 0.05 ms | 2 |
| 20 | 0.60 ms | 0.12 ms | 12 |
| 50 | 4.7 ms | 0.61 ms | 74 |
| 100 | 24 ms | 2.8 ms | 223 |
| 200 | 167 ms | 8.9 ms | 480 |

What it offers instead is a different *kind* of answer. An interior-point method
approaches the optimal face from inside and leaves you to decide which near-zero
slacks were meant to be zero; an active-set method returns that face. So:

- **The active set is exact.** Which names sit on their cap, and which
  constraints bind, is a discrete answer, returned as one — no thresholding.
- **The solution on that face is a linear solve, not a limit.** The residuals
  come back at `1e-16` rather than at `1e-8`; on the enclosing-ball program the
  radius is exact to the last bit, where Clarabel is out by `~1e-9`.
- **The multipliers are shadow prices of the same quality**, which matters when
  they are what you are actually after.
- **Infeasibility is proved, not inferred** from a residual that stopped
  improving: the method finds a constraint whose multiplier can be raised
  without bound and says which row it was.

The two natural next steps, neither implemented: incremental factorisation
updates (the Goldfarb–Idnani QR updates, turning $O(nk^2 + k^3)$ per pivot into
$O(n^2)$), and **warm starts** — re-solving from a previous active set, which is
where an active-set method structurally beats an interior-point one and which
finance asks for constantly (a frontier sweep, a daily rebalance, a parametric
study). The tables above measure the cost of finding the active set from
scratch; a warm start skips almost all of it.

## 🧮 Background

We are solving the convex optimization problem:

$$
\min_r \quad r
$$

subject to the constraint that for each point $p_i$, the Euclidean
distance from $p_i$ to the center of the circle is less than or
equal to the radius $r$:

$$
\| p_i - \text{center} \| \leq r, \quad \forall i = 1, 2, \dots, n
$$

Where:

- $p_i$ are the points in $\mathbb{R}^d$.
- $\text{center}$ is the center of the circle we are trying to find.
- $r$ is the radius of the circle.

The goal is to minimize the radius $r$ such that all points
lie inside or on the boundary of the circle.

---

### 📊 Interpretation as a Min/Max Problem

The constraint $\| p_i - \text{center} \| \leq r$ implies that
the radius $r$ must be at least as large as the maximum distance
from the center to any of the points $p_i$.

If we define the distance from the center to each point as:

$$
d_i = \| p_i - \text{center} \|
$$

Then, the radius $r$ must satisfy:

$$
r \geq \max_i d_i
$$

Thus, the optimization problem becomes:

$$
r = \min \max_i d_i
$$

This is a **min-max** problem, where we want to
minimize the maximum distance from the center to any of the points.
In other words, we are looking for the smallest possible radius $r$
such that the maximum distance from the center to any point is minimized.

---

### 📐 Geometric Interpretation

Geometrically, the problem is about finding the **smallest enclosing circle**
(or ball in higher dimensions) that contains all the given points.
The center of the circle is positioned in such a way that the radius
is minimized, but all points still lie inside or on the boundary of the circle.

- **Convexity**: The objective function $r = \min \max_i d_i$ is
convex, as the maximum of a set of convex functions is convex.
Minimizing a convex function over a convex set is a convex optimization problem.

- **Center and Radius**: The solution involves determining
both the center and the radius of the circle.
The optimal center minimizes the maximum distance to any of the points,
and the optimal radius ensures all points are inside
or on the boundary of the circle.

---

### ♊ The dual, and what the active-set method exploits

Attaching multipliers $u_i \geq 0$ to the constraints above and eliminating the
centre gives the dual problem

$$
\max_u \quad \sum_i u_i \|p_i\|^2 - \Big\| \sum_i u_i p_i \Big\|^2
\qquad \text{subject to} \quad u \geq 0, \ \sum_i u_i = 1,
$$

a quadratic program over the unit simplex. At the optimum the centre is the
weighted average $\text{center} = \sum_i u_i p_i$ and the objective value is
$r^2$. The complementary slackness conditions read

$$
\|p_i - \text{center}\| = r \ \text{ where } u_i > 0,
\qquad
\|p_i - \text{center}\| \leq r \ \text{ where } u_i = 0,
$$

which is exactly the geometric statement that the ball is pinned by the points
on its boundary and encloses everything else. Those boundary points are the
*support set*, they number at most $d + 1$, and the centre is a convex
combination of them.

That is what makes an active-set method natural here: the whole problem is
determined by which handful of points sit on the boundary. The method searches
over support sets directly, and every candidate it produces carries its own
optimality certificate.
