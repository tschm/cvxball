# [cvxball](/book)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxball)

We compute the smallest enclosing sphere for a set of points.

```bash
# create a numpy array where each row corresponds to a point
# Each row should have the same number of elements and they
# can be higher than 2... :-)
points = np.array([[2.0, 4.0], [0, 0], [2.5, 2.0]])
circle = min_circle_cvx(points, solver="CLARABEL")

# display the points & the circle
fig = create_figure()
fig.add_trace(circle.scatter())
fig.add_trace(Cloud(points).scatter())
fig.show()

```

## Background

We are solving the convex optimization problem:

\[
\min_r \quad r
\]

subject to the constraint that for each point \( p_i \), the Euclidean
distance from \( p_i \) to the center of the circle is less than or
equal to the radius \( r \):

\[
\| p_i - \text{center} \| \leq r, \quad \forall i = 1, 2, \dots, n
\]

Where:

- \( p_i \) are the points in \( \mathbb{R}^d \).
- \( \text{center} \) is the center of the circle we are trying to find.
- \( r \) is the radius of the circle.

The goal is to minimize the radius \( r \) such that all points
lie inside or on the boundary of the circle.

---

## Interpretation as a Min/Max Problem

The constraint \( \| p_i - \text{center} \| \leq r \) implies that
the radius \( r \) must be at least as large as the maximum distance
from the center to any of the points \( p_i \).

If we define the distance from the center to each point as:

\[
d_i = \| p_i - \text{center} \|
\]

Then, the radius \( r \) must satisfy:

\[
r \geq \max_i d_i
\]

Thus, the optimization problem becomes:

\[
r = \min \max_i d_i
\]

This is a **min-max** problem, where we want to
minimize the maximum distance from the center to any of the points.
In other words, we are looking for the smallest possible radius \( r \)
such that the maximum distance from the center to any point is minimized.

---

## Geometric Interpretation

Geometrically, the problem is about finding the **smallest enclosing circle**
(or ball in higher dimensions) that contains all the given points.
The center of the circle is positioned in such a way that the radius
is minimized, but all points still lie inside or on the boundary of the circle.

- **Convexity**: The objective function \( r = \min \max_i d_i \) is
convex, as the maximum of a set of convex functions is convex.
Minimizing a convex function over a convex set is a convex optimization problem.

- **Center and Radius**: The solution involves determining
both the center and the radius of the circle.
The optimal center minimizes the maximum distance to any of the points,
and the optimal radius ensures all points are inside
or on the boundary of the circle.

---

## Relation to Welzl's Algorithm

In 2D, this problem is often solved using **Welzl's algorithm**.
Welzl's algorithm is a recursive method for finding the
smallest enclosing circle. It works by:

1. Randomly choosing a subset of points.
2. Recursively refining the circle so that all points are inside or on its boundary.
3. Ensuring the radius is minimized.

---

## Conclusion

The problem you're describing is a **min-max** optimization problem
aimed at finding the smallest enclosing circle (or ball).
The radius \( r \) is the smallest value that ensures all points
lie within or on the boundary of the circle, and the goal
is to minimize the maximum distance from the center to any
point in the set.
