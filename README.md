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
