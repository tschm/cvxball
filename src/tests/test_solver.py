import numpy as np

from cvx.ball.solver import min_circle_cvx
from cvx.ball.utils.cloud import Cloud
from cvx.ball.utils.figure import create_figure


def test_random():
    p = np.array([[2.0, 4.0], [0, 0], [2.5, 2.0]])
    cloud = Cloud(p)
    circle = min_circle_cvx(p, solver="CLARABEL")

    fig = create_figure()
    fig.add_trace(circle.scatter())
    fig.add_trace(cloud.scatter())

    # fig.show()
