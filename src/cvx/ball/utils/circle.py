from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class Circle:
    center: np.ndarray
    radius: float

    def scatter(self, num=100, color="red"):
        t = np.linspace(0, 2 * np.pi, num=num)
        radius = self.radius
        circle_x = self.center[0] + radius * np.cos(t)
        circle_y = self.center[1] + radius * np.sin(t)

        return go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"Circle(r = {self.radius})",
        )
