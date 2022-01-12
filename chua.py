

from math import fabs
from typing import Tuple

from src.attractors.attractor import BaseAttractor


class Chua(BaseAttractor):
    """Chua attractor."""

    def attractor(
        self,
        x: float,
        y: float,
        z: float,
        alpha: float = 5,
        beta: float = 20,
        mu0: float = -8/7,
        mu1: float = -5/7,
    ) -> Tuple[float, float, float]:
        ht = mu1 * x + 0.5 * (mu0 - mu1) * (fabs(x + 1) - fabs(x - 1))
        # Next step coordinates:
        x_out = alpha * (y - x - ht)
        y_out = x - y + z
        z_out = -beta * y
        return x_out, y_out, z_out


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
