

from abc import abstractmethod
from typing import Tuple

import numpy as np


class BaseAttractor:
    def __init__(
        self,
        num_points: int,
        init_point: Tuple[float, float, float] = (1e-4, 1e-4, 1e-4),
        step: float = 1.0,
        show_log: bool = False,
        **kwargs: dict,
    ):
        if show_log:
            print(f"[INFO]: Initialize chaotic system: {self.__class__.__name__}\n")
        self.num_points = num_points
        self.init_point = init_point
        self.step = step
        self.kwargs = kwargs

    def get_coordinates(self):
        return np.array(list(next(self)))

    def __len__(self):
        return self.num_points

    def __iter__(self):
        return self

    def __next__(self):
        points = self.init_point
        for i in range(self.num_points):
            try:
                yield points
                next_points = self.attractor(*points, **self.kwargs)
                points = tuple(prev + curr / self.step for prev, curr in zip(points, next_points))
            except OverflowError:
                print(f"[FAIL]: Cannot do the next step because of floating point overflow. Step: {i}")
                break

    @abstractmethod
    def attractor(self, x: float, y: float, z: float, **kwargs) -> Tuple[float, float, float]:
        """Calculate the next coordinate X, Y, Z for chaotic system.
        Do not use this method for parent BaseAttractor class.

        Parameters
        ----------
        x, y, z : float
            Input coordinates: X, Y, Z respectively

        Returns
        -------
        result: tuple
            The next coordinates: X, Y, Z respectively
        """
        # raise NotImplementedError
        return x + y + z, y - z, x ** 2 - z ** 2

    def update_attributes(self, **kwargs):
        """Update chaotic system parameters."""
        for key in kwargs:
            if key in self.__dict__ and not key.startswith("_"):
                self.__dict__[key] = kwargs.get(key)


if __name__ == "__main__":
    base_model = BaseAttractor(num_points=10, init_point=(-0.01, 0.5, 2), step=100)
    print(f"Model attributes: {base_model.__dict__}")
    print(f"Model length: {len(base_model)}")
    xyz = base_model.get_coordinates()
    print(xyz)
