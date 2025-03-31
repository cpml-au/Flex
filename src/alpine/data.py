import numpy.typing as npt
from jax import Array


class Dataset:
    def __init__(
        self, name: str, X: Array | npt.NDArray, y: Array | npt.NDArray | None = None
    ) -> None:
        self.name = name
        self.X = X
        self.y = y
