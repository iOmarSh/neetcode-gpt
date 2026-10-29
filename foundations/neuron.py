import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z:NDArray[np.float64]) -> float:
        return 1 / (1 + np.exp(-z))

    def relu(self, z:NDArray[np.float64]) -> float:
        return max(0, z) / 1

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = (w @ x) + b
        if activation == 'sigmoid':
            y = self.sigmoid(z)
        else:
            y = self.relu(z)

        return round(y, 5)
