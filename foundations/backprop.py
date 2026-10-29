import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def sigmoid(self, z:NDArray[np.float64]) -> float:
        return 1.0 / (1.0 + np.exp(-z))


    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        
        z = (x @ w) + b
        y_hat = self.sigmoid(z)
        
        error = y_hat - y_true
        sig_der = y_hat * (1 - y_hat)
        delta = error * sig_der
        dl_dw = np.round(delta * x, 5)
        dl_db = round(float(delta), 5)

        
        return (dl_dw, dl_db)
        
