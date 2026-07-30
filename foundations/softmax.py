import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        mx = max(z)
        soft = np.exp(z - mx) / sum(np.exp(z - mx))
        return np.round(soft, 4)
        
