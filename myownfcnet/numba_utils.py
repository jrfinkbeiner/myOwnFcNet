from typing import Optional
import numpy as np
import numba

@numba.jit(nopython=True)
def numba_matmul(x: np.ndarray, y: np.ndarray, result: Optional[np.ndarray] = None): # TODO check if result tensor really is needed as input
    x1, x2 = x.shape
    y1, y2 = y.shape
    assert x2 == y1
    if result is None:
        result = np.zeros((x1, y2))
    else:
        assert (x1, y2) == result.shape
    for i in range(x1):
        for j in range(y2):
            for k in range(x2):
                result[i,j] += x[i,k] * y[k,j]
    return result

@numba.jit(nopython=True)
def mean_list_numba(x: np.ndarray):
    sum_val = 0
    N = len(x)
    for i in range(N):
        sum_val += x[i]
    return sum_val / N