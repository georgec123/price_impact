# distutils: cython_compile_time_env=2
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def ewm_alpha(np.ndarray[np.float64_t, ndim=2] arr, np.float64_t alpha):
    cdef np.ndarray[double, ndim=2] result = np.copy(arr)
    cdef int i, j
    cdef np.npy_intp nrows = arr.shape[0]
    cdef np.npy_intp ncols = arr.shape[1]
    
    for i in range(nrows):
        for j in range(1, ncols):
            result[i, j] += alpha*result[i, j - 1]

    return result
