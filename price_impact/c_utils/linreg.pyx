import numpy as np
cimport numpy as np


ctypedef np.float64_t DTYPE_t

cpdef r_squared(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] y_pred, np.float64_t y_mean):
    
    cdef np.float64_t r2, numsum = 0, densum = 0
    cdef np.npy_intp n = y.shape[0]

    for i in range(n):
        numsum += (y[i] - y_pred[i])**2
        densum += (y[i] - y_mean)**2
    
    r2 = 1 - numsum / densum

    return r2


cpdef predict_and_score(np.ndarray[np.float64_t, ndim=1] x,
                        np.ndarray[np.float64_t, ndim=1] y, 
                        np.float64_t slope):

    cdef np.float64_t y_mean, r2
    cdef np.float64_t y_sum =0
    cdef np.npy_intp n = x.shape[0]

    for i in range(n):
        y_sum += y[i]

    y_mean = y_sum / n

    y_pred = slope * x

    r2 = r_squared(y, y_pred, y_mean)

    return r2

cpdef linear_regression(np.ndarray[np.float64_t, ndim=1] x,
                        np.ndarray[np.float64_t, ndim=1] y, 
                        np.float64_t coef_mean = 0,
                        np.float64_t penalisation_coef = 0):


    cdef np.float64_t x_mean, y_mean
    cdef np.float64_t slope
    cdef np.float64_t x_sum =0, y_sum=0, numerator=0, denominator=0
    cdef np.npy_intp n = x.shape[0]

    for i in range(n):
        y_sum += y[i]
        denominator += x[i]**2
        numerator += x[i] * y[i]

    y_mean = y_sum / n

    slope = (numerator + penalisation_coef*coef_mean) / (denominator+penalisation_coef)


    # predict y_os and y_is
    y_pred = slope * x 

    # calculare r2
    r2 = r_squared(y, y_pred, y_mean)

    return slope, r2

