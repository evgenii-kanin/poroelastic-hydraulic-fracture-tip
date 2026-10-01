"""
Dimensionless kernel functions for steadily moving fluid sources and
edge dislocations in plane poroelasticity.

The suffix ``s`` denotes a steadily moving fluid source, while ``e``
denotes a steadily moving normal edge dislocation. The functions correspond to
the dimensionless kernel expressions presented in Section 2.2.1 of the paper.

The kernels are singular at x = 0 and should be evaluated away from the
singular point.
"""

import numpy as np
from scipy.special import k0e, k1e

def exp_func(x):
    return np.exp(x - np.abs(x))

# Steadily moving fluid source kernels

def p_s(x, regular_part=False):
    x = np.asarray(x, dtype=float)

    result = exp_func(x) * k0e(np.abs(x))

    if regular_part:
        result += np.log(np.abs(x))

        value_at_zero = -np.euler_gamma + np.log(2.0)

        mask = np.abs(x) < 1e-10

        result = np.where(mask, value_at_zero, result)

    return result

def sigma_yy_s(x, cauchy_flag=True):
    x = np.asarray(x, dtype=float)

    result = - 1 - x * exp_func(x) * (k0e(np.abs(x)) - np.sign(x) * k1e(np.abs(x)))

    value_at_zero = 0

    mask = x == 0

    result = np.where(mask, value_at_zero, result)

    if cauchy_flag:
        result /= x

    return result

# Steadily moving normal edge dislocation kernels

def p_e(x, cauchy_flag=True):
    return sigma_yy_s(x, cauchy_flag)

def sigma_yy_e(x, beta, cauchy_flag=True):
    x = np.asarray(x, dtype=float)

    result = 1 - beta * (1 / x - np.sign(x) * exp_func(x) * k1e(np.abs(x)) -
                         2 * x * exp_func(x) * (k0e(np.abs(x)) - np.sign(x) * k1e(np.abs(x))))

    value_at_zero = 1 - beta

    mask = np.abs(x) < 1e-10

    result = np.where(mask, value_at_zero, result)

    if cauchy_flag:
        result /= x

    return result
