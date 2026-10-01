import numpy as np
from scipy.integrate import quad, quad_vec
import mpmath as mp
from tqdm import tqdm

def integral_near_tip_log(xi_m, xi_0, alpha, kernel):
    """
    Compute near-tip integrals for the kernel P_s with logarithmic singularity.
    """
    result = np.array([quad(lambda s: kernel(x - s), 0, xi_0,
                            weight='alg', wvar=(-alpha, 0.0), epsrel=1e-15)[0] for x in xi_m])
    return result

def integral_near_tip_cauchy(xi_m, xi_0, alpha, kernel):
    """
    Compute near-tip integrals for kernels with Cauchy-type singularity (S_e, S_s, P_e).
    """
    result = np.array([quad(lambda s: 1 / (x - s) * kernel(x - s), 0, xi_0,
                            weight='alg', wvar=(-alpha, 0.0), epsrel=1e-12, limit=100)[0] for x in xi_m])
    return result

def int_x_alpha_log(x, a, alpha):
    x, a, alpha = mp.mpf(x), mp.mpf(a), mp.mpf(alpha)

    return (x ** (1 - alpha) * (x * mp.hyper([1, 2 - alpha], [3 - alpha], x / a) -
                                a * (-2 + alpha) * mp.log(a - x))) / (a * (-2 + alpha) * (-1 + alpha))

def integral_intervals_log(xi_m, xi, alpha, kernel):
    """
    Compute interval integrals for the P_s kernel with logarithmic singularity.

    The diagonal entries are evaluated by decomposing the kernel into a regular
    part and a logarithmic singular term. The regular part is integrated
    numerically, while the logarithmic contribution is evaluated analytically.
    """
    num_intervals = xi.shape[0] - 1
    output = np.zeros((num_intervals, xi_m.shape[0]))

    for i in tqdm(range(xi.shape[0] - 1)):
        mask = (xi_m <= xi[i]) | (xi_m >= xi[i + 1])
        x_copy = xi_m[mask]

        vals_cur = np.zeros_like(xi_m)

        vals_cur[mask] = quad_vec(lambda s: s ** (-alpha) * kernel(x_copy - s), xi[i], xi[i + 1], epsrel=1e-14)[0]

        vals_cur[i] = (quad(lambda s: s ** (-alpha) * kernel(xi_m[i] - s, regular_part=True),
                            xi[i], xi[i + 1], epsabs=1e-14, epsrel=1e-14, limit=100)[0] -
                       np.float64(mp.re(int_x_alpha_log(xi[i + 1], xi_m[i], alpha) - int_x_alpha_log(xi[i], xi_m[i], alpha))))

        output[i, :] = vals_cur

    return output

def integral_intervals_cauchy(xi_m, xi, alpha, kernel):
    """
    Compute interval integrals for kernels with Cauchy-type singularity (S_e, S_s, P_e).
    """
    num_intervals = xi.shape[0] - 1
    output = np.zeros((num_intervals, xi_m.shape[0]))

    for i in tqdm(range(xi.shape[0] - 1)):
        mask = (xi_m <= xi[i]) | (xi_m >= xi[i + 1])
        x_copy = xi_m[mask]

        vals_cur = np.zeros_like(xi_m)

        vals_cur[mask] = quad_vec(lambda s: s ** (-alpha) / (x_copy - s) * kernel(x_copy - s),
                                  xi[i], xi[i + 1], epsrel=1e-10, limit=100)[0]

        vals_cur[i] = quad(lambda s: -s ** (-alpha) * kernel(xi_m[i] - s),
                           xi[i], xi[i + 1], weight='cauchy', wvar=xi_m[i], epsabs=1e-10, epsrel=1e-10, limit=500)[0]

        output[i, :] = vals_cur

    return output
