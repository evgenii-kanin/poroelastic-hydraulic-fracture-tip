import numpy as np

def compute_power_law_interpolation_coefficients(array_x, power_0, power_inf):
    """
    Compute interpolation coefficients for a piecewise representation based
    on two power-law asymptotic functions.

    On each interval [x_i, x_{i+1}], the unknown function is represented as

        f(x) = A_i f_0(x) + B_i f_inf(x),

    where A_i and B_i are reconstructed from the nodal values f_i and f_{i+1}.

    Parameters
    ----------
    array_x : ndarray
        Interpolation nodes.
    power_0 : float
        Power-law exponent for the near-tip basis function f_0.
    power_inf : float
        Power-law exponent for the far-field basis function f_inf.

    Returns
    -------
    a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node : ndarray
        Coefficients such that

            A_i = a_coef_left_node[i] * f_i + a_coef_right_node[i] * f_{i+1},

            B_i = b_coef_left_node[i] * f_i + b_coef_right_node[i] * f_{i+1}.
    """

    omega_0_nodes = array_x ** power_0
    omega_inf_nodes = array_x ** power_inf

    denom = omega_inf_nodes[:-1] * omega_0_nodes[1:] - omega_inf_nodes[1:] * omega_0_nodes[:-1]

    a_coef_left_node = -omega_inf_nodes[1:] / denom
    a_coef_right_node = omega_inf_nodes[:-1] / denom

    b_coef_left_node = omega_0_nodes[1:] / denom
    b_coef_right_node = -omega_0_nodes[:-1] / denom

    return a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node

def build_integral_operator_matrix(a_coef, b_coef, kernel_matrices):
    """
    Assemble the discrete integral-operator matrix from precomputed
    interval kernel integrals and power-law interpolation coefficients.

    Parameters
    ----------
    a_coef, b_coef : list of ndarray
        Interpolation coefficients for the two basis-function amplitudes.
        The first array contains coefficients multiplying the left nodal
        value f_i, and the second array contains coefficients multiplying
        the right nodal value f_{i+1}.
    kernel_matrices : list of ndarray
        Precomputed integral matrices corresponding to the two basis functions.

    Returns
    -------
    ndarray
        Collocation matrix mapping nodal unknowns to integral-operator values
        at the collocation points.
    """

    matrix = np.zeros((kernel_matrices[0].shape[0], kernel_matrices[0].shape[1] + 1))

    coef_left_node = a_coef[0] * kernel_matrices[0].T + b_coef[0] * kernel_matrices[1].T
    coef_right_node = a_coef[1] * kernel_matrices[0].T + b_coef[1] * kernel_matrices[1].T

    matrix[:, :-1] += coef_left_node
    matrix[:, 1:] += coef_right_node

    return matrix