"""Asymptotic opening, net pressure, and fluid displacement in the mk-scaling."""

import numpy as np

def near_field_opening(x):
    return (32 / np.pi) ** 0.5 * x ** 0.5

def near_field_fluid_displacement(x):
    return -(32 / np.pi) ** 0.5 * x ** 0.5

def far_field_opening(x, beta):
    return 2 ** (1 / 3) * 3 ** (5 / 6) * (1 - beta) ** (1 / 3) * x ** (2 / 3)

def far_field_pressure(x, beta):
    return -6 ** (-2 / 3) * (1 - beta) ** (-2 / 3) * x ** (-1 / 3)

def far_field_fluid_displacement(x, rho, Sigma_0_p, S):
    return 4 * (2 / np.pi) ** 0.5 * Sigma_0_p * (rho * S) ** 0.5 * x ** 0.5
