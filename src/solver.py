import numpy as np
from scipy.special import hyp2f1
from scipy.optimize import root

from src.collocation_utils import (compute_power_law_interpolation_coefficients,
                                   build_integral_operator_matrix)
from src.precomputed_integrals import load_integrals

class Solver:
    def __init__(self, params, gc):
        self.gc = gc
        self.n = gc.n
        self.x_z = self.gc.mapping_func(self.gc.z)
        self.x_s = self.gc.mapping_func(self.gc.s)

        self.coef_K = (32 / np.pi) ** 0.5

        self.K = self.coef_K * 1 / params['Sigma_0_p'] * (params['S'] / params['rho']) ** 0.5
        self.M = 1 / params['Sigma_0_p'] ** 3 * params['S'] / params['rho']
        self.S = params['S']

        self.beta = params['beta']
        self.eta = params['eta']

        self.flag_one_way = params['flag_one_way']

        self.method = params['method']

        self.p_init_param = params['p_init_param']

        self.aux_vars = self.auxiliary_variables()

    def initial_guess(self):
        def smooth_transition(x, a, b, x0=1e-1, decades=10):
            return a + (b - a) * 0.5 * (1 + np.tanh((np.log10(x / x0)) / (decades / 10)))

        p_init = -6 ** (-2 / 3) * self.M ** (1 / 3) * (1 - self.beta) ** (-2 / 3) / (self.p_init_param + self.x_z ** (1 / 3))

        F_w_near = 3 * 2 ** 0.5 * self.K
        F_w_far = 4 * 6 ** (5 / 6) * self.M ** (1 / 3) * (1 - self.beta) ** (1 / 3)
        F_w_init = smooth_transition(self.x_s, F_w_near, F_w_far, 1e-2, 35)

        F_v_near = -3 * self.K
        F_v_far = 12 * self.S * (2 / np.pi) ** 0.5
        F_v_init = smooth_transition(self.x_s, F_v_near, F_v_far, 1e-5, 5)

        output = np.concatenate((F_w_init, p_init, F_v_init))

        return output

    def auxiliary_variables(self):
        array_x_p, array_x_c = self.x_s[::-1], self.x_z[::-1]

        a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node = (
            compute_power_law_interpolation_coefficients(array_x_p, -1/2, -1/3))

        # Load precomputed integral matrices used by the collocation method.
        integrals = load_integrals()

        def integral_xi_n_inf(x, xi_n, alpha):
            return -(xi_n ** (-alpha) / alpha) * hyp2f1(alpha, 1, alpha + 1, x / xi_n)

        integral_last_segment = (2 ** (4 / 3) * 3 ** (-1 / 6) * self.M ** (1 / 3) * (1 - self.beta) ** (1 / 3) *
                                 integral_xi_n_inf(array_x_c, array_x_p[-1], 1 / 3))

        S_e_matrix_0 = integrals["s_e_matrix_0"]
        S_e_matrix_inf = integrals["s_e_matrix_inf"]
        S_e_near_field_integral = integrals["s_e_near_field_integral"]

        cauchy_matrix_0 = integrals["cauchy_matrix_0"]
        cauchy_matrix_inf = integrals["cauchy_matrix_inf"]
        cauchy_near_field_integral = integrals["cauchy_near_field_integral"]

        # The S_e kernels were precomputed for beta = 0.25. Since the beta-dependent
        # part enters linearly after separating the Cauchy contribution, the matrices
        # are reconstructed here for the requested beta value.
        beta_default = 0.25
        S_e_matrix_0_r = (S_e_matrix_0 - cauchy_matrix_0) / beta_default
        S_e_matrix_inf_r = (S_e_matrix_inf - cauchy_matrix_inf) / beta_default
        S_e_near_field_r = (S_e_near_field_integral - cauchy_near_field_integral) / beta_default

        S_e_matrix_0 = cauchy_matrix_0 + self.beta * S_e_matrix_0_r
        S_e_matrix_inf = cauchy_matrix_inf + self.beta * S_e_matrix_inf_r
        S_e_near_field_integral = cauchy_near_field_integral + self.beta * S_e_near_field_r

        S_e = build_integral_operator_matrix(
            [a_coef_left_node, a_coef_right_node],
            [b_coef_left_node, b_coef_right_node],
            [S_e_matrix_0, S_e_matrix_inf]
        )

        S_e_0 = self.K / 2 * S_e_near_field_integral + integral_last_segment

        output = {}
        output['S_e_0'] = S_e_0
        output['S_e'] = S_e

        P_e_matrix_0 = integrals["p_e_matrix_0"]
        P_e_matrix_inf = integrals["p_e_matrix_inf"]
        P_e_near_field_integral = integrals["p_e_near_field_integral"]

        P_e = build_integral_operator_matrix(
            [a_coef_left_node, a_coef_right_node],
            [b_coef_left_node, b_coef_right_node],
            [P_e_matrix_0, P_e_matrix_inf]
        )

        P_e_0 = self.K / 2 * P_e_near_field_integral - integral_last_segment

        output['P_e_0'] = P_e_0
        output['P_e'] = P_e

        P_s_matrix_0 = integrals["p_s_matrix_0"]
        P_s_matrix_inf = integrals["p_s_matrix_inf"]
        P_s_near_field_integral = integrals["p_s_near_field_integral"]

        P_s = build_integral_operator_matrix(
            [a_coef_left_node, a_coef_right_node],
            [b_coef_left_node, b_coef_right_node],
            [P_s_matrix_0, P_s_matrix_inf]
        )

        P_s_0  = P_s_near_field_integral * (-self.K / 2 - 2 * self.S * (2 / np.pi) ** 0.5)

        output['P_s_0'] = P_s_0
        output['P_s'] = P_s

        # The same discrete coupling matrix is used for S_s and P_e;
        # separate names are kept to mirror the block-operator notation.
        S_s = np.copy(P_e)
        S_s_0 = P_e_near_field_integral * (-self.K / 2 - 2 * self.S * (2 / np.pi) ** 0.5)

        output['S_s_0'] = S_s_0
        output['S_s'] = S_s

        return output

    def residual_func(self, x, integrals=False):
        f = np.zeros(3 * self.n - 1)

        x_F_W, x_P, x_F_Y = x[:self.n], x[self.n:2*self.n-1], x[2*self.n-1:]
        x_F_Y_r = x_F_Y - 12 * self.S * (2 / np.pi) ** 0.5

        dw_dx = (x_F_W * self.gc.weight_star_func(self.gc.s) / self.gc.mapping_derivative_func(self.gc.s))[::-1]
        dv_dx_r = (x_F_Y_r * self.gc.weight_star_gamma_func(self.gc.s) / self.gc.mapping_derivative_func(self.gc.s))[::-1]

        sigma_M = 1 / (4 * np.pi * (1 - self.beta)) * (self.aux_vars['S_e_0'] + self.aux_vars['S_e'] @ dw_dx)
        p_MH = -self.eta / (2 * np.pi * self.S) * (self.aux_vars['P_e_0'] + self.aux_vars['P_e'] @ dw_dx)

        p_H = 1 / (2 * np.pi * self.S) * (self.aux_vars['P_s_0'] + self.aux_vars['P_s'] @ dv_dx_r)
        sigma_HM = -self.eta / (2 * np.pi * self.S) * (self.aux_vars['S_s_0'] + self.aux_vars['S_s'] @ dv_dx_r)

        opening = self.gc.S @ x_F_W
        pressure_derivative = self.gc.D @ x_P
        fluid_exchange_volume = self.gc.S_gamma @ x_F_Y
        mapping_derivative = self.gc.mapping_derivative_func(self.gc.z)

        f[1:self.n] = x_P - sigma_M[::-1] - sigma_HM[::-1]
        f[self.n:2*self.n-1] = opening ** 2 / mapping_derivative * pressure_derivative / self.M - 1 - fluid_exchange_volume / opening
        f[2*self.n-1:-1] = x_P - p_H[::-1] - self.flag_one_way * p_MH[::-1]

        f[0] = self.gc.P_s @ x_F_W - 3 * 2 ** 0.5 * self.K
        f[-1] = self.gc.Q_z @ x_P
        f[-2] = self.gc.Q_s @ x_F_Y - 12 * self.S * (2 / np.pi) ** 0.5

        if integrals:
            return sigma_M[::-1], sigma_HM[::-1], p_H[::-1], p_MH[::-1]
        else:
            return f

    def solve(self, x0):
        if x0 is None:
            x0 = self.initial_guess()

        sol = root(self.residual_func, x0, method=self.method)

        if not sol.success:
            print(f"Warning: Solver did not converge. Message: {sol.message}")
        else:
            print("Solver converged successfully")

        print(f"Residual norm: {np.linalg.norm(sol.fun):.2e}")

        F_W = sol.x[:self.n]
        opening, pressure = self.gc.S @ F_W, sol.x[self.n:2*self.n-1]

        x_coef = self.M ** 2 / self.K ** 6 * self.coef_K ** 6
        w_coef = self.M / self.K ** 4 * self.coef_K ** 4
        p_coef = self.K ** 2 / self.M * self.coef_K ** -2

        output = {
            'x_z': self.x_z * x_coef,
            'x_s': self.x_s * x_coef,
            'w': opening * w_coef,
            'p': pressure * p_coef,
            'F_W': F_W,
            'sol': sol.x
        }

        F_Y = sol.x[2 * self.n - 1:]

        fluid_exchange_volume = self.gc.S_gamma @ F_Y
        fluid_exchange_rate = self.gc.weight_star_gamma_func(self.gc.s) * F_Y / self.gc.mapping_derivative_func(self.gc.s)

        output.update({
            'v': fluid_exchange_volume * w_coef,
            'gamma': fluid_exchange_rate * p_coef,
            'F_Y': F_Y,
            'sol': sol.x
        })

        integrals = self.residual_func(sol.x, True)

        output['integrals'] = {
            'sigma_M': integrals[0] * p_coef, 'sigma_HM': integrals[1] * p_coef,
            'p_H': integrals[2] * p_coef, 'p_MH': integrals[3] * p_coef
        }

        return output