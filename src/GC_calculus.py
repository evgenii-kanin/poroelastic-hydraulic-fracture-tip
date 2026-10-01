import numpy as np
from scipy.integrate import quad_vec

class GCQuadrature:
    def __init__(self, params):
        self.n = params['n']
        self.m = params['n'] - 1
        self.p = params['p']

        self.z = self.z_vector()
        self.s = self.s_vector()

        self.P_s = self.P_s_vector()
        self.Q_z = self.Q_z_vector()
        self.Q_s = self.Q_s_vector()

        self.B = self.B_matrix()
        self.D = self.D_matrix()

        self.S_linear_mapping = self.S_linear_mapping_matrix()

        if self.p == 1:
            self.S = self.S_linear_mapping * self.weight_star_func(self.s) / self.weight_func(self.s)
        else:
            self.S = self.S_matrix()

        if self.p == 1:
            self.S_gamma = self.S_linear_mapping * self.weight_star_gamma_func(self.s) / self.weight_func(self.s)
        else:
            self.S_gamma = self.S_gamma_matrix()

    def mapping_func(self, x):
        return ((1 + x) / (1 - x)) ** self.p

    def mapping_derivative_func(self, x):
        return 2 * self.p * self.mapping_func(x) / (1 - x ** 2)

    def weight_star_func(self, x):
        return (1 + x) ** (self.p / 2 - 1) / (1 - x) ** (2 * self.p / 3 + 1)

    def weight_star_gamma_func(self, x):
        return (1 + x) ** (self.p / 2 - 1) / (1 - x) ** (self.p / 2 + 1)

    def weight_func(self, x):
        return 1 / (1 - x ** 2) ** 0.5

    def s_vector(self):
        return np.cos(np.pi / self.n * (np.arange(1, self.n + 1) - 0.5))

    def z_vector(self):
        return np.cos(np.pi / self.n * np.arange(1, self.n))

    def P_s_vector(self):
        return (-1) ** np.arange(1, self.n + 1) * np.tan(np.arccos(self.s) / 2) / self.n

    def Q_s_vector(self):
        return (-1) ** (np.arange(1, self.n + 1) + 1) * np.tan(np.arccos(self.s) / 2) ** -1 / self.n

    def Q_z_vector(self):
        return (-1) ** (np.arange(1, self.n) + 1) * (1 + self.z)

    # s-nodes, 1st kind
    def B_matrix(self):
        # shape: n * n
        result = np.zeros((self.n, self.n))
        result[0, :] = 1 / self.n

        for i in range(1, self.n):
            result[i, :] = 2 / self.n * np.cos(i * np.arccos(self.s))

        return result

    def D_matrix(self):
        # shape: m * m
        result = np.zeros((self.m, self.m))
        i_prime = np.arange(1, self.n)
        omega = (-1) ** i_prime * np.sin(np.arccos(self.z)) ** 2

        for i in range(self.m):
            musk_cur = i_prime != (i + 1)
            i_prime_cur = i_prime[musk_cur]
            result[i, musk_cur] = omega[i_prime_cur - 1] / omega[i] * 1 / (self.z[i] - self.z[musk_cur])
            result[i, i] = -np.sum(result[i, musk_cur])

        return result

    def S_matrix(self):
        # shape: Phi (m * n) @ B (n * n) -> m * n

        # shape: m * n
        phi_matrix = np.zeros((self.m, self.n))
        poly_vector = lambda x: np.cos(np.arange(self.n) * np.arccos(x)) * self.weight_star_func(x)

        for i, z_i in enumerate(self.z):
            phi_matrix[i, :] = quad_vec(poly_vector, -1, z_i, limit=200)[0]

        return phi_matrix @ self.B

    def S_gamma_matrix(self):
        # shape: Phi (m * n) @ B (n * n) -> m * n

        # shape: m * n
        phi_matrix = np.zeros((self.m, self.n))
        poly_vector = lambda x: np.cos(np.arange(self.n) * np.arccos(x)) * self.weight_star_gamma_func(x)

        for i, z_i in enumerate(self.z):
            phi_matrix[i, :] = quad_vec(poly_vector, -1, z_i, epsabs=1e-10, epsrel=1e-10, limit=200)[0]

        return phi_matrix @ self.B

    def S_linear_mapping_matrix(self):
        # shape: Phi (m * n) @ B (n * n) -> m * n

        # shape: m * n
        phi_matrix = np.zeros((self.m, self.n))

        def Phi_k(k, z):
            result = np.zeros(k.shape[0])
            arccos_z = np.arccos(z)
            result[k == 0] = -arccos_z
            mask = k != 0
            result[mask] = -np.sin(k[mask] * arccos_z) / k[mask]
            return result

        for i, z_i in enumerate(self.z):
            phi_matrix[i, :] = Phi_k(np.arange(self.n), z_i) - Phi_k(np.arange(self.n), -1)

        return phi_matrix @ self.B