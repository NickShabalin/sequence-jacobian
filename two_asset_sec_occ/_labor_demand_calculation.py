from itertools import product

import numpy as np

import utils


# TODO: add comments and type annotations

# noinspection PyPep8Naming
class LaborDemandCalculation:
    def __init__(self, alpha_sec_occ, sigma_sec, p_sec, Y_sec, nu_sec, w_occ):
        self._alpha_sec_occ = alpha_sec_occ
        self._sigma_sec = sigma_sec
        self._p_sec = p_sec
        self._Y_sec = Y_sec
        self._nu_sec = nu_sec
        self._w_occ = w_occ

    def out(self):
        start_values = np.array([0.7, 0.2, 0.2, 0.4, 0.2, 0.4, 0.3, 0.2, 0.5])
        x, _ = utils.broyden_solver(self._res, start_values, noisy=True, maxcount=100)
        return [x[:3], x[3:6], x[6:]]

    def _res(self, x):
        N_sec_occ = [x[:3], x[3:6], x[6:]]
        L_sec = [self._calc_l_sec(N_sec_occ, i) for i in range(3)]
        return np.array([self._calc_err(N_sec_occ, L_sec, i, j) for i, j in product(range(3), repeat=2)])

    def _calc_l_sec(self, N_sec_occ: list, i: int):
        return (self._alpha_sec_occ[i][0] * N_sec_occ[i][0] ** self._sigma_sec[i] +
                self._alpha_sec_occ[i][1] * N_sec_occ[i][1] ** self._sigma_sec[i] +
                self._alpha_sec_occ[i][2] * N_sec_occ[i][2] ** self._sigma_sec[i]) ** (1 / self._sigma_sec[i])

    def _calc_err(self, N_sec_occ: list, L_sec: list, i: int, j: int):
        return N_sec_occ[i][j] - ((self._p_sec[i] * self._Y_sec[i] * (1 - self._nu_sec[i]) * L_sec[i] ** (
            -self._sigma_sec[i]) * self._alpha_sec_occ[i][j]) / self._w_occ[j]) ** (1 / (1 - self._sigma_sec[i]))
