from typing import List

from numpy import ndarray


# noinspection PyPep8Naming
def income_labour_supply(w_occ: List[float], gamma_hh: List[float], m: float, N: float = 0.33) -> List[float]:
    N_occ = [N * m * gamma_hh[i] for i in range(3)]
    choices = [N_occ[i] * w_occ[i] for i in range(3)]
    occupation = choices.index(max(choices))
    N_hh_occ = [(occupation == i) * N_occ[i] for i in range(3)]
    return N_hh_occ


# noinspection PyPep8Naming
def income_grid(e_grid: ndarray, tax: float, w: List[float], gamma: List[float], m: float, N: float) -> ndarray:
    N_occ = [N * m * gamma[i] for i in range(3)]
    choices = [N_occ[i] * w[i] for i in range(3)]
    z_grid = (1 - tax) * e_grid * max(choices)
    return z_grid
