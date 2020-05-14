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
def income_grid(e_grid: ndarray,
                tax: float,
                w_occ: List[float],
                gamma_hh: List[float],
                m: float,
                N_hh_occ: List[float]) -> ndarray:
    occupation = N_hh_occ.index(max(N_hh_occ))
    z_grid = (1 - tax) * e_grid * m * gamma_hh[occupation] * w_occ[occupation] * N_hh_occ[occupation]
    return z_grid
