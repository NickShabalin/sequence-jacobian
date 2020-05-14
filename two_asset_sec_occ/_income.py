from typing import List

from numpy import ndarray

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
