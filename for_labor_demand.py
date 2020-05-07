import utils
import numpy as np


def out(alpha_sec_occ11, alpha_sec_occ13, alpha_sec_occ12,
        alpha_sec_occ21, alpha_sec_occ22, alpha_sec_occ23,
        alpha_sec_occ31, alpha_sec_occ32, alpha_sec_occ33,
        sigma_sec1, sigma_sec2, sigma_sec3,
        p_sec1, p_sec2, p_sec3,
        Y_sec1, Y_sec2, Y_sec3,
        nu_sec1, nu_sec2, nu_sec3,
        w_occ1, w_occ2, w_occ3):


    def res(x):
        N_sec_occ11, N_sec_occ12, N_sec_occ13, N_sec_occ21, N_sec_occ22, N_sec_occ23, N_sec_occ31, N_sec_occ32, N_sec_occ33 = x

        L_sec1 = (alpha_sec_occ11 * N_sec_occ11 ** sigma_sec1 + alpha_sec_occ12 * N_sec_occ12 ** sigma_sec1 + alpha_sec_occ13 * N_sec_occ13 ** sigma_sec1) ** (1 / sigma_sec1)
        L_sec2 = (alpha_sec_occ21 * N_sec_occ21 ** sigma_sec2 + alpha_sec_occ22 * N_sec_occ22 ** sigma_sec2 + alpha_sec_occ23 * N_sec_occ23 ** sigma_sec2) ** (1 / sigma_sec2)
        L_sec3 = (alpha_sec_occ31 * N_sec_occ31 ** sigma_sec3 + alpha_sec_occ32 * N_sec_occ32 ** sigma_sec3 + alpha_sec_occ33 * N_sec_occ33 ** sigma_sec3) ** (1 / sigma_sec3)

        err1 = N_sec_occ11 - (
                    (p_sec1 * Y_sec1 * (1 - nu_sec1) * L_sec1 ** (-sigma_sec1) * alpha_sec_occ11) / w_occ1) ** (
                       1 / (1 - sigma_sec1))
        err2 = N_sec_occ12 - (
                    (p_sec1 * Y_sec1 * (1 - nu_sec1) * L_sec1 ** (-sigma_sec1) * alpha_sec_occ12) / w_occ2) ** (
                       1 / (1 - sigma_sec1))
        err3 = N_sec_occ13 - (
                    (p_sec1 * Y_sec1 * (1 - nu_sec1) * L_sec1 ** (-sigma_sec1) * alpha_sec_occ13) / w_occ3) ** (
                       1 / (1 - sigma_sec1))
        err4 = N_sec_occ21 - (
                    (p_sec2 * Y_sec2 * (1 - nu_sec2) * L_sec2 ** (-sigma_sec2) * alpha_sec_occ21) / w_occ1) ** (
                       1 / (1 - sigma_sec2))
        err5 = N_sec_occ22 - (
                    (p_sec2 * Y_sec2 * (1 - nu_sec2) * L_sec2 ** (-sigma_sec2) * alpha_sec_occ22) / w_occ2) ** (
                       1 / (1 - sigma_sec2))
        err6 = N_sec_occ23 - (
                    (p_sec2 * Y_sec2 * (1 - nu_sec2) * L_sec2 ** (-sigma_sec2) * alpha_sec_occ23) / w_occ3) ** (
                       1 / (1 - sigma_sec2))
        err7 = N_sec_occ31 - (
                    (p_sec3 * Y_sec3 * (1 - nu_sec3) * L_sec3 ** (-sigma_sec3) * alpha_sec_occ31) / w_occ1) ** (
                       1 / (1 - sigma_sec3))
        err8 = N_sec_occ32 - (
                    (p_sec3 * Y_sec3 * (1 - nu_sec3) * L_sec3 ** (-sigma_sec3) * alpha_sec_occ32) / w_occ2) ** (
                       1 / (1 - sigma_sec3))
        err9 = N_sec_occ33 - (
                    (p_sec3 * Y_sec3 * (1 - nu_sec3) * L_sec3 ** (-sigma_sec3) * alpha_sec_occ33) / w_occ3) ** (
                       1 / (1 - sigma_sec3))
        return np.array([err1, err2, err3, err4, err5, err6, err7, err8, err9])

    start_values = np.array([0.7, 0.2, 0.2, 0.4, 0.2, 0.4, 0.3, 0.2, 0.5])
    (N_sec_occ11, N_sec_occ12, N_sec_occ13, N_sec_occ21, N_sec_occ22, N_sec_occ23, N_sec_occ31, N_sec_occ32,
                            N_sec_occ33), _ = utils.broyden_solver(res, start_values, noisy=True, maxcount=100)


    return (N_sec_occ11, N_sec_occ12, N_sec_occ13, N_sec_occ21, N_sec_occ22, N_sec_occ23, N_sec_occ31, N_sec_occ32, N_sec_occ33)


