import utils
import numpy as np


def out(sigma_sec1, sigma_sec2, sigma_sec3,
        p_sec1, p_sec2, p_sec3,
        Y_sec1, Y_sec2, Y_sec3,
        nu_sec1, nu_sec2, nu_sec3,
        N_sec_occ_1_1, N_sec_occ_1_2, N_sec_occ_1_3,
        N_sec_occ_2_1, N_sec_occ_2_2, N_sec_occ_2_3,
        N_sec_occ_3_1, N_sec_occ_3_2, N_sec_occ_3_3,
        N1, N2, N3, pi, e_grid,
        gamma_hh_occ11, gamma_hh_occ12, gamma_hh_occ13,
        gamma_hh_occ21, gamma_hh_occ22, gamma_hh_occ23,
        gamma_hh_occ31, gamma_hh_occ32, gamma_hh_occ33):

    def res(x):
        w_occ1, w_occ2, w_occ3, N_sec1, N_sec2, N_sec3, m1, m2, m3, N_hh_eff_1, N_hh_eff_2, N_hh_eff_3 = x

        N_sec_occ11 = N_sec1 * N_sec_occ_1_1
        N_sec_occ12 = N_sec1 * N_sec_occ_1_2
        N_sec_occ13 = N_sec1 * N_sec_occ_1_3
        N_sec_occ21 = N_sec2 * N_sec_occ_2_1
        N_sec_occ22 = N_sec2 * N_sec_occ_2_2
        N_sec_occ23 = N_sec2 * N_sec_occ_2_3
        N_sec_occ31 = N_sec3 * N_sec_occ_3_1
        N_sec_occ32 = N_sec3 * N_sec_occ_3_2
        N_sec_occ33 = N_sec3 * N_sec_occ_3_3

        err4 = N_sec_occ11 + N_sec_occ21 + N_sec_occ31 - N_hh_eff_1 * m1
        err5 = N_sec_occ12 + N_sec_occ22 + N_sec_occ32 - N_hh_eff_2 * m2
        err6 = N_sec_occ13 + N_sec_occ23 + N_sec_occ33 - N_hh_eff_3 * m3

        alpha_sec_occ11 = (N_sec_occ11 ** (1 - sigma_sec1) * w_occ1) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 +
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 +
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
        alpha_sec_occ12 = (N_sec_occ12 ** (1 - sigma_sec1) * w_occ2) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 +
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 +
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
        alpha_sec_occ13 = (N_sec_occ13 ** (1 - sigma_sec1) * w_occ3) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 +
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 +
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
        alpha_sec_occ21 = (N_sec_occ21 ** (1 - sigma_sec2) * w_occ1) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 +
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 +
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
        alpha_sec_occ22 = (N_sec_occ22 ** (1 - sigma_sec2) * w_occ2) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 +
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 +
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
        alpha_sec_occ23 = (N_sec_occ23 ** (1 - sigma_sec2) * w_occ3) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 +
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 +
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
        alpha_sec_occ31 = (N_sec_occ31 ** (1 - sigma_sec3) * w_occ1) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 +
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 +
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)
        alpha_sec_occ32 = (N_sec_occ32 ** (1 - sigma_sec3) * w_occ2) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 +
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 +
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)
        alpha_sec_occ33 = (N_sec_occ33 ** (1 - sigma_sec3) * w_occ3) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 +
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 +
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)


        L_sec1 = (alpha_sec_occ11 * N_sec_occ11 ** sigma_sec1 + alpha_sec_occ12 * N_sec_occ12 ** sigma_sec1 + alpha_sec_occ13 * N_sec_occ13 ** sigma_sec1) ** (1 / sigma_sec1)
        L_sec2 = (alpha_sec_occ21 * N_sec_occ21 ** sigma_sec2 + alpha_sec_occ22 * N_sec_occ22 ** sigma_sec2 + alpha_sec_occ23 * N_sec_occ23 ** sigma_sec2) ** (1 / sigma_sec2)
        L_sec3 = (alpha_sec_occ31 * N_sec_occ31 ** sigma_sec3 + alpha_sec_occ32 * N_sec_occ32 ** sigma_sec3 + alpha_sec_occ33 * N_sec_occ33 ** sigma_sec3) ** (1 / sigma_sec3)

        err1 = N_sec_occ11 - (
                    (p_sec1 * Y_sec1 * (1 - nu_sec1) * L_sec1 ** (-sigma_sec1) * alpha_sec_occ11) / w_occ1) ** (
                       1 / (1 - sigma_sec1))

        err2 = N_sec_occ22 - (
                (p_sec2 * Y_sec2 * (1 - nu_sec2) * L_sec2 ** (-sigma_sec2) * alpha_sec_occ22) / w_occ2) ** (
                       1 / (1 - sigma_sec2))


        err3 = N_sec_occ33 - (
                (p_sec3 * Y_sec3 * (1 - nu_sec3) * L_sec3 ** (-sigma_sec3) * alpha_sec_occ33) / w_occ3) ** (
                       1 / (1 - sigma_sec3))

        wage_normalizer = 0.19

        err7 = w_occ1 - 1 * wage_normalizer
        err8 = w_occ2 - 1.944502 * wage_normalizer
        err9 = w_occ3 - 1.563125 * wage_normalizer

        err10 = N1 * pi[2] * m1 * (1 + gamma_hh_occ11) ** e_grid[2] - N_hh_eff_1
        err11 = N2 * (pi[0] * m1 * (1 + gamma_hh_occ12) ** e_grid[0] +
                           pi[1] * m1 * (1 + gamma_hh_occ12) ** e_grid[1] +
                           pi[0] * m2 * (1 + gamma_hh_occ22) ** e_grid[0] +
                           pi[1] * m2 * (1 + gamma_hh_occ22) ** e_grid[1] +
                           pi[2] * m2 * (1 + gamma_hh_occ22) ** e_grid[2] +
                           pi[0] * m3 * (1 + gamma_hh_occ32) ** e_grid[0]) - N_hh_eff_2
        err12 = N3 * (pi[2] * m3 * (1 + gamma_hh_occ33) ** e_grid[2] +
                      pi[1] * m3 * (1 + gamma_hh_occ33) ** e_grid[1]) - N_hh_eff_3



        return np.array([err1, err2, err3, err4, err5, err6, err7, err8, err9, err10, err11, err12])

    start_values = np.array([1, 1.944502, 1.563125, 0.1, 0.05, 0.3, 0.33, 0.25, 0.55, 1, 1, 1])
    (w_occ1, w_occ2, w_occ3, N_sec1, N_sec2, N_sec3, m1, m2, m3, N_hh_eff_1, N_hh_eff_2, N_hh_eff_3), _ = utils.broyden_solver(res, start_values, noisy=True, maxcount=100)


    return (w_occ1, w_occ2, w_occ3, N_sec1, N_sec2, N_sec3, m1, m2, m3, N_hh_eff_1, N_hh_eff_2, N_hh_eff_3)


