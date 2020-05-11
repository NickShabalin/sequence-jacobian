def income1(w_occ, gamma_hh_1, m1, N = 0.33):
    N_occ1 = N * m1 *  gamma_hh_1[0]
    N_occ2 = N * m1 *  gamma_hh_1[1]
    N_occ3 = N * m1 * gamma_hh_1[2]


    choice1 = N_occ1 * w_occ[0]
    choice2 = N_occ2 * w_occ[1]
    choice3 = N_occ3 * w_occ[2]
    choices = [choice1, choice2, choice3]

    occupation = choices.index(max(choices))

    N_hh_occ_1_1 = (occupation == 0) * N_occ1
    N_hh_occ_1_2 = (occupation == 1) * N_occ2
    N_hh_occ_1_3 = (occupation == 2) * N_occ3

    return N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3


def income2(w_occ_1, w_occ_2, w_occ_3, gamma_hh_2, m2, N = 0.33):

    gamma_occ1 = gamma_hh_2[0]
    gamma_occ2 = gamma_hh_2[1]
    gamma_occ3 = gamma_hh_2[2]

    N_occ1 = N * m2 * gamma_occ1
    N_occ2 = N * m2 * gamma_occ2
    N_occ3 = N * m2 * gamma_occ3


    choice1 = N_occ1 * w_occ_1
    choice2 = N_occ2 * w_occ_2
    choice3 = N_occ3 * w_occ_3
    choices = [choice1, choice2, choice3]

    occupation = choices.index(max(choices))

    N_hh_occ_2_1 = (occupation == 0) * N_occ1
    N_hh_occ_2_2 = (occupation == 1) * N_occ2
    N_hh_occ_2_3 = (occupation == 2) * N_occ3

    return N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3


def income3(w_occ_1, w_occ_2, w_occ_3, gamma_hh_3, m3, N = 0.33):

    gamma_occ1 = gamma_hh_3[0]
    gamma_occ2 = gamma_hh_3[1]
    gamma_occ3 = gamma_hh_3[2]

    N_occ1 = N * m3 * gamma_occ1
    N_occ2 = N * m3 * gamma_occ2
    N_occ3 = N * m3 * gamma_occ3


    choice1 = N_occ1 * w_occ_1
    choice2 = N_occ2 * w_occ_2
    choice3 = N_occ3 * w_occ_3
    choices = [choice1, choice2, choice3]

    occupation = choices.index(max(choices))

    N_hh_occ_3_1 = (occupation == 0) * N_occ1
    N_hh_occ_3_2 = (occupation == 1) * N_occ2
    N_hh_occ_3_3 = (occupation == 2) * N_occ3


    return N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3


def income(e_grid, tax, w1, w2, w3, gamma, m, N):

    gamma_occ1 = gamma[0]
    gamma_occ2 = gamma[1]
    gamma_occ3 = gamma[2]

    N_occ1 = N * m * gamma_occ1
    N_occ2 = N * m * gamma_occ2
    N_occ3 = N * m * gamma_occ3


    choice1 = N_occ1 * w1
    choice2 = N_occ2 * w2
    choice3 = N_occ3 * w3
    choices = [choice1, choice2, choice3]

    z_grid = (1 - tax) * e_grid * max(choices)

    return z_grid