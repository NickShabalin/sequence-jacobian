import numpy as np

from het_block import het
from households.misc import Psi_fun, Psi1_fun, Psi2_fun, post_decision_vfun
from households.steps import step3, step4, step5, step6

'''Part 1: HA block'''


@het(exogenous='Pi', policy=['b1', 'a1'], backward=['Vb1', 'Va1'])  # order as in grid!
def household_1(Va1_p, Vb1_p, Pi_p, a1_grid, b1_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    a_grid = a1_grid
    b_grid = b1_grid
    # get grid dimensions
    nZ, nB, nA = Va1_p.shape
    nK = k_grid.shape[0]

    # step 2: Wb(z, b', a') and Wa(z, b', a')
    Wb, Wa = post_decision_vfun(Va1_p, Vb1_p, Pi_p, beta)

    # step 3: a'(z, b', a) for UNCONSTRAINED
    lhs_unc = Wa / Wb
    Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)
    a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, k_grid, eis, nZ, nK, nA)

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    Va1 = Va
    Vb1 = Vb
    a1 = a
    b1 = b
    c1 = c
    u1 = u

    return Va1, Vb1, a1, b1, c1, u1


@het(exogenous='Pi', policy=['b2', 'a2'], backward=['Vb2', 'Va2'])  # order as in grid!
def household_2(Va2_p, Vb2_p, Pi_p, a2_grid, b2_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    a_grid = a2_grid
    b_grid = b2_grid
    # get grid dimensions
    nZ, nB, nA = Va2_p.shape
    nK = k_grid.shape[0]

    # step 2: Wb(z, b', a') and Wa(z, b', a')
    Wb, Wa = post_decision_vfun(Va2_p, Vb2_p, Pi_p, beta)

    # step 3: a'(z, b', a) for UNCONSTRAINED
    lhs_unc = Wa / Wb
    Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)
    a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, k_grid, eis, nZ, nK, nA)

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    Va2 = Va
    Vb2 = Vb
    a2 = a
    b2 = b
    c2 = c
    u2 = u

    return Va2, Vb2, a2, b2, c2, u2


@het(exogenous='Pi', policy=['b3', 'a3'], backward=['Vb3', 'Va3'])  # order as in grid!
def household_3(Va3_p, Vb3_p, Pi_p, a3_grid, b3_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    a_grid = a3_grid
    b_grid = b3_grid
    # get grid dimensions
    nZ, nB, nA = Va3_p.shape
    nK = k_grid.shape[0]

    # step 2: Wb(z, b', a') and Wa(z, b', a')
    Wb, Wa = post_decision_vfun(Va3_p, Vb3_p, Pi_p, beta)

    # step 3: a'(z, b', a) for UNCONSTRAINED
    lhs_unc = Wa / Wb
    Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)
    a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, k_grid, eis, nZ, nK, nA)

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    Va3 = Va
    Vb3 = Vb
    a3 = a
    b3 = b
    c3 = c
    u3 = u

    return Va3, Vb3, a3, b3, c3, u3


def income_hh_1(e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3,
                m1, N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3):
    N_hh_1 = [N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3]
    occupation = N_hh_1.index(max(N_hh_1))
    gamma_hh_1 = [gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid = (1 - tax) * e_grid * m1 * gamma_hh_1[occupation] * w_occ[occupation] * N_hh_1[occupation]
    return z_grid


def income_hh_2(e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3,
                m2, N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3):
    N_hh_1 = [N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3]
    occupation = N_hh_1.index(max(N_hh_1))
    gamma_hh_1 = [gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid = (1 - tax) * e_grid * m2 * gamma_hh_1[occupation] * w_occ[occupation] * N_hh_1[occupation]
    return z_grid


def income_hh_3(e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3,
                m3, N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3):
    N_hh_1 = [N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3]
    occupation = N_hh_1.index(max(N_hh_1))
    gamma_hh_1 = [gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid = (1 - tax) * e_grid * m3 * gamma_hh_1[occupation] * w_occ[occupation] * N_hh_1[occupation]
    return z_grid


# comment: two different types of hh instead of one
household_inc_1 = household_1.attach_hetinput(income_hh_1)
household_inc_2 = household_2.attach_hetinput(income_hh_2)
household_inc_3 = household_3.attach_hetinput(income_hh_3)
