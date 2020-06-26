import numpy as np
from numba import njit
import utils
from het_block import het




@het(exogenous='Pi', policy=['b1', 'a1'], backward=['Vb1', 'Va1'])  # order as in grid!
def household1(Va1_p, Vb1_p, Pi_p, a1_grid, b1_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2,
               gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3, w_occ_1, w_occ_2, w_occ_3):

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
    #u = e_grid[:, np.newaxis, np.newaxis] * uc
    w = [w_occ_2, w_occ_2, w_occ_1]
    gamma = [gamma_hh_1_2, gamma_hh_1_2, gamma_hh_1_1]
    u = [w_k * (1 + gamma_k) ** e_grid_k * uc_k for gamma_k, w_k, e_grid_k, uc_k in zip(gamma, w, e_grid, uc)]
    u = np.reshape(u, (3, 50, 70))

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
def household2(Va2_p, Vb2_p, Pi_p, a2_grid, b2_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2,
               gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3, w_occ_1, w_occ_2, w_occ_3):
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
    #u = e_grid[:, np.newaxis, np.newaxis] * uc
    w = [w_occ_3, w_occ_2, w_occ_2]
    gamma = [gamma_hh_2_3, gamma_hh_2_2, gamma_hh_2_2]
    u = [w_k * (1 + gamma_k) ** e_grid_k * uc_k for gamma_k, w_k, e_grid_k, uc_k in zip(gamma, w, e_grid, uc)]
    u = np.reshape(u, (3, 50, 70))

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
def household3(Va3_p, Vb3_p, Pi_p, a3_grid, b3_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2,
               gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3, w_occ_1, w_occ_2, w_occ_3):
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
    #u = e_grid[:, np.newaxis, np.newaxis] * uc
    w = [w_occ_2, w_occ_2, w_occ_3]
    gamma = [gamma_hh_3_2, gamma_hh_3_2, gamma_hh_3_3]
    u = [w_k * (1 + gamma_k) ** e_grid_k * uc_k for gamma_k, w_k, e_grid_k, uc_k in zip(gamma, w, e_grid, uc)]
    u = np.reshape(u, (3, 50, 70))

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



def post_decision_vfun(Va_p, Vb_p, Pi, beta):
    Wb = (Vb_p.T @ (beta * Pi.T)).T
    Wa = (Va_p.T @ (beta * Pi.T)).T
    return Wb, Wa


def Psi_fun(ap, a, ra, chi0, chi1, chi2):
    return chi1 / chi2 * np.abs((ap - (1 + ra) * a)) ** chi2 / ((1 + ra) * a + chi0)


def Psi1_fun(ap, a, ra, chi0, chi1, chi2):
    return np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)


def Psi2_fun(ap, a, ra, chi0, chi1, chi2):
    Psi1 = np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)
    return -(1 + ra) * (Psi1 + chi1 * (chi2 - 1) / chi2 * (np.abs(ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** chi2)


@njit
def step3(lhs, rhs, Wb, a_grid, eis, nZ, nB, nA):
    ap_endo = np.empty((nZ, nB, nA))
    Wb_endo = np.empty((nZ, nB, nA))
    for iz in range(nZ):
        for ibp in range(nB):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ibp, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ibp, ia] = 0
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, 0]
                elif iap == nA:
                    ap_endo[iz, ibp, ia] = a_grid[iap]
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap]
                else:
                    y0 = lhs[iz, ibp, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ibp, iap] - rhs[iap, ia]
                    ap_endo[iz, ibp, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap - 1] + (
                                ap_endo[iz, ibp, ia] - a_grid[iap - 1]) * (
                                Wb[iz, ibp, iap] - Wb[iz, ibp, iap - 1]) / (a_grid[iap] - a_grid[iap - 1])
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step4(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, b', a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + bbb - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) > 0, 'b(bp) is not increasing'
    # assert np.min(np.diff(ap_endo, axis=1)) > 0, 'ap(bp) is not increasing'
    i, pi = utils.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    ap = utils.apply_coord(i, pi, ap_endo.swapaxes(1, 2)).swapaxes(1, 2)
    bp = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2)
    return bp, ap


@njit
def step5(lhs, rhs, Wb, a_grid, k_grid, eis, nZ, nK, nA):
    ap_endo = np.empty((nZ, nK, nA))
    Wb_endo = np.empty((nZ, nK, nA))
    for iz in range(nZ):
        for ik in range(nK):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ik, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ik, ia] = 0
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, 0]
                elif iap == nA:
                    ap_endo[iz, ik, ia] = a_grid[iap]
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, iap]
                else:
                    y0 = lhs[iz, ik, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ik, iap] - rhs[iap, ia]
                    ap_endo[iz, ik, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * (
                            Wb[iz, 0, iap - 1] + (ap_endo[iz, ik, ia] - a_grid[iap - 1]) *
                            (Wb[iz, 0, iap] - Wb[iz, 0, iap - 1]) / (a_grid[iap] - a_grid[iap - 1]))
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step6(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, k, a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + b_grid[0] - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) < 0, 'b(kappa) is not decreasing'
    # assert np.min(np.diff(ap_endo, axis=1)) < 0, 'ap(kappa) is not decreasing'
    ap = utils.interpolate_y(b_endo[:, ::-1, :].swapaxes(1, 2), b_grid,
                             ap_endo[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)
    return ap


'''Part 2: Simple blocks'''

def income1(N):
    N_hh_occ_1_1 = N
    N_hh_occ_1_2 = N
    N_hh_occ_1_3 = N

    return N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3

def income2(N):
    N_hh_occ_2_1 = N
    N_hh_occ_2_2 = N
    N_hh_occ_2_3 = N

    return N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3

def income3(N):
    N_hh_occ_3_1 = N
    N_hh_occ_3_2 = N
    N_hh_occ_3_3 = N

    return N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3


# def income_hh_1(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3,
#                 m1, N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3):
#     N_hh_1 = [N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3]
#     gamma_hh_1 = [gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3]
#     w_occ = [w_occ_1, w_occ_2, w_occ_3]
#     z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 * m1 for k in range(len(e_grid))]
#     z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 * m1 for k in range(len(e_grid))]
#     z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 * m1 for k in range(len(e_grid))]
#     all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
#     z_grid = all_grids.max(axis=1)
#     return z_grid
#
# def income_hh_2(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3,
#                 m2, N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3):
#     N_hh_1 = [N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3]
#     gamma_hh_1 = [gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3]
#     w_occ = [w_occ_1, w_occ_2, w_occ_3]
#     z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 * m2 for k in range(len(e_grid))]
#     z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 * m2 for k in range(len(e_grid))]
#     z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 * m2 for k in range(len(e_grid))]
#     all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
#     z_grid = all_grids.max(axis=1)
#     return z_grid
#
# def income_hh_3(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3,
#                 m3, N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3):
#     N_hh_1 = [N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3]
#     gamma_hh_1 = [gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3]
#     w_occ = [w_occ_1, w_occ_2, w_occ_3]
#     z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 * m3 for k in range(len(e_grid))]
#     z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 * m3 for k in range(len(e_grid))]
#     z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 * m3 for k in range(len(e_grid))]
#     all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
#     z_grid = all_grids.max(axis=1)
#     return z_grid


def income_hh_1(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3,
                m1, N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3):
    N_hh_1 = [N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3]
    gamma_hh_1 = [gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 for k in range(len(e_grid))]
    z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 for k in range(len(e_grid))]
    z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 for k in range(len(e_grid))]
    all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
    z_grid = all_grids.max(axis=0)
    return z_grid

def income_hh_2(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3,
                m2, N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3):
    N_hh_1 = [N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3]
    gamma_hh_1 = [gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 for k in range(len(e_grid))]
    z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 for k in range(len(e_grid))]
    z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 for k in range(len(e_grid))]
    all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
    z_grid = all_grids.max(axis=0)
    return z_grid

def income_hh_3(q_1, q_2, q_3, e_grid, tax, w_occ_1, w_occ_2, w_occ_3, gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3,
                m3, N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3):
    N_hh_1 = [N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3]
    gamma_hh_1 = [gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]
    z_grid_1 = [(1 - tax) * (1 + gamma_hh_1[0]) ** e_grid[k] * w_occ[0] * N_hh_1[0] * q_1 for k in range(len(e_grid))]
    z_grid_2 = [(1 - tax) * (1 + gamma_hh_1[1]) ** e_grid[k] * w_occ[1] * N_hh_1[1] * q_2 for k in range(len(e_grid))]
    z_grid_3 = [(1 - tax) * (1 + gamma_hh_1[2]) ** e_grid[k] * w_occ[2] * N_hh_1[2] * q_3 for k in range(len(e_grid))]
    all_grids = np.array([z_grid_1, z_grid_2, z_grid_3])
    z_grid = all_grids.max(axis=0)
    return z_grid



# comment: two different types of hh instead of one
household_inc1 = household1.attach_hetinput(income_hh_1)
household_inc2 = household2.attach_hetinput(income_hh_2)
household_inc3 = household3.attach_hetinput(income_hh_3)

