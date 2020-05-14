import numpy as np
from numba import njit
import utils
from het_block import het
from simple_block import simple
import for_labor_demand


'''Part 1: HA block'''


@het(exogenous='Pi', policy=['b1', 'a1'], backward=['Vb1', 'Va1'])  # order as in grid!
def household1(Va1_p, Vb1_p, Pi_p, a1_grid, b1_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):

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
def household2(Va2_p, Vb2_p, Pi_p, a2_grid, b2_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
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
def household3(Va3_p, Vb3_p, Pi_p, a3_grid, b3_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2):
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

def income1(w_occ_1, w_occ_2, w_occ_3, gamma_hh_1, m1, N = 0.33):

    gamma_occ1 = gamma_hh_1[0]
    gamma_occ2 = gamma_hh_1[1]
    gamma_occ3 = gamma_hh_1[2]

    N_occ1 = N * m1 * gamma_occ1
    N_occ2 = N * m1 * gamma_occ2
    N_occ3 = N * m1 * gamma_occ3


    choice1 = N_occ1 * w_occ_1
    choice2 = N_occ2 * w_occ_2
    choice3 = N_occ3 * w_occ_3
    choices = [choice1, choice2, choice3]

    occupation = choices.index(max(choices))

    N_hh_occ_1_1 = (occupation == 0) * N
    N_hh_occ_1_2 = (occupation == 1) * N
    N_hh_occ_1_3 = (occupation == 2) * N

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

    N_hh_occ_2_1 = (occupation == 0) * N
    N_hh_occ_2_2 = (occupation == 1) * N
    N_hh_occ_2_3 = (occupation == 2) * N

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

    N_hh_occ_3_1 = (occupation == 0) * N
    N_hh_occ_3_2 = (occupation == 1) * N
    N_hh_occ_3_3 = (occupation == 2) * N


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



# comment: two different types of hh instead of one
household_inc1 = household1.attach_hetinput(income)
household_inc2 = household2.attach_hetinput(income)
household_inc3 = household3.attach_hetinput(income)

'''Part 3: Steady state'''


def hank_ss(beta_guess=0.976, vphi_guess=2.07, chi1_guess=6.5, r=0.0125, delta=0.02, kappap=0.1,
            muw=1.1, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
            phi=1.5, nZ=3, nB=50, nA=70, nK=50, bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92,
            tot_wealth=14, Bh=1.04/2, Bg=2.8/2, G=0.2/2, noisy=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)
    k_grid = utils.agrid(amax=kmax, n=nK)
    e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)


    # solve analytically what we can
    mup = 6.3863129
    #mup = 1.009861
    #mup = 1.00648
    #mup = 1.015
    #mup = 1.00985
    #mup = 1.5 # to get A + B = 14Y
    eta = (1 - mup) / mup

    gamma_hh_occ11 = 1
    gamma_hh_occ12 = 0.2
    gamma_hh_occ13 = 0.3
    gamma_hh1 = [gamma_hh_occ11, gamma_hh_occ12, gamma_hh_occ13]

    gamma_hh_occ21 = 0.278417140245438
    gamma_hh_occ22 = 1
    gamma_hh_occ23 = 0.357559561729431
    gamma_hh2 = [gamma_hh_occ21, gamma_hh_occ22, gamma_hh_occ23]

    gamma_hh_occ31 = 0.257629066705704
    gamma_hh_occ32 = 0.32952556014061
    gamma_hh_occ33 = 1
    gamma_hh3 = [gamma_hh_occ31, gamma_hh_occ32, gamma_hh_occ33]

    p = 1
    '''
    w_occ1 = 1
    w_occ2 = 1.944502
    w_occ3 = 1.563125
    '''

    w_occ1 = 0.35
    w_occ2 = 1.5
    w_occ3 = 0.7

    sigma_sec1 = sigma_sec2 = sigma_sec3 = 0.2

    eta = (1 - mup) / mup

    # new values will be here
    Y_sec1 = 0.260986566543579
    Y_sec2 = 0.343330949544907
    Y_sec3 = 0.398539662361145

    Y = (Y_sec1 ** ((eta - 1) / eta) + Y_sec2 ** ((eta - 1) / eta) + Y_sec3 ** ((eta - 1) / eta)) ** (eta / (eta - 1))

    nu_sec1 = 0.425027757883072
    nu_sec2 = 0.538959443569183
    nu_sec3 = 0.273549377918243

    Q_sec1 = 1
    Q_sec2 = 1
    Q_sec3 = 1

    N_sec_occ11 = 0.66251665353775
    N_sec_occ12 = 0.182436376810074
    N_sec_occ13 = 0.155046954751015
    N_sec_occ21 = 0.372738629579544
    N_sec_occ22 = 0.19160558283329
    N_sec_occ23 = 0.435655802488327
    N_sec_occ31 = 0.255537301301956
    N_sec_occ32 = 0.227060705423355
    N_sec_occ33 = 0.517401993274688

    alpha_sec_occ11 = (N_sec_occ11 ** (1 - sigma_sec1) * w_occ1) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 + \
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 + \
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
    alpha_sec_occ12 = (N_sec_occ12 ** (1 - sigma_sec1) * w_occ2) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 + \
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 + \
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
    alpha_sec_occ13 = (N_sec_occ13 ** (1 - sigma_sec1) * w_occ3) / (
                N_sec_occ11 ** (1 - sigma_sec1) * w_occ1 + \
                N_sec_occ12 ** (1 - sigma_sec1) * w_occ2 + \
                N_sec_occ13 ** (1 - sigma_sec1) * w_occ3)
    alpha_sec_occ21 = (N_sec_occ21 ** (1 - sigma_sec2) * w_occ1) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 + \
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 + \
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
    alpha_sec_occ22 = (N_sec_occ22 ** (1 - sigma_sec2) * w_occ2) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 + \
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 + \
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
    alpha_sec_occ23 = (N_sec_occ23 ** (1 - sigma_sec2) * w_occ3) / (
                N_sec_occ21 ** (1 - sigma_sec2) * w_occ1 + \
                N_sec_occ22 ** (1 - sigma_sec2) * w_occ2 + \
                N_sec_occ23 ** (1 - sigma_sec2) * w_occ3)
    alpha_sec_occ31 = (N_sec_occ31 ** (1 - sigma_sec3) * w_occ1) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 + \
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 + \
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)
    alpha_sec_occ32 = (N_sec_occ32 ** (1 - sigma_sec3) * w_occ2) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 + \
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 + \
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)
    alpha_sec_occ33 = (N_sec_occ33 ** (1 - sigma_sec3) * w_occ3) / (
                N_sec_occ31 ** (1 - sigma_sec3) * w_occ1 + \
                N_sec_occ32 ** (1 - sigma_sec3) * w_occ2 + \
                N_sec_occ33 ** (1 - sigma_sec3) * w_occ3)

    p_sec1 = (Y / Y_sec1) ** (1 / eta) * p
    p_sec2 = (Y / Y_sec2) ** (1 / eta) * p
    p_sec3 = (Y / Y_sec3) ** (1 / eta) * p

    N_sec_occ11, N_sec_occ12, N_sec_occ13, N_sec_occ21, \
    N_sec_occ22, N_sec_occ23, N_sec_occ31, N_sec_occ32, N_sec_occ33 = for_labor_demand.out(alpha_sec_occ11, alpha_sec_occ13, alpha_sec_occ12,
                                                                                           alpha_sec_occ21, alpha_sec_occ22, alpha_sec_occ23,
                                                                                           alpha_sec_occ31, alpha_sec_occ32, alpha_sec_occ33,
                                                                                           sigma_sec1, sigma_sec2, sigma_sec3,
                                                                                           p_sec1, p_sec2, p_sec3,
                                                                                           Y_sec1, Y_sec2, Y_sec3,
                                                                                           nu_sec1, nu_sec2, nu_sec3,
                                                                                           w_occ1, w_occ2, w_occ3)

    L_sec1 = (alpha_sec_occ11 * N_sec_occ11 ** sigma_sec1 + alpha_sec_occ12 * N_sec_occ12 ** sigma_sec1 + alpha_sec_occ13 * N_sec_occ13 ** sigma_sec1) ** (1 / sigma_sec1)
    L_sec2 = (alpha_sec_occ21 * N_sec_occ21 ** sigma_sec2 + alpha_sec_occ22 * N_sec_occ22 ** sigma_sec2 + alpha_sec_occ23 * N_sec_occ23 ** sigma_sec2) ** (1 / sigma_sec2)
    L_sec3 = (alpha_sec_occ31 * N_sec_occ31 ** sigma_sec3 + alpha_sec_occ32 * N_sec_occ32 ** sigma_sec3 + alpha_sec_occ33 * N_sec_occ33 ** sigma_sec3) ** (1 / sigma_sec3)


    K_sec1 = 1 / (1 + r * Q_sec1) * nu_sec1 / (1 - nu_sec1) * L_sec1 ** sigma_sec1 * (
            w_occ1 * N_sec_occ11 ** (1 - sigma_sec1) / alpha_sec_occ11 + \
            w_occ2 * N_sec_occ12 ** (1 - sigma_sec1) / alpha_sec_occ12 + w_occ3 * N_sec_occ13 ** (
                    1 - sigma_sec3) / alpha_sec_occ13)

    K_sec2 = 1 / (1 + r * Q_sec2) * nu_sec2 / (1 - nu_sec2) * L_sec2 ** sigma_sec2 * (
            w_occ1 * N_sec_occ21 ** (1 - sigma_sec2) / alpha_sec_occ21 + \
            w_occ2 * N_sec_occ22 ** (1 - sigma_sec2) / alpha_sec_occ22 + w_occ3 * N_sec_occ23 ** (
                    1 - sigma_sec2) / alpha_sec_occ23)

    K_sec3 = 1 / (1 + r * Q_sec3) * nu_sec3 / (1 - nu_sec3) * L_sec3 ** sigma_sec3 * (
            w_occ1 * N_sec_occ31 ** (1 - sigma_sec3) / alpha_sec_occ31 + \
            w_occ2 * N_sec_occ32 ** (1 - sigma_sec3) / alpha_sec_occ32 + w_occ3 * N_sec_occ33 ** (
                    1 - sigma_sec3) / alpha_sec_occ33)

    mc_sec1 = (r * Q_sec1 + delta) * K_sec1 / nu_sec1 / Y_sec1
    mc_sec2 = (r * Q_sec2 + delta) * K_sec2 / nu_sec2 / Y_sec2
    mc_sec3 = (r * Q_sec3 + delta) * K_sec3 / nu_sec3 / Y_sec3

    mc = mc_sec1 + mc_sec2 + mc_sec3

    m1 = 0.33
    m2 = 0.33
    m3 = 0.33

    Q = 1

    ra = r
    rb = r - omega

    K = K_sec1 + K_sec2 + K_sec3

    I_sec1 = delta * K_sec1
    I_sec2 = delta * K_sec2
    I_sec3 = delta * K_sec3

    I = delta * K

    productivity_sec1 = Y_sec1 * K_sec1 ** (-nu_sec1) * L_sec1 ** (nu_sec1 - 1)
    productivity_sec2 = Y_sec2 * K_sec2 ** (-nu_sec2) * L_sec2 ** (nu_sec2 - 1)
    productivity_sec3 = Y_sec3 * K_sec3 ** (-nu_sec3) * L_sec3 ** (nu_sec3 - 1)

    N_sec1 = N_sec_occ11 + N_sec_occ12 + N_sec_occ13
    N_sec2 = N_sec_occ21 + N_sec_occ22 + N_sec_occ23
    N_sec3 = N_sec_occ31 + N_sec_occ32 + N_sec_occ33

    N_occ1 = N_sec_occ11 + N_sec_occ21 + N_sec_occ31
    N_occ2 = N_sec_occ12 + N_sec_occ22 + N_sec_occ32
    N_occ3 = N_sec_occ13 + N_sec_occ23 + N_sec_occ33

    w_sec1 = (w_occ1 * N_sec_occ11 + w_occ2 * N_sec_occ12 + w_occ3 * N_sec_occ13) / N_sec1
    w_sec2 = (w_occ1 * N_sec_occ21 + w_occ2 * N_sec_occ22 + w_occ3 * N_sec_occ23) / N_sec2
    w_sec3 = (w_occ1 * N_sec_occ31 + w_occ2 * N_sec_occ32 + w_occ3 * N_sec_occ33) / N_sec3

    N = N_sec1 + N_sec2 + N_sec3

    w = (w_sec1 * N_sec1 + w_sec2 * N_sec2 + w_sec3 * N_sec3) / N

    tax = (r * Bg + G) / w / N
    div = p * Y - w * N - I
    equity_price = div / r
    pshare = equity_price / (tot_wealth - Bh)

    N1 = N2 = N3 = 0.33

    N_hh_occ11, N_hh_occ12, N_hh_occ13 = income1(w_occ1, w_occ2, w_occ3, gamma_hh1, m1, N1)
    N_hh_occ21, N_hh_occ22, N_hh_occ23 = income2(w_occ1, w_occ2, w_occ3, gamma_hh2, m2, N2)
    N_hh_occ31, N_hh_occ32, N_hh_occ33 = income3(w_occ1, w_occ2, w_occ3, gamma_hh3, m3, N3)

    z1_grid = income(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh1, m1, N1)
    z2_grid = income(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh2, m2, N2)
    z3_grid = income(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh3, m3, N3)

    Va1 = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z1_grid.shape[0], 1, 1))
    Vb1 = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z1_grid.shape[0], 1, 1))

    Va2 = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z2_grid.shape[0], 1, 1))
    Vb2 = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z2_grid.shape[0], 1, 1))

    Va3 = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z3_grid.shape[0], 1, 1))
    Vb3 = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z3_grid.shape[0], 1, 1))

    wages = [w_occ1, w_occ2, w_occ3]
    labor_hours_hh1 = [N_hh_occ11, N_hh_occ12, N_hh_occ13]
    labor_hours_hh2 = [N_hh_occ21, N_hh_occ22, N_hh_occ23]
    labor_hours_hh3 = [N_hh_occ31, N_hh_occ32, N_hh_occ33]
    occupation1 = 0
    occupation2 = 1
    occupation3 = 2
    wage1 = wages[occupation1]
    labor_hours1 = labor_hours_hh1[occupation1]
    wage2 = wages[occupation2]
    labor_hours2 = labor_hours_hh2[occupation2]
    wage3 = wages[occupation3]
    labor_hours3 = labor_hours_hh3[occupation3]

    '''
    mc_sec1 = w_sec1 * (N_sec_occ11 + N_sec_occ12 + N_sec_occ13) / (1 - nu_sec1) / Y_sec1
    mc_sec2 = w_sec2 * (N_sec_occ21 + N_sec_occ22 + N_sec_occ23) / (1 - nu_sec2) / Y_sec2
    mc_sec3 = w_sec3 * (N_sec_occ31 + N_sec_occ32 + N_sec_occ33) / (1 - nu_sec3) / Y_sec3
    mc = mc_sec1 + mc_sec2 + mc_sec3
    '''

    div_sec1 = Y_sec1 * p_sec1 - w_sec1 * N_sec1 - I_sec1
    div_sec2 = Y_sec2 * p_sec2 - w_sec2 * N_sec2 - I_sec2
    div_sec3 = Y_sec3 * p_sec3 - w_sec3 * N_sec3 - I_sec3

    #err5 = div - div_sec1 - div_sec2 - div_sec3


    equity_price_sec1 = div_sec1 / r
    equity_price_sec2 = div_sec2 / r
    equity_price_sec3 = div_sec3 / r


    # other things of interest
    pshare_sec1 = equity_price_sec1 / (tot_wealth - Bh)
    pshare_sec2 = equity_price_sec2 / (tot_wealth - Bh)
    pshare_sec3 = equity_price_sec3 / (tot_wealth - Bh)


    err20 = 1 / mup - mc

    # residual function
    def res(x):
        beta_loc, vphi_loc1, vphi_loc2, vphi_loc3, chi1_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc1 < 0.001 or vphi_loc2 < 0.001 or vphi_loc3 < 0.001 or chi1_loc < 0.5:
            raise ValueError('Clearly invalid inputs')
        out1 = household_inc1.ss(Va1=Va1, Vb1=Vb1, Pi=Pi, a1_grid=a_grid, b1_grid=b_grid, N = N1,
                                 tax=tax, w1 = w_occ1, w2 = w_occ2, w3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh1, m = m1)

        out2 = household_inc2.ss(Va2=Va2, Vb2=Vb2, Pi=Pi, a2_grid=a_grid, b2_grid=b_grid, N = N2,
                                 tax=tax, w1 = w_occ1, w2 = w_occ2, w3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh2, m = m2)

        out3 = household_inc3.ss(Va3=Va3, Vb3=Vb3, Pi=Pi, a3_grid=a_grid, b3_grid=b_grid, N = N3,
                                 tax=tax, w1=w_occ1, w2=w_occ2, w3=w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh3, m = m3)

        asset_mkt = out1['A1'] + out2['A2'] + out3['A3'] + out1['B1'] + out2['B2'] + out3['B3'] - equity_price - Bg
        intratemp_hh1 = vphi_loc1 * (labor_hours1 / m1 / gamma_hh1[occupation1]) ** (1/frisch) - (1 - tax) * wage1 * out1['U1'] * gamma_hh1[occupation1] * m1  # comment: changed (multiplied by gamma and m)
        intratemp_hh2 = vphi_loc2 * (labor_hours2 / m2 / gamma_hh2[occupation2]) ** (1/frisch) - (1 - tax) * wage2 * out2['U2'] * gamma_hh2[occupation2] * m2  # comment: changed (multiplied by gamma and m)
        intratemp_hh3 = vphi_loc3 * (labor_hours3 / m3 / gamma_hh3[occupation3]) ** (1/frisch) - (1 - tax) * wage3 * out3['U3'] * gamma_hh3[occupation3] * m3  # comment: changed (multiplied by gamma and m)

        #labor_mk1 = N_occ1 - labor_hours1 * (occupation1 == 0) - labor_hours2 * (occupation2 == 0) - labor_hours3 * (occupation3 == 0)
        #labor_mk2 = N_occ2 - labor_hours1 * (occupation1 == 1) - labor_hours2 * (occupation2 == 1) - labor_hours3 * (occupation3 == 1)
        #labor_mk3 = N_occ3 - labor_hours1 * (occupation1 == 2) - labor_hours2 * (occupation2 == 2) - labor_hours3 * (occupation3 == 3)


        return np.array([asset_mkt, intratemp_hh1, intratemp_hh2, intratemp_hh3, out1['B1'] + out2['B2'] + out3['B3'] - Bh])

    # solve for beta, vphi, omega
    (beta, vphi1, vphi2, vphi3, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess, vphi_guess,
                                                                                                       vphi_guess, chi1_guess]), noisy=noisy)

    labor_mk1 = N_occ1 - labor_hours1 * (occupation1 == 0) - labor_hours2 * (occupation2 == 0) - labor_hours3 * (
                occupation3 == 0)
    labor_mk2 = N_occ2 - labor_hours1 * (occupation1 == 1) - labor_hours2 * (occupation2 == 1) - labor_hours3 * (occupation3 == 1)
    labor_mk3 = N_occ3 - labor_hours1 * (occupation1 == 2) - labor_hours2 * (occupation2 == 2) - labor_hours3 * (occupation3 == 3)

    # extra evaluation to report variables
    ss1 = household_inc1.ss(Va1=Va1, Vb1=Vb1, Pi=Pi, a1_grid=a_grid, b1_grid=b_grid,
                                 tax=tax, w1 = w_occ1, w2 = w_occ2, w3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh1, m = m1, N = N1)
    ss2 = household_inc2.ss(Va2=Va2, Vb2=Vb2, Pi=Pi, a2_grid=a_grid, b2_grid=b_grid,
                                 tax=tax, w1 = w_occ1, w2 = w_occ2, w3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh2, m = m2, N = N2)

    ss3 = household_inc3.ss(Va3=Va3, Vb3=Vb3, Pi=Pi, a3_grid=a_grid, b3_grid=b_grid,
                            tax=tax, w1=w_occ1, w2=w_occ2, w3=w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                            eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh3, m = m3, N = N3)



    # calculate aggregate adjustment cost and check Walras's law
    chi_hh1 = Psi_fun(ss1['a1'], a_grid, r, chi0, chi1, chi2)
    chi_hh2 = Psi_fun(ss2['a2'], a_grid, r, chi0, chi1, chi2)
    chi_hh3 = Psi_fun(ss3['a3'], a_grid, r, chi0, chi1, chi2)
    Chi1 = np.vdot(ss1['D'], chi_hh1)
    Chi2 = np.vdot(ss2['D'], chi_hh2)
    Chi3 = np.vdot(ss3['D'], chi_hh3)
    goods_mkt = ss1['C1'] + ss2['C2'] + ss3['C3'] + I + G + Chi1 + Chi2 + Chi3 + omega * (ss1['B1'] + ss2['B2'] + ss3['B3']) - Y
#    assert np.abs(goods_mkt) < 1E-7

    ss = ss1


    ss.update({'pi': 0, 'piw': 0, 'Q': Q, 'Y': Y, 'mc': mc, 'K': K, 'I': I, 'tax': tax,
               'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi1 + Chi2 + Chi3, 'chi': chi_hh1 + chi_hh2 + chi_hh3, 'phi': phi,
               'beta': beta, 'vphi_1': vphi1, 'vphi_2': vphi2, 'vphi_3': vphi3, 'omega': omega, 'delta': delta, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z1_grid + z2_grid + z3_grid, 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'rstar': r, 'i': r, 'w': w,
               'p': p, 'mup': mup, 'eta': eta, 'ra': ra, 'rb': rb,

                'C': ss1['C1'] + ss2['C2'] + ss3['C3'], 'A': ss1['A1'] + ss2['A2'] + ss3['A3'], 'B': ss1['B1'] + ss2['B2'] + ss3['B3'], 'U': ss1['U1'] + ss2['U2'] + ss3['U3'],

               'Vb2': ss2['Vb2'], 'Va2': ss2['Va2'], 'b2_grid': ss2['b2_grid'], 'A2': ss2['A2'], 'B2': ss2['B2'], 'U2': ss2['U2'],
               'a2_grid': ss2['a2_grid'], 'b2': ss2['b2'], 'a2': ss2['a2'], 'c2': ss2['c2'], 'u2': ss2['u2'], 'C2': ss2['C2'],

               'Vb3': ss3['Vb3'], 'Va3': ss3['Va3'], 'b3_grid': ss3['b3_grid'], 'A3': ss3['A3'], 'B3': ss3['B3'],
               'U3': ss3['U3'], 'a3_grid': ss3['a3_grid'], 'b3': ss3['b3'], 'a3': ss3['a3'], 'c3': ss3['c3'], 'u3': ss3['u3'], 'C3': ss3['C3'],

               'K_sec_1': K_sec1, 'K_sec_2': K_sec2, 'K_sec_3': K_sec3,
               'Y_sec_1': Y_sec1, 'Y_sec_2': Y_sec2, 'Y_sec_3': Y_sec3,
               'N_sec_1': N_sec1, 'N_sec_2': N_sec2, 'N_sec_3': N_sec3, 'N': N,
               'gamma_hh_1_1': gamma_hh_occ11, 'gamma_hh_1_2': gamma_hh_occ12, 'gamma_hh_1_3': gamma_hh_occ13,
               'gamma_hh_2_1': gamma_hh_occ21, 'gamma_hh_2_2': gamma_hh_occ22, 'gamma_hh_2_3': gamma_hh_occ23,
               'gamma_hh_3_1': gamma_hh_occ31, 'gamma_hh_3_2': gamma_hh_occ32, 'gamma_hh_3_3': gamma_hh_occ33,
               'w_occ_1': w_occ1, 'w_occ_2': w_occ2, 'w_occ_3': w_occ3,
               'w_sec_1': w_sec1, 'w_sec_2': w_sec2, 'w_sec_3': w_sec3,
               'I_sec_1': I_sec1, 'I_sec_2': I_sec2, 'I_sec_3': I_sec3,
               'productivity_sec_1': productivity_sec1, 'productivity_sec_2': productivity_sec2, 'productivity_sec_3': productivity_sec3,
                'p_sec_1': p_sec1, 'p_sec_2': p_sec2, 'p_sec_3': p_sec3,
               'div_sec_1': div_sec1, 'div_sec_2': div_sec2, 'div_sec_3': div_sec3, 'div': div,
               'Q_sec_1': Q_sec1, 'Q_sec_2': Q_sec2, 'Q_sec_3': Q_sec3,
               'N_occ_sec_1_1': N_sec_occ11, 'N_occ_sec_1_2': N_sec_occ21, 'N_occ_sec_1_3': N_sec_occ31,
               'N_occ_sec_2_1': N_sec_occ12, 'N_occ_sec_2_2': N_sec_occ22, 'N_occ_sec_2_3': N_sec_occ32,
               'N_occ_sec_3_1': N_sec_occ13, 'N_occ_sec_3_2': N_sec_occ23, 'N_occ_sec_3_3': N_sec_occ33,

               'sigma_sec_1': sigma_sec1, 'sigma_sec_2': sigma_sec2, 'sigma_sec_3': sigma_sec3,
               'psip_sec_1': 0, 'psip_sec_2': 0, 'psip_sec_3': 0, 'psip': 0,
               "mc_sec_1": mc_sec1, "mc_sec_2": mc_sec2, "mc_sec_3": mc_sec3,
               "nu_sec_1": nu_sec1, "nu_sec_2": nu_sec2, "nu_sec_3": nu_sec3,
               'equity_price_sec_1': equity_price_sec1, 'equity_price_sec_2': equity_price_sec2, 'equity_price_sec_3': equity_price_sec3,

               'alpha_occ_sec_1_1': alpha_sec_occ11, 'alpha_occ_sec_1_2': alpha_sec_occ21, 'alpha_occ_sec_1_3': alpha_sec_occ31,
               'alpha_occ_sec_2_1': alpha_sec_occ12, 'alpha_occ_sec_2_2': alpha_sec_occ22, 'alpha_occ_sec_2_3': alpha_sec_occ32,
               'alpha_occ_sec_3_1': alpha_sec_occ13, 'alpha_occ_sec_3_2': alpha_sec_occ23, 'alpha_occ_sec_3_3': alpha_sec_occ33,

               'pshare_1': pshare_sec1, 'pshare_2': pshare_sec2, 'pshare_3': pshare_sec3, 'pshare': pshare,
               'N_occ_1': N_occ1, 'N_occ_2': N_occ2, 'N_occ_3': N_occ3,

               'L_sec_1': L_sec1, 'L_sec_2': L_sec2, 'L_sec_3': L_sec3,
               'm1': m1, 'm2': m2, 'm3': m3,

               'N_hh_occ_1_1': N_hh_occ11, 'N_hh_occ_1_2': N_hh_occ12, 'N_hh_occ_1_3': N_hh_occ13,
               'N_hh_occ_2_1': N_hh_occ21, 'N_hh_occ_2_2': N_hh_occ22, 'N_hh_occ_2_3': N_hh_occ23,
               'N_hh_occ_3_1': N_hh_occ31, 'N_hh_occ_3_2': N_hh_occ32, 'N_hh_occ_3_3': N_hh_occ33,

               })
    return ss

ss = hank_ss()