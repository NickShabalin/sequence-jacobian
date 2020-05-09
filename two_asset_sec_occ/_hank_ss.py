import numpy as np

import utils
from ._households import household1, household2, household3
from ._incomes import income1, income
from ._labor_demand_calculation import LaborDemandCalculation
from ._misc import Psi_fun

household_inc1 = household1.attach_hetinput(income)
household_inc2 = household2.attach_hetinput(income)
household_inc3 = household3.attach_hetinput(income)

def hank_ss(amax=4000,
            Bg=2.8 / 2,
            Bh=1.04 / 2,
            bmax=50,
            chi0=0.25,
            chi1_guess=6.5,
            chi2=2,
            delta=0.02,
            eis=0.5,
            epsI=4,
            frisch=1,
            G=0.2 / 2,
            kappap=0.1,
            kappaw=0.1,
            kmax=1,
            muw=1.1,
            nA=70,
            nB=50,
            nK=50,
            nZ=3,
            omega=0.005,
            phi=1.5,
            r=0.0125,
            rho_z=0.966,
            sigma_z=0.92,
            tot_wealth=14,
            vphi_guess=2.07,
            beta_guess=0.976,
            noisy=True):
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

    gamma_hh_occ21 = 0.278417140245438
    gamma_hh_occ22 = 1
    gamma_hh_occ23 = 0.357559561729431

    gamma_hh_occ31 = 0.257629066705704
    gamma_hh_occ32 = 0.32952556014061
    gamma_hh_occ33 = 1

    gamma_hh = [[gamma_hh_occ11, gamma_hh_occ12, gamma_hh_occ13],
                [gamma_hh_occ21, gamma_hh_occ22, gamma_hh_occ23],
                [gamma_hh_occ31, gamma_hh_occ32, gamma_hh_occ33]]

    p = 1
    '''
    w_occ1 = 1
    w_occ2 = 1.944502
    w_occ3 = 1.563125
    '''

    w_occ     = [1.1, 2.044502, 1.663125]
    sigma_sec = [0.2, 0.2, 0.2]

    eta = (1 - mup) / mup

    # new values will be here
    Y_sec = [0.260986566543579, 0.343330949544907, 0.398539662361145]
    Y = (Y_sec[0] ** ((eta - 1) / eta) + Y_sec[1] ** ((eta - 1) / eta) + Y_sec[2] ** ((eta - 1) / eta)) ** (eta / (eta - 1))

    nu_sec = [0.425027757883072, 0.538959443569183, 0.273549377918243]
    Q_sec  = [1, 1, 1]

    N_sec_occ = np.zeros((3,3))
    N_sec_occ[0][0] = 0.66251665353775
    N_sec_occ[0][1] = 0.182436376810074
    N_sec_occ[0][2] = 0.155046954751015
    N_sec_occ[1][0] = 0.372738629579544
    N_sec_occ[1][1] = 0.19160558283329
    N_sec_occ[1][2] = 0.435655802488327
    N_sec_occ[2][0] = 0.255537301301956
    N_sec_occ[2][1] = 0.227060705423355
    N_sec_occ[2][2] = 0.517401993274688

    alpha_sec_occ = np.zeros((3,3))

    from itertools import product

    for i, j in product(range(3), repeat=2):
        alpha_sec_occ[i][j] =  (N_sec_occ[i][j] ** (1 - sigma_sec[i]) * w_occ[j]) / (
                N_sec_occ[i][0] ** (1 - sigma_sec[i]) * w_occ[0] +
                N_sec_occ[i][1] ** (1 - sigma_sec[i]) * w_occ[1] +
                N_sec_occ[i][2] ** (1 - sigma_sec[i]) * w_occ[2])

    p_sec = [ (Y / Y_sec[i]) ** (1 / eta) * p for i in range(3)]

    N_sec_occ = LaborDemandCalculation(alpha_sec_occ, sigma_sec, p_sec, Y_sec, nu_sec, w_occ).out()

    L_sec = np.zeros((3,))

    for i in range(3):
        L_sec[i] = (alpha_sec_occ[i][0] * N_sec_occ[i][0] ** sigma_sec[i] +
                    alpha_sec_occ[i][1] * N_sec_occ[i][1] ** sigma_sec[i] +
                    alpha_sec_occ[i][2] * N_sec_occ[i][2] ** sigma_sec[i]) ** (1 / sigma_sec[i])


    K_sec = np.zeros((3,))

    for i in range(3):
        K_sec[i] = 1 / (1 + r * Q_sec[i]) * nu_sec[i] / (1 - nu_sec[i]) * L_sec[i] ** sigma_sec[i] * (
                w_occ[0] * N_sec_occ[i][0] ** (1 - sigma_sec[i]) / alpha_sec_occ[i][0] +
                w_occ[1] * N_sec_occ[i][1] ** (1 - sigma_sec[i]) / alpha_sec_occ[i][1] +
                w_occ[2] * N_sec_occ[i][2] ** (1 - sigma_sec[i]) / alpha_sec_occ[i][2])


    mc_sec = [ (r * Q_sec[i] + delta) * K_sec[i] / nu_sec[i] / Y_sec[i] for i in range(3)]
    mc = sum(mc_sec)

    m = [0.33, 0.33, 0.33]

    Q = 1

    ra = r
    rb = r - omega

    K = sum(K_sec)

    I_sec = [delta * K_sec[i] for i in range(3)]

    I = delta * K

    productivity_sec = [Y_sec[i] * K_sec[i] ** (-nu_sec[i]) * L_sec[i] ** (nu_sec[i] - 1) for i in range(3)]
    N_sec = [sum(N_sec_occ[i]) for i in range(3)]
    N_occ = [N_sec_occ[0][i] + N_sec_occ[1][i] + N_sec_occ[2][i] for i in range(3)]
    w_sec = [(w_occ[0] * N_sec_occ[i][0] + w_occ[1] * N_sec_occ[i][1] + w_occ[2] * N_sec_occ[i][2]) / N_sec[i] for i in range(3)]
    N_sum = sum(N_sec)

    w = (w_sec[0] * N_sec[0] + w_sec[1] * N_sec[1] + w_sec[2] * N_sec[2]) / N_sum

    tax = (r * Bg + G) / w / N_sum
    div = p * Y - w * N_sum - I
    equity_price = div / r
    pshare = equity_price / (tot_wealth - Bh)

    N = [0.33, 0.33, 0.33]

    N_hh_occ = np.zeros((3,3))

    for i in range(3):
        N_hh_occ[i] = income1(w_occ[0], w_occ[1], w_occ[2], gamma_hh[i], m[i], N[i])

    z_grid = [income(e_grid, tax, w_occ[0], w_occ[1], w_occ[2], gamma_hh[i], m[i], N[i]) for i in range(3)]

    Va = [None] * 3
    Vb = [None] * 3

    for i in range(3):
        Va[i] = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid[i].shape[0], 1, 1))
        Vb[i] = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid[i].shape[0], 1, 1))

    wages = w_occ

    labor_hours_hh = [N_hh_occ[i] for i in range(3)]

    occupation = [0,1,2]

    wage1 = wages[occupation[0]]
    labor_hours1 = labor_hours_hh[0][occupation[0]]
    wage2 = wages[occupation[1]]
    labor_hours2 = labor_hours_hh[1][occupation[1]]
    wage3 = wages[occupation[2]]
    labor_hours3 = labor_hours_hh[2][occupation[2]]

    '''
    mc_sec1 = w_sec1 * (N_sec_occ11 + N_sec_occ12 + N_sec_occ13) / (1 - nu_sec1) / Y_sec1
    mc_sec2 = w_sec2 * (N_sec_occ21 + N_sec_occ22 + N_sec_occ23) / (1 - nu_sec2) / Y_sec2
    mc_sec3 = w_sec3 * (N_sec_occ31 + N_sec_occ32 + N_sec_occ33) / (1 - nu_sec3) / Y_sec3
    mc = mc_sec1 + mc_sec2 + mc_sec3
    '''

    div_sec = [Y_sec[i] * p_sec[i] - w_sec[i] * N_sec[i] - I_sec[i] for i in range(3)]

    #err5 = div - div_sec1 - div_sec2 - div_sec3

    equity_price_sec = [div_sec[i] / r for i in range(3)]

    # other things of interest
    pshare_sec = [equity_price_sec[i] / (tot_wealth - Bh) for i in range(3)]


    # residual function
    def res(x):
        beta_loc, vphi_loc1, vphi_loc2, vphi_loc3, chi1_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc1 < 0.001 or vphi_loc2 < 0.001 or vphi_loc3 < 0.001 or chi1_loc < 0.5:
            raise ValueError('Clearly invalid inputs')


        out1 = household_inc1.ss(Va1=Va[0], Vb1=Vb[0], Pi=Pi, a1_grid=a_grid, b1_grid=b_grid, N = N[0],
                                 tax=tax, w1 = w_occ[0], w2 = w_occ[1], w3 = w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh[0], m = m[0])

        out2 = household_inc2.ss(Va2=Va[1], Vb2=Vb[1], Pi=Pi, a2_grid=a_grid, b2_grid=b_grid, N = N[1],
                                 tax=tax, w1 = w_occ[0], w2 = w_occ[1], w3 = w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh[1], m = m[1])

        out3 = household_inc3.ss(Va3=Va[2], Vb3=Vb[2], Pi=Pi, a3_grid=a_grid, b3_grid=b_grid, N = N[2],
                                 tax=tax, w1=w_occ[0], w2=w_occ[1], w3=w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = gamma_hh[2], m = m[2])

        asset_mkt = out1['A1'] + out2['A2'] + out3['A3'] + out1['B1'] + out2['B2'] + out3['B3'] - equity_price - Bg
        intratemp_hh1 = vphi_loc1 * (labor_hours1 / m[0] / gamma_hh[0][occupation[0]]) ** (1/frisch) - muw * (1 - tax) * wage1 * out1['U1'] * gamma_hh[0][occupation[0]] * m[0]  # comment: changed (multiplied by gamma and m)
        intratemp_hh2 = vphi_loc2 * (labor_hours2 / m[1] / gamma_hh[1][occupation[1]]) ** (1/frisch) - muw * (1 - tax) * wage2 * out2['U2'] * gamma_hh[1][occupation[1]] * m[1]  # comment: changed (multiplied by gamma and m)
        intratemp_hh3 = vphi_loc3 * (labor_hours3 / m[2] / gamma_hh[2][occupation[2]]) ** (1/frisch) - muw * (1 - tax) * wage3 * out3['U3'] * gamma_hh[2][occupation[2]] * m[2]  # comment: changed (multiplied by gamma and m)

        #labor_mk1 = N_occ1 - labor_hours1 * (occupation1 == 0) - labor_hours2 * (occupation2 == 0) - labor_hours3 * (occupation3 == 0)
        #labor_mk2 = N_occ2 - labor_hours1 * (occupation1 == 1) - labor_hours2 * (occupation2 == 1) - labor_hours3 * (occupation3 == 1)
        #labor_mk3 = N_occ3 - labor_hours1 * (occupation1 == 2) - labor_hours2 * (occupation2 == 2) - labor_hours3 * (occupation3 == 3)


        return np.array([asset_mkt, intratemp_hh1, intratemp_hh2, intratemp_hh3, out1['B1'] + out2['B2'] + out3['B3'] - Bh])

    # solve for beta, vphi, omega
    (beta, vphi1, vphi2, vphi3, chi1), _ = utils.broyden_solver(res, np.array([beta_guess,
                                                                               vphi_guess,
                                                                               vphi_guess,
                                                                               vphi_guess,
                                                                               chi1_guess]), noisy=noisy)

    labor_mk = [None, None, None]
    for i in range(3):
        labor_mk[i] = N_occ[i] - labor_hours1 * (occupation[0] == i) - labor_hours2 * (occupation[1] == i) - labor_hours3 * (
                occupation[2] == i)

    # extra evaluation to report variables
    ss1 = household_inc1.ss(Va1=Va[0], Vb1=Vb[0], Pi=Pi, a1_grid=a_grid, b1_grid=b_grid,
                            tax=tax, w1 = w_occ[0], w2 = w_occ[1], w3 = w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta,
                            eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh[0], m = m[0], N = N[0])
    ss2 = household_inc2.ss(Va2=Va[1], Vb2=Vb[1], Pi=Pi, a2_grid=a_grid, b2_grid=b_grid,
                            tax=tax, w1 = w_occ[0], w2 = w_occ[1], w3 = w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta,
                            eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh[1], m = m[1], N = N[1])

    ss3 = household_inc3.ss(Va3=Va[2], Vb3=Vb[2], Pi=Pi, a3_grid=a_grid, b3_grid=b_grid,
                            tax=tax, w1=w_occ[0], w2=w_occ[1], w3=w_occ[2], e_grid=e_grid, k_grid=k_grid, beta=beta,
                            eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = gamma_hh[2], m = m[2], N = N[2])



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
               'beta': beta, 'vphi': (vphi1 * vphi2 * vphi3) ** (1 / 3), 'omega': omega, 'delta': delta, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid,
               'z_grid': sum(z_grid), 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'rstar': r, 'i': r, 'w': w,
               'p': p, 'mup': mup, 'eta': eta, 'ra': ra, 'rb': rb,

               'C': ss1['C1'] + ss2['C2'] + ss3['C3'], 'A': ss1['A1'] + ss2['A2'] + ss3['A3'],
               'B': ss1['B1'] + ss2['B2'] + ss3['B3'], 'U': ss1['U1'] + ss2['U2'] + ss3['U3'],

               'Vb2': ss2['Vb2'], 'Va2': ss2['Va2'], 'b2_grid': ss2['b2_grid'], 'A2': ss2['A2'], 'B2': ss2['B2'],
               'U2': ss2['U2'],
               'a2_grid': ss2['a2_grid'], 'b2': ss2['b2'], 'a2': ss2['a2'], 'c2': ss2['c2'], 'u2': ss2['u2'],
               'C2': ss2['C2'],

               'Vb3': ss3['Vb3'], 'Va3': ss3['Va3'], 'b3_grid': ss3['b3_grid'], 'A3': ss3['A3'], 'B3': ss3['B3'],
               'U3': ss3['U3'], 'a3_grid': ss3['a3_grid'], 'b3': ss3['b3'], 'a3': ss3['a3'], 'c3': ss3['c3'],
               'u3': ss3['u3'], 'C3': ss3['C3'],

               'K_sec_1': K_sec[0], 'K_sec_2': K_sec[1], 'K_sec_3': K_sec[2],
               'Y_sec_1': Y_sec[0], 'Y_sec_2': Y_sec[1], 'Y_sec_3': Y_sec[2],
               'N_sec_1': N_sec[0], 'N_sec_2': N_sec[1], 'N_sec_3': N_sec[2],
               'N': N_sum,
               'gamma_hh_1_1': gamma_hh_occ11, 'gamma_hh_1_2': gamma_hh_occ12, 'gamma_hh_1_3': gamma_hh_occ13,
               'gamma_hh_2_1': gamma_hh_occ21, 'gamma_hh_2_2': gamma_hh_occ22, 'gamma_hh_2_3': gamma_hh_occ23,
               'gamma_hh_3_1': gamma_hh_occ31, 'gamma_hh_3_2': gamma_hh_occ32, 'gamma_hh_3_3': gamma_hh_occ33,
               'w_occ_1': w_occ[0], 'w_occ_2': w_occ[1], 'w_occ_3': w_occ[2],
               'w_sec_1': w_sec[0], 'w_sec_2': w_sec[1], 'w_sec_3': w_sec[2],
               'I_sec_1': I_sec[0], 'I_sec_2': I_sec[1], 'I_sec_3': I_sec[2],
               'productivity_sec_1': productivity_sec[0],
               'productivity_sec_2': productivity_sec[1],
               'productivity_sec_3': productivity_sec[2],
               'p_sec_1': p_sec[0], 'p_sec_2': p_sec[1], 'p_sec_3': p_sec[2],
               'div_sec_1': div_sec[0], 'div_sec_2': div_sec[1], 'div_sec_3': div_sec[2],
               'div': div,
               'Q_sec_1': Q_sec[0], 'Q_sec_2': Q_sec[1], 'Q_sec_3': Q_sec[2],
               'N_occ_sec_1_1': N_sec_occ[0][0], 'N_occ_sec_1_2': N_sec_occ[1][0], 'N_occ_sec_1_3': N_sec_occ[2][0],
               'N_occ_sec_2_1': N_sec_occ[0][1], 'N_occ_sec_2_2': N_sec_occ[1][1], 'N_occ_sec_2_3': N_sec_occ[2][1],
               'N_occ_sec_3_1': N_sec_occ[0][2], 'N_occ_sec_3_2': N_sec_occ[1][2], 'N_occ_sec_3_3': N_sec_occ[2][2],

               'sigma_sec_1': sigma_sec[0], 'sigma_sec_2': sigma_sec[0], 'sigma_sec_3': sigma_sec[0],
               'psip_sec_1': 0, 'psip_sec_2': 0, 'psip_sec_3': 0, 'psip': 0,
               "mc_sec_1": mc_sec[0], "mc_sec_2": mc_sec[1], "mc_sec_3": mc_sec[2],
               "nu_sec_1": nu_sec[0], "nu_sec_2": nu_sec[1], "nu_sec_3": nu_sec[2],
               'equity_price_sec_1': equity_price_sec[0],
               'equity_price_sec_2': equity_price_sec[1],
               'equity_price_sec_3': equity_price_sec[2],
               'alpha_occ_sec_1_1': alpha_sec_occ[0][0],
               'alpha_occ_sec_1_2': alpha_sec_occ[1][0],
               'alpha_occ_sec_1_3': alpha_sec_occ[2][0],
               'alpha_occ_sec_2_1': alpha_sec_occ[0][1],
               'alpha_occ_sec_2_2': alpha_sec_occ[1][1],
               'alpha_occ_sec_2_3': alpha_sec_occ[2][1],
               'alpha_occ_sec_3_1': alpha_sec_occ[0][2],
               'alpha_occ_sec_3_2': alpha_sec_occ[1][2],
               'alpha_occ_sec_3_3': alpha_sec_occ[2][2],

               'pshare_1': pshare_sec[0],
               'pshare_2': pshare_sec[1],
               'pshare_3': pshare_sec[2],
               'pshare': pshare,
               'N_occ_1': N_occ[0], 'N_occ_2': N_occ[1], 'N_occ_3': N_occ[2],
               'L_sec_1': L_sec[0], 'L_sec_2': L_sec[1], 'L_sec_3': L_sec[2],
               'm1': m[0], 'm2': m[1], 'm3': m[2],

               'N_hh_occ_1_1': N_hh_occ[0][0], 'N_hh_occ_1_2': N_hh_occ[0][1], 'N_hh_occ_1_3': N_hh_occ[0,2],
               'N_hh_occ_2_1': N_hh_occ[1][0], 'N_hh_occ_2_2': N_hh_occ[1][1], 'N_hh_occ_2_3': N_hh_occ[1][2],
               'N_hh_occ_3_1': N_hh_occ[2][0], 'N_hh_occ_3_2': N_hh_occ[2][1], 'N_hh_occ_3_3': N_hh_occ[2][2]})
    return ss




