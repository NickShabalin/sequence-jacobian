import utils
import numpy as np
from household_blocks import household_inc1, household_inc2, household_inc3, \
                                income1, income2, income3, income_hh_1, income_hh_2, income_hh_3, Psi_fun
import test_wage_targeting_v3

def hank_ss(beta_guess=0.976, vphi_guess=2.07, chi1_guess=6.5, r=0.0125, delta=0.02, kappap=0.1,
            muw=1.1, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
            phi=1.5, nZ=3, nB=50*2, nA=70*2, nK=50*2, bmax=50*2, amax=4000*2, kmax=1*2, rho_z=0.966, sigma_z=0.92,
            tot_wealth=14*2, Bh=1.04*2, Bg=2.8*2, G=0.2*2, noisy=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)
    k_grid = utils.agrid(amax=kmax, n=nK)
    e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)


    # solve analytically what we can
    mup = 1.01
    #mup = 1.009861
    #mup = 1.00648
    #mup = 1.015
    #mup = 1.00985
    #mup = 1.5 # to get A + B = 14Y
    #eta = (1 - mup) / mup
    eta = mup / (mup - 1)

    gamma_hh_occ11 = 1.0
    gamma_hh_occ12 = 0.447435140609741
    gamma_hh_occ13 = 0.464943289756775
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


    sigma_sec1 = sigma_sec2 = sigma_sec3 = 0.2


    # new values will be here
    Y_sec1 = 0.260986566543579*2
    Y_sec2 = 0.343330949544907*2
    Y_sec3 = 0.398539662361145*2

    Y = (Y_sec1 ** ((eta - 1) / eta) + Y_sec2 ** ((eta - 1) / eta) + Y_sec3 ** ((eta - 1) / eta)) ** (eta / (eta - 1))

    nu_sec1 = 0.425027757883072
    nu_sec2 = 0.538959443569183
    nu_sec3 = 0.273549377918243

    Q_sec1 = 1
    Q_sec2 = 1
    Q_sec3 = 1

    N1 = N2 = N3 = 1*2
    '''
    m1 = 0.33
    m2 = 0.32
    m3 = 0.728
    '''

    N_hh_eff_1 = N1 * gamma_hh_occ11
    N_hh_eff_2 = N2 * gamma_hh_occ22
    N_hh_eff_3 = N3 * gamma_hh_occ33


    p_sec1 = (Y / Y_sec1) ** (1 / eta) * p
    p_sec2 = (Y / Y_sec2) ** (1 / eta) * p
    p_sec3 = (Y / Y_sec3) ** (1 / eta) * p

    #err1 = p - (p_sec1 ** (1 - eta) + p_sec2 ** (1 - eta) + p_sec3 ** (1 - eta)) ** (1 / (1 - eta))

    N_sec_occ11 = 0.66251665353775*2
    N_sec_occ12 = 0.18243637681074*2
    N_sec_occ13 = 0.155046954751015*2
    N_sec_occ21 = 0.372738629579544*2
    N_sec_occ22 = 0.19160558283329*2
    N_sec_occ23 = 0.435655802488327*2
    N_sec_occ31 = 0.255537301301956*2
    N_sec_occ32 = 0.227060705423355*2
    N_sec_occ33 = 0.517401993274688*2


    w_occ1, w_occ2, w_occ3, N_sec1, N_sec2, N_sec3, m1, m2, m3 = test_wage_targeting_v3.out(sigma_sec1, sigma_sec2, sigma_sec3,
                                                                                            p_sec1, p_sec2, p_sec3,
                                                                                            Y_sec1, Y_sec2, Y_sec3,
                                                                                            nu_sec1, nu_sec2, nu_sec3,
                                                                                            N_sec_occ11, N_sec_occ12, N_sec_occ13,
                                                                                            N_sec_occ21, N_sec_occ22, N_sec_occ23,
                                                                                            N_sec_occ31, N_sec_occ32, N_sec_occ33,
                                                                                            N_hh_eff_1, N_hh_eff_2, N_hh_eff_3)



    N_sec_occ11 *= N_sec1
    N_sec_occ12 *= N_sec1
    N_sec_occ13 *= N_sec1
    N_sec_occ21 *= N_sec2
    N_sec_occ22 *= N_sec2
    N_sec_occ23 *= N_sec2
    N_sec_occ31 *= N_sec3
    N_sec_occ32 *= N_sec3
    N_sec_occ33 *= N_sec3


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

    '''
    K_sec1 = nu_sec1 * Y_sec1 / (r + delta) * p_sec1 / p
    K_sec2 = nu_sec2 * Y_sec2 / (r + delta) * p_sec2 / p
    K_sec3 = nu_sec3 * Y_sec3 / (r + delta) * p_sec3 / p
    '''
    '''
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
    '''

    #mc_sec1 = (r * Q_sec1 + delta) * K_sec1 / nu_sec1 / Y_sec1
    #mc_sec2 = (r * Q_sec2 + delta) * K_sec2 / nu_sec2 / Y_sec2
    #mc_sec3 = (r * Q_sec3 + delta) * K_sec3 / nu_sec3 / Y_sec3

    #mc = mc_sec1 + mc_sec2 + mc_sec3
    mc = p


    mc_sec1 = 1 / mup * p_sec1 / p
    mc_sec2 = 1 / mup * p_sec2 / p
    mc_sec3 = 1 / mup * p_sec3 / p

    K_sec1 = nu_sec1 * Y_sec1 * mc_sec1 / (r * Q_sec1 + delta)
    K_sec2 = nu_sec2 * Y_sec2 * mc_sec2 / (r * Q_sec2 + delta)
    K_sec3 = nu_sec3 * Y_sec3 * mc_sec3 / (r * Q_sec3 + delta)

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

    N_hh_occ11, N_hh_occ12, N_hh_occ13 = income1(N1)
    N_hh_occ21, N_hh_occ22, N_hh_occ23 = income2(N2)
    N_hh_occ31, N_hh_occ32, N_hh_occ33 = income3(N3)

    z1_grid = income_hh_1(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh_occ11, gamma_hh_occ12, gamma_hh_occ13, m1, N_hh_occ11, N_hh_occ12, N_hh_occ13)
    z2_grid = income_hh_2(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh_occ21, gamma_hh_occ22, gamma_hh_occ23, m2, N_hh_occ21, N_hh_occ22, N_hh_occ23)
    z3_grid = income_hh_3(e_grid, tax, w_occ1, w_occ2, w_occ3, gamma_hh_occ31, gamma_hh_occ32, gamma_hh_occ33, m3, N_hh_occ31, N_hh_occ32, N_hh_occ33)

    Va1 = (0.6*2 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z1_grid.shape[0], 1, 1))
    Vb1 = (0.5*2 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z1_grid.shape[0], 1, 1))

    Va2 = (0.6*2 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z2_grid.shape[0], 1, 1))
    Vb2 = (0.5*2 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z2_grid.shape[0], 1, 1))

    Va3 = (0.6*2 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z3_grid.shape[0], 1, 1))
    Vb3 = (0.5*2 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z3_grid.shape[0], 1, 1))

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

    err5 = div - div_sec1 - div_sec2 - div_sec3


    equity_price_sec1 = div_sec1 / r
    equity_price_sec2 = div_sec2 / r
    equity_price_sec3 = div_sec3 / r


    # other things of interest
    pshare_sec1 = equity_price_sec1 / (tot_wealth - Bh)
    pshare_sec2 = equity_price_sec2 / (tot_wealth - Bh)
    pshare_sec3 = equity_price_sec3 / (tot_wealth - Bh)



    # residual function
    def res(x):
        beta_loc, vphi_loc1, vphi_loc2, vphi_loc3, chi1_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc1 < 0.001 or vphi_loc2 < 0.001 or vphi_loc3 < 0.001 or chi1_loc < 0.5:
            raise ValueError('Clearly invalid inputs')
        out1 = household_inc1.ss(Va1=Va1, Vb1=Vb1, Pi=Pi, a1_grid=a_grid, b1_grid=b_grid, N_hh_occ_1_1 = N_hh_occ11, N_hh_occ_1_2 = N_hh_occ12, N_hh_occ_1_3 = N_hh_occ13,
                                 tax=tax, w_occ_1 = w_occ1, w_occ_2 = w_occ2, w_occ_3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma_hh_1_1 = gamma_hh_occ11,
                                 gamma_hh_1_2 = gamma_hh_occ12, gamma_hh_1_3 = gamma_hh_occ13, m1 = m1)

        out2 = household_inc2.ss(Va2=Va2, Vb2=Vb2, Pi=Pi, a2_grid=a_grid, b2_grid=b_grid, N_hh_occ_2_1 = N_hh_occ21, N_hh_occ_2_2 = N_hh_occ22, N_hh_occ_2_3 = N_hh_occ23,
                                 tax=tax, w_occ_1 = w_occ1, w_occ_2 = w_occ2, w_occ_3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma_hh_2_1 = gamma_hh_occ21,
                                 gamma_hh_2_2 = gamma_hh_occ22, gamma_hh_2_3 = gamma_hh_occ23, m2 = m2)

        out3 = household_inc3.ss(Va3=Va3, Vb3=Vb3, Pi=Pi, a3_grid=a_grid, b3_grid=b_grid, N_hh_occ_3_1 = N_hh_occ31, N_hh_occ_3_2 = N_hh_occ32, N_hh_occ_3_3 = N_hh_occ33,
                                 tax=tax, w_occ_1=w_occ1, w_occ_2=w_occ2, w_occ_3=w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta_loc,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma_hh_3_1 = gamma_hh_occ31,
                                 gamma_hh_3_2 = gamma_hh_occ32, gamma_hh_3_3 = gamma_hh_occ33, m3 = m3)

        asset_mkt = out1['A1'] + out2['A2'] + out3['A3'] + out1['B1'] + out2['B2'] + out3['B3'] - equity_price - Bg
        intratemp_hh1 = vphi_loc1 * (labor_hours1 / m1 / gamma_hh1[occupation1]) ** (1/frisch) - (1 - tax) * wage1 * out1['U1'] * gamma_hh1[occupation1] / p  # comment: changed (multiplied by gamma and m)
        intratemp_hh2 = vphi_loc2 * (labor_hours2 / m2 / gamma_hh2[occupation2]) ** (1/frisch) - (1 - tax) * wage2 * out2['U2'] * gamma_hh2[occupation2] / p  # comment: changed (multiplied by gamma and m)
        intratemp_hh3 = vphi_loc3 * (labor_hours3 / m3 / gamma_hh3[occupation3]) ** (1/frisch) - (1 - tax) * wage3 * out3['U3'] * gamma_hh3[occupation3] / p  # comment: changed (multiplied by gamma and m)

        return np.array([asset_mkt, intratemp_hh1, intratemp_hh2, intratemp_hh3, out1['B1'] + out2['B2'] + out3['B3'] - Bh])

    # solve for beta, vphi, omega
    (beta, vphi1, vphi2, vphi3, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess, vphi_guess,
                                                                                                       vphi_guess, chi1_guess]), noisy=noisy)
    # beta = 0.9782212530661712
    # chi1 = 9.214625716147443
    # vphi1 = 6.549901834416055
    # vphi2 = 6.591757183416546
    # vphi3 = 4.271567158955762

    # extra evaluation to report variables
    ss1 = household_inc1.ss(Va1=Va1, Vb1=Vb1, Pi=Pi, a1_grid=a_grid, b1_grid=b_grid, N_hh_occ_1_1 = N_hh_occ11, N_hh_occ_1_2 = N_hh_occ12, N_hh_occ_1_3 = N_hh_occ13,
                                 tax=tax, w_occ_1 = w_occ1, w_occ_2 = w_occ2, w_occ_3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, m1 = m1, gamma_hh_1_1 = gamma_hh_occ11,
                                 gamma_hh_1_2 = gamma_hh_occ12, gamma_hh_1_3 = gamma_hh_occ13)

    ss2 = household_inc2.ss(Va2=Va2, Vb2=Vb2, Pi=Pi, a2_grid=a_grid, b2_grid=b_grid, N_hh_occ_2_1 = N_hh_occ21, N_hh_occ_2_2 = N_hh_occ22, N_hh_occ_2_3 = N_hh_occ23,
                                 tax=tax, w_occ_1 = w_occ1, w_occ_2 = w_occ2, w_occ_3 = w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                                 eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, m2 = m2,
                                 gamma_hh_2_1=gamma_hh_occ21, gamma_hh_2_2=gamma_hh_occ22, gamma_hh_2_3=gamma_hh_occ23)

    ss3 = household_inc3.ss(Va3=Va3, Vb3=Vb3, Pi=Pi, a3_grid=a_grid, b3_grid=b_grid, N_hh_occ_3_1 = N_hh_occ31, N_hh_occ_3_2 = N_hh_occ32, N_hh_occ_3_3 = N_hh_occ33,
                            tax=tax, w_occ_1=w_occ1, w_occ_2=w_occ2, w_occ_3=w_occ3, e_grid=e_grid, k_grid=k_grid, beta=beta,
                            eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, m3 = m3,
                            gamma_hh_3_1=gamma_hh_occ31, gamma_hh_3_2=gamma_hh_occ32, gamma_hh_3_3=gamma_hh_occ33)

    # asset_mkt = ss1['A1'] + ss2['A2'] + ss3['A3'] + ss1['B1'] + ss2['B2'] + ss3['B3'] - equity_price - Bg
    # intratemp_hh1 = vphi1 * (labor_hours1 / m1 / gamma_hh1[occupation1]) ** (1 / frisch) - (1 - tax) * wage1 * ss1[
    #     'U1'] * gamma_hh1[occupation1] / p  # comment: changed (multiplied by gamma and m)
    # intratemp_hh2 = vphi2 * (labor_hours2 / m2 / gamma_hh2[occupation2]) ** (1 / frisch) - (1 - tax) * wage2 * ss2[
    #     'U2'] * gamma_hh2[occupation2] / p  # comment: changed (multiplied by gamma and m)
    # intratemp_hh3 = vphi3 * (labor_hours3 / m3 / gamma_hh3[occupation3]) ** (1 / frisch) - (1 - tax) * wage3 * ss3[
    #     'U3'] * gamma_hh3[occupation3] / p  # comment: changed (multiplied by gamma and m)

    # calculate aggregate adjustment cost and check Walras's law
    chi_hh1 = Psi_fun(ss1['a1'], a_grid, r, chi0, chi1, chi2)
    chi_hh2 = Psi_fun(ss2['a2'], a_grid, r, chi0, chi1, chi2)
    chi_hh3 = Psi_fun(ss3['a3'], a_grid, r, chi0, chi1, chi2)
    Chi1 = np.vdot(ss1['D'], chi_hh1)
    Chi2 = np.vdot(ss2['D'], chi_hh2)
    Chi3 = np.vdot(ss3['D'], chi_hh3)
    goods_mkt = ss1['C1'] + ss2['C2'] + ss3['C3'] + I + G + Chi1 + Chi2 + Chi3 + omega * (ss1['B1'] + ss2['B2'] + ss3['B3']) - Y
#    assert np.abs(goods_mkt) < 1E-7
    err15 = ss1['A1'] + ss2['A2'] + ss3['A3'] + ss1['B1'] + ss2['B2'] + ss3['B3'] - 14 * Y
    err16 = ss1["B1"] + ss2["B2"] + ss3["B3"] - 1.04 * Y
    ss = ss1


    ss.update({'pi': 0, 'piw': 0, 'Q': Q, 'Y': Y, 'mc': mc, 'K': K, 'I': I, 'tax': tax,
               'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi1 + Chi2 + Chi3, 'chi': chi_hh1 + chi_hh2 + chi_hh3, 'phi': phi,
               'beta': beta, 'vphi_1': vphi1, 'vphi_2': vphi2, 'vphi_3': vphi3, 'omega': omega, 'delta': delta, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z1_grid + z2_grid + z3_grid, 'e_grid': e_grid,
               'k_grid': k_grid, 'Pi': Pi, 'kappap': kappap, 'kappaw': kappaw, 'rstar': r, 'i': r, 'w': w,
               'p': p, 'mup': mup, 'eta': eta, 'ra': ra, 'rb': rb, 'beta_sir': 1.5, 'gamma_sir': 0.8, 'covid_shock': 0,
               'susceptible': 1, 'infected': 0, 'recovered': 0,

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
               'equity_price_sec1': equity_price_sec1, 'equity_price_sec2': equity_price_sec2, 'equity_price_sec3': equity_price_sec3,
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

               'possible_occupation_1': 0, 'possible_occupation_2': 1, 'possible_occupation_3': 2,
               'occupation_1': 0, 'occupation_2': 1, 'occupation_3': 2,

               'occupation_mult_for_covid_1': 1.2, 'occupation_mult_for_covid_2': 1, 'occupation_mult_for_covid_3': 2

               })
    return ss
