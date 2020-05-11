from itertools import product

import numpy as np

import utils
from ._households import household1, household2, household3
from ._incomes import income1, income
from ._labor_demand_calculation import LaborDemandCalculation
from ._misc import Psi_fun

household_inc1 = household1.attach_hetinput(income)
household_inc2 = household2.attach_hetinput(income)
household_inc3 = household3.attach_hetinput(income)


# noinspection PyPep8Naming
class SSBuilder:
    def _calc_grid(self):
        self._b_grid = utils.agrid(amax=self._bmax, n=self._nB)
        self._a_grid = utils.agrid(amax=self._amax, n=self._nA)
        self._k_grid = utils.agrid(amax=self._kmax, n=self._nK)
        self._e_grid, self._pi, self._Pi = utils.markov_rouwenhorst(rho=self._rho_z, sigma=self._sigma_z, N=self._nZ)

    def _set_up_gamma_hh(self):
        self._gamma_hh = [[1.000000000000000, 0.20000000000000, 0.300000000000000],
                          [0.278417140245438, 1.00000000000000, 0.357559561729431],
                          [0.257629066705704, 0.32952556014061, 1.000000000000000]]

    def _set_up_nu_sec(self):
        self._nu_sec = [0.425027757883072, 0.538959443569183, 0.273549377918243]

    def _set_up_Q_sec(self):
        self._Q_sec = [1, 1, 1]

    def _set_up_N_sec_occ(self):
        self._N_sec_occ = [[0.662516653537750, 0.182436376810074, 0.155046954751015],
                           [0.372738629579544, 0.191605582833290, 0.435655802488327],
                           [0.255537301301956, 0.227060705423355, 0.517401993274688]]

    def _calc_alpha_sec_occ(self):
        self._alpha_sec_occ = np.zeros((3, 3))
        for i, j in product(range(3), repeat=2):
            self._alpha_sec_occ[i][j] = (
                    self._N_sec_occ[i][j] ** (1 - self._sigma_sec[i]) * self._w_occ[j]) / (
                    self._N_sec_occ[i][0] ** (1 - self._sigma_sec[i]) * self._w_occ[0] +
                    self._N_sec_occ[i][1] ** (1 - self._sigma_sec[i]) * self._w_occ[1] +
                    self._N_sec_occ[i][2] ** (1 - self._sigma_sec[i]) * self._w_occ[2])

    def _calc_L_sec(self):
        self._L_sec = [0, 0, 0]
        for i in range(3):
            self._L_sec[i] = (self._alpha_sec_occ[i][0] * self._N_sec_occ[i][0] ** self._sigma_sec[i] +
                              self._alpha_sec_occ[i][1] * self._N_sec_occ[i][1] ** self._sigma_sec[i] +
                              self._alpha_sec_occ[i][2] * self._N_sec_occ[i][2] ** self._sigma_sec[i]) ** (
                                         1 / self._sigma_sec[i])

    def _calc_K_sec(self):
        K_sec = [0, 0, 0]
        for i in range(3):
            K_sec[i] = 1 / (1 + self._r * self._Q_sec[i]) * self._nu_sec[i] / (1 - self._nu_sec[i]) * self._L_sec[i] ** self._sigma_sec[i] * (
                    self._w_occ[0] * self._N_sec_occ[i][0] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][0] +
                    self._w_occ[1] * self._N_sec_occ[i][1] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][1] +
                    self._w_occ[2] * self._N_sec_occ[i][2] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][2])
        self._K_sec = K_sec

    def _calc_N_hh_occ(self):
        self._N_hh_occ = [income1(self._w_occ, self._gamma_hh[i], self._m[i], self._N[i]) for i in range(3)]

    def hank_ss(self,
                amax=4000,
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
        """
        Solve steady state of full GE model.
        Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for (r, tot_wealth, Bh, K, Y=N=1).
        """

        self._amax = amax
        self._bmax = bmax
        self._kmax = kmax
        self._nA = nA
        self._nB = nB
        self._nK = nK
        self._nZ = nZ
        self._rho_z = rho_z
        self._sigma_z = sigma_z
        self._r = r


        self._calc_grid()

        # solve analytically what we can
        mup = 6.3863129
        # mup = 1.009861
        # mup = 1.00648
        # mup = 1.015
        # mup = 1.00985
        # mup = 1.5 # to get A + B = 14Y

        self._set_up_gamma_hh()

        p = 1

        # w_occ1 = 1
        # w_occ2 = 1.944502
        # w_occ3 = 1.563125

        self._w_occ     = [1.1, 2.044502, 1.663125]
        self._sigma_sec = [0.2, 0.2, 0.2]

        eta = (1 - mup) / mup

        # new values will be here
        Y_sec = [0.260986566543579, 0.343330949544907, 0.398539662361145]
        Y = (Y_sec[0] ** ((eta - 1) / eta) + Y_sec[1] ** ((eta - 1) / eta) + Y_sec[2] ** ((eta - 1) / eta)) ** (eta / (eta - 1))

        self._set_up_nu_sec()
        self._set_up_Q_sec()
        self._set_up_N_sec_occ()
        self._calc_alpha_sec_occ()

        p_sec = [ (Y / Y_sec[i]) ** (1 / eta) * p for i in range(3)]

        self._N_sec_occ = LaborDemandCalculation(self._alpha_sec_occ, self._sigma_sec, p_sec, Y_sec, self._nu_sec, self._w_occ).out()

        self._calc_L_sec()
        self._calc_K_sec()


        mc_sec = [ (r * self._Q_sec[i] + delta) * self._K_sec[i] / self._nu_sec[i] / Y_sec[i] for i in range(3)]
        mc = sum(mc_sec)

        self._m = [0.33, 0.33, 0.33]

        Q = 1

        ra = r
        rb = r - omega

        K = sum(self._K_sec)

        I_sec = [delta * self._K_sec[i] for i in range(3)]

        I = delta * K

        productivity_sec = [Y_sec[i] * self._K_sec[i] ** (-self._nu_sec[i]) * self._L_sec[i] ** (self._nu_sec[i] - 1) for i in range(3)]
        N_sec = [sum(self._N_sec_occ[i]) for i in range(3)]
        N_occ = [self._N_sec_occ[0][i] + self._N_sec_occ[1][i] + self._N_sec_occ[2][i] for i in range(3)]
        w_sec = [(self._w_occ[0] * self._N_sec_occ[i][0] + self._w_occ[1] * self._N_sec_occ[i][1] + self._w_occ[2] * self._N_sec_occ[i][2]) / N_sec[i] for i in range(3)]
        N_sum = sum(N_sec)

        w = (w_sec[0] * N_sec[0] + w_sec[1] * N_sec[1] + w_sec[2] * N_sec[2]) / N_sum

        tax = (r * Bg + G) / w / N_sum
        div = p * Y - w * N_sum - I
        equity_price = div / r
        pshare = equity_price / (tot_wealth - Bh)

        self._N = [0.33, 0.33, 0.33]

        self._calc_N_hh_occ()

        z_grid = [income(self._e_grid, tax, self._w_occ[0], self._w_occ[1], self._w_occ[2], self._gamma_hh[i], self._m[i], self._N[i]) for i in range(3)]

        Va = [None] * 3
        Vb = [None] * 3

        for i in range(3):
            Va[i] = (0.6 + 1.1 * self._b_grid[:, np.newaxis] + self._a_grid) ** (-1 / eis) * np.ones((z_grid[i].shape[0], 1, 1))
            Vb[i] = (0.5 + self._b_grid[:, np.newaxis] + 1.2 * self._a_grid) ** (-1 / eis) * np.ones((z_grid[i].shape[0], 1, 1))

        labor_hours_hh = [self._N_hh_occ[i] for i in range(3)]


        labor_hours1 = labor_hours_hh[0][0]
        labor_hours2 = labor_hours_hh[1][1]
        labor_hours3 = labor_hours_hh[2][2]

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

            out1 = household_inc1.ss(Va1=Va[0], Vb1=Vb[0], Pi=self._Pi, a1_grid=self._a_grid, b1_grid=self._b_grid, N = self._N[0],
                                     tax=tax, w1 = self._w_occ[0], w2 = self._w_occ[1], w3 = self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta_loc,
                                     eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = self._gamma_hh[0], m = self._m[0])

            out2 = household_inc2.ss(Va2=Va[1], Vb2=Vb[1], Pi=self._Pi, a2_grid=self._a_grid, b2_grid=self._b_grid, N = self._N[1],
                                     tax=tax, w1 = self._w_occ[0], w2 = self._w_occ[1], w3 = self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta_loc,
                                     eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = self._gamma_hh[1], m = self._m[1])

            out3 = household_inc3.ss(Va3=Va[2], Vb3=Vb[2], Pi=self._Pi, a3_grid=self._a_grid, b3_grid=self._b_grid, N = self._N[2],
                                     tax=tax, w1=self._w_occ[0], w2=self._w_occ[1], w3=self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta_loc,
                                     eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1_loc, chi2=chi2, gamma = self._gamma_hh[2], m = self._m[2])

            asset_mkt = out1['A1'] + out2['A2'] + out3['A3'] + out1['B1'] + out2['B2'] + out3['B3'] - equity_price - Bg
            intratemp_hh1 = vphi_loc1 * (labor_hours1 / self._m[0] / self._gamma_hh[0][0]) ** (1/frisch) - muw * (1 - tax) * self._w_occ[0] * out1['U1'] * self._gamma_hh[0][0] * self._m[0]  # comment: changed (multiplied by gamma and m)
            intratemp_hh2 = vphi_loc2 * (labor_hours2 / self._m[1] / self._gamma_hh[1][1]) ** (1/frisch) - muw * (1 - tax) * self._w_occ[1] * out2['U2'] * self._gamma_hh[1][1] * self._m[1]  # comment: changed (multiplied by gamma and m)
            intratemp_hh3 = vphi_loc3 * (labor_hours3 / self._m[2] / self._gamma_hh[2][2]) ** (1/frisch) - muw * (1 - tax) * self._w_occ[2] * out3['U3'] * self._gamma_hh[2][2] * self._m[2]  # comment: changed (multiplied by gamma and m)

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
            labor_mk[i] = N_occ[i] - labor_hours1 * (0 == i) - labor_hours2 * (1 == i) - labor_hours3 * (2 == i)

        # extra evaluation to report variables
        ss1 = household_inc1.ss(Va1=Va[0], Vb1=Vb[0], Pi=self._Pi, a1_grid=self._a_grid, b1_grid=self._b_grid,
                                tax=tax, w1 = self._w_occ[0], w2 = self._w_occ[1], w3 = self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta,
                                eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = self._gamma_hh[0], m = self._m[0], N = self._N[0])
        ss2 = household_inc2.ss(Va2=Va[1], Vb2=Vb[1], Pi=self._Pi, a2_grid=self._a_grid, b2_grid=self._b_grid,
                                tax=tax, w1 = self._w_occ[0], w2 = self._w_occ[1], w3 = self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta,
                                eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = self._gamma_hh[1], m = self._m[1], N = self._N[1])

        ss3 = household_inc3.ss(Va3=Va[2], Vb3=Vb[2], Pi=self._Pi, a3_grid=self._a_grid, b3_grid=self._b_grid,
                                tax=tax, w1=self._w_occ[0], w2=self._w_occ[1], w3=self._w_occ[2], e_grid=self._e_grid, k_grid=self._k_grid, beta=beta,
                                eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2, gamma = self._gamma_hh[2], m = self._m[2], N = self._N[2])



        # calculate aggregate adjustment cost and check Walras's law
        chi_hh1 = Psi_fun(ss1['a1'], self._a_grid, r, chi0, chi1, chi2)
        chi_hh2 = Psi_fun(ss2['a2'], self._a_grid, r, chi0, chi1, chi2)
        chi_hh3 = Psi_fun(ss3['a3'], self._a_grid, r, chi0, chi1, chi2)
        Chi1 = np.vdot(ss1['D'], chi_hh1)
        Chi2 = np.vdot(ss2['D'], chi_hh2)
        Chi3 = np.vdot(ss3['D'], chi_hh3)
        goods_mkt = ss1['C1'] + ss2['C2'] + ss3['C3'] + I + G + Chi1 + Chi2 + Chi3 + omega * (ss1['B1'] + ss2['B2'] + ss3['B3']) - Y
        #    assert np.abs(goods_mkt) < 1E-7

        ss = ss1

        ss.update({'pi': 0, 'piw': 0, 'Q': Q, 'Y': Y, 'mc': mc, 'K': K, 'I': I, 'tax': tax,
                   'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi1 + Chi2 + Chi3, 'chi': chi_hh1 + chi_hh2 + chi_hh3, 'phi': phi,
                   'beta': beta, 'vphi': (vphi1 * vphi2 * vphi3) ** (1 / 3), 'omega': omega, 'delta': delta, 'muw': muw,
                   'frisch': frisch, 'epsI': epsI, 'a_grid': self._a_grid, 'b_grid': self._b_grid,
                   'z_grid': sum(z_grid), 'e_grid': self._e_grid,
                   'k_grid': self._k_grid, 'Pi': self._Pi, 'kappap': kappap, 'kappaw': kappaw, 'rstar': r, 'i': r, 'w': w,
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

                   'K_sec_1': self._K_sec[0], 'K_sec_2': self._K_sec[1], 'K_sec_3': self._K_sec[2],
                   'Y_sec_1': Y_sec[0], 'Y_sec_2': Y_sec[1], 'Y_sec_3': Y_sec[2],
                   'N_sec_1': N_sec[0], 'N_sec_2': N_sec[1], 'N_sec_3': N_sec[2],
                   'N': N_sum,
                   'gamma_hh_1_1': self._gamma_hh[0][0], 'gamma_hh_1_2': self._gamma_hh[0][1], 'gamma_hh_1_3': self._gamma_hh[0][2],
                   'gamma_hh_2_1': self._gamma_hh[1][0], 'gamma_hh_2_2': self._gamma_hh[1][1], 'gamma_hh_2_3': self._gamma_hh[1][2],
                   'gamma_hh_3_1': self._gamma_hh[2][0], 'gamma_hh_3_2': self._gamma_hh[2][1], 'gamma_hh_3_3': self._gamma_hh[2][2],
                   'w_occ_1': self._w_occ[0], 'w_occ_2': self._w_occ[1], 'w_occ_3': self._w_occ[2],
                   'w_sec_1': w_sec[0], 'w_sec_2': w_sec[1], 'w_sec_3': w_sec[2],
                   'I_sec_1': I_sec[0], 'I_sec_2': I_sec[1], 'I_sec_3': I_sec[2],
                   'productivity_sec_1': productivity_sec[0],
                   'productivity_sec_2': productivity_sec[1],
                   'productivity_sec_3': productivity_sec[2],
                   'p_sec_1': p_sec[0], 'p_sec_2': p_sec[1], 'p_sec_3': p_sec[2],
                   'div_sec_1': div_sec[0], 'div_sec_2': div_sec[1], 'div_sec_3': div_sec[2],
                   'div': div,
                   'Q_sec_1': self._Q_sec[0], 'Q_sec_2': self._Q_sec[1], 'Q_sec_3': self._Q_sec[2],
                   'N_occ_sec_1_1': self._N_sec_occ[0][0], 'N_occ_sec_1_2': self._N_sec_occ[1][0], 'N_occ_sec_1_3': self._N_sec_occ[2][0],
                   'N_occ_sec_2_1': self._N_sec_occ[0][1], 'N_occ_sec_2_2': self._N_sec_occ[1][1], 'N_occ_sec_2_3': self._N_sec_occ[2][1],
                   'N_occ_sec_3_1': self._N_sec_occ[0][2], 'N_occ_sec_3_2': self._N_sec_occ[1][2], 'N_occ_sec_3_3': self._N_sec_occ[2][2],

                   'sigma_sec_1': self._sigma_sec[0], 'sigma_sec_2': self._sigma_sec[0], 'sigma_sec_3': self._sigma_sec[0],
                   'psip_sec_1': 0, 'psip_sec_2': 0, 'psip_sec_3': 0, 'psip': 0,
                   "mc_sec_1": mc_sec[0], "mc_sec_2": mc_sec[1], "mc_sec_3": mc_sec[2],
                   "nu_sec_1": self._nu_sec[0], "nu_sec_2": self._nu_sec[1], "nu_sec_3": self._nu_sec[2],
                   'equity_price_sec_1': equity_price_sec[0],
                   'equity_price_sec_2': equity_price_sec[1],
                   'equity_price_sec_3': equity_price_sec[2],
                   'alpha_occ_sec_1_1': self._alpha_sec_occ[0][0],
                   'alpha_occ_sec_1_2': self._alpha_sec_occ[1][0],
                   'alpha_occ_sec_1_3': self._alpha_sec_occ[2][0],
                   'alpha_occ_sec_2_1': self._alpha_sec_occ[0][1],
                   'alpha_occ_sec_2_2': self._alpha_sec_occ[1][1],
                   'alpha_occ_sec_2_3': self._alpha_sec_occ[2][1],
                   'alpha_occ_sec_3_1': self._alpha_sec_occ[0][2],
                   'alpha_occ_sec_3_2': self._alpha_sec_occ[1][2],
                   'alpha_occ_sec_3_3': self._alpha_sec_occ[2][2],

                   'pshare_1': pshare_sec[0],
                   'pshare_2': pshare_sec[1],
                   'pshare_3': pshare_sec[2],
                   'pshare': pshare,
                   'N_occ_1': N_occ[0], 'N_occ_2': N_occ[1], 'N_occ_3': N_occ[2],
                   'L_sec_1': self._L_sec[0], 'L_sec_2': self._L_sec[1], 'L_sec_3': self._L_sec[2],
                   'm1': self._m[0], 'm2': self._m[1], 'm3': self._m[2],

                   'N_hh_occ_1_1': self._N_hh_occ[0][0], 'N_hh_occ_1_2': self._N_hh_occ[0][1], 'N_hh_occ_1_3': self._N_hh_occ[0][2],
                   'N_hh_occ_2_1': self._N_hh_occ[1][0], 'N_hh_occ_2_2': self._N_hh_occ[1][1], 'N_hh_occ_2_3': self._N_hh_occ[1][2],
                   'N_hh_occ_3_1': self._N_hh_occ[2][0], 'N_hh_occ_3_2': self._N_hh_occ[2][1], 'N_hh_occ_3_3': self._N_hh_occ[2][2]})
        return ss
