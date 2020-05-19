from itertools import product

import numpy as np

import utils
from households import household, income_grid
from households.misc import Psi_fun
from ._labor_demand_calculation import LaborDemandCalculation

household_inc = household.attach_hetinput(income_grid)


# noinspection PyPep8Naming
class SSBuilder:
    def _set_up_gamma_hh(self):
        self._gamma_hh = [[1.000000000000000, 0.20000000000000, 0.300000000000000],
                          [0.278417140245438, 1.00000000000000, 0.357559561729431],
                          [0.257629066705704, 0.32952556014061, 1.000000000000000]]

    def _set_up_N_sec_occ(self):
        self._N_sec_occ = [[0.662516653537750, 0.182436376810074, 0.155046954751015],
                           [0.372738629579544, 0.191605582833290, 0.435655802488327],
                           [0.255537301301956, 0.227060705423355, 0.517401993274688]]

    def _calc_grids(self):
        self._b_grid = utils.agrid(amax=self._bmax, n=self._nB)
        self._a_grid = utils.agrid(amax=self._amax, n=self._nA)
        self._k_grid = utils.agrid(amax=self._kmax, n=self._nK)
        self._e_grid, self._pi, self._Pi = utils.markov_rouwenhorst(rho=self._rho_z, sigma=self._sigma_z, N=self._nZ)

    def _calc_eta(self):
        self._eta = (1 - self._mup) / self._mup

    def _calc_Y(self):
        self._Y = (self._Y_sec[0] ** ((self._eta - 1) / self._eta) +
                   self._Y_sec[1] ** ((self._eta - 1) / self._eta) +
                   self._Y_sec[2] ** ((self._eta - 1) / self._eta)) ** (self._eta / (self._eta - 1))

    def _calc_alpha_sec_occ(self):
        self._alpha_sec_occ = np.zeros((3, 3))
        for i, j in product(range(3), repeat=2):
            self._alpha_sec_occ[i][j] = (
                    self._N_sec_occ[i][j] ** (1 - self._sigma_sec[i]) * self._w_occ[j]) / (
                    self._N_sec_occ[i][0] ** (1 - self._sigma_sec[i]) * self._w_occ[0] +
                    self._N_sec_occ[i][1] ** (1 - self._sigma_sec[i]) * self._w_occ[1] +
                    self._N_sec_occ[i][2] ** (1 - self._sigma_sec[i]) * self._w_occ[2])

    def _calc_p_sec(self):
        self._p_sec = [(self._Y / self._Y_sec[i]) ** (1 / self._eta) * self._p for i in range(3)]

    def _recalc_N_sec_occ_by_labor_demand_calculation(self):
        self._N_sec_occ = LaborDemandCalculation(self._alpha_sec_occ,
                                                 self._sigma_sec,
                                                 self._p_sec,
                                                 self._Y_sec,
                                                 self._nu_sec,
                                                 self._w_occ).out()

    def _calc_L_sec(self):
        self._L_sec = [(self._alpha_sec_occ[i][0] * self._N_sec_occ[i][0] ** self._sigma_sec[i] +
                        self._alpha_sec_occ[i][1] * self._N_sec_occ[i][1] ** self._sigma_sec[i] +
                        self._alpha_sec_occ[i][2] * self._N_sec_occ[i][2] ** self._sigma_sec[i]) **
                       (1 / self._sigma_sec[i])
                       for i in range(3)]

    def _calc_K_sec_and_K(self):
        self._K_sec = [1 /
                       (1 + self._r * self._Q_sec[i]) * self._nu_sec[i] /
                       (1 - self._nu_sec[i]) *
                       self._L_sec[i] ** self._sigma_sec[i] *
                       (self._w_occ[0] * self._N_sec_occ[i][0] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][0] +
                        self._w_occ[1] * self._N_sec_occ[i][1] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][1] +
                        self._w_occ[2] * self._N_sec_occ[i][2] ** (1 - self._sigma_sec[i]) / self._alpha_sec_occ[i][2])
                       for i in range(3)]
        self._K = sum(self._K_sec)

    def _calc_mc(self):
        self._mc_sec = [(self._r * self._Q_sec[i] + self._delta) * self._K_sec[i] / self._nu_sec[i] / self._Y_sec[i]
                        for i in range(3)]
        self._mc = sum(self._mc_sec)

    def _calc_ra_and_rb(self):
        self._ra = self._r
        self._rb = self._r - self._omega

    def _calc_I_sec_and_I(self):
        self._I_sec = [self._delta * self._K_sec[i] for i in range(3)]
        self._I = self._delta * self._K

    def _calc_productivity_sec(self):
        self._productivity_sec = [self._Y_sec[i] *
                                  self._K_sec[i] ** (-self._nu_sec[i]) *
                                  self._L_sec[i] ** (self._nu_sec[i] - 1)
                                  for i in range(3)]

    def _calc_N_sec_and_N_sum(self):
        self._N_sec = [sum(self._N_sec_occ[i]) for i in range(3)]
        self._N_sum = sum(self._N_sec)

    def _calc_N_occ(self):
        self._N_occ = [self._N_sec_occ[0][i] + self._N_sec_occ[1][i] + self._N_sec_occ[2][i] for i in range(3)]

    def _calc_w_sec(self):
        self._w_sec = [(self._w_occ[0] * self._N_sec_occ[i][0] +
                        self._w_occ[1] * self._N_sec_occ[i][1] +
                        self._w_occ[2] * self._N_sec_occ[i][2]) / self._N_sec[i]
                       for i in range(3)]

    def _calc_w(self):
        self._w = (self._w_sec[0] * self._N_sec[0] +
                   self._w_sec[1] * self._N_sec[1] +
                   self._w_sec[2] * self._N_sec[2]) / self._N_sum

    def _calc_tax(self):
        self._tax = (self._r * self._Bg + self._G) / self._w / self._N_sum

    def _calc_div(self):
        self._div = self._p * self._Y - self._w * self._N_sum - self._I

    def _calc_equity_price(self):
        self._equity_price = self._div / self._r

    def _calc_pshare(self):
        self._pshare = self._equity_price / (self._tot_wealth - self._Bh)

    def _calc_z_grid(self):
        self._z_grid = [income_grid(self._e_grid, self._tax, self._w_occ, self._gamma_hh[i],self._m[i], self._N_hh_occ[i])
                        for i in range(3)]

    def _calc_Va(self):
        self._Va = [(0.6 + 1.1 * self._b_grid[:, np.newaxis] + self._a_grid) ** (-1 / self._eis) *
                    np.ones((self._z_grid[i].shape[0], 1, 1))
                    for i in range(3)]

    def _calc_Vb(self):
        self._Vb = [(0.5 + self._b_grid[:, np.newaxis] + 1.2 * self._a_grid) ** (-1 / self._eis) *
                    np.ones((self._z_grid[i].shape[0], 1, 1))
                    for i in range(3)]

    def _calc_labor_hours_hh(self):
        self._labor_hours_hh = [self._N_hh_occ[i] for i in range(3)]

    def _calc_div_sec(self):
        self._div_sec = [self._Y_sec[i] * self._p_sec[i] - self._w_sec[i] * self._N_sec[i] - self._I_sec[i]
                         for i in range(3)]

    def _calc_equity_price_sec(self):
        self._equity_price_sec = [self._div_sec[i] / self._r for i in range(3)]

    def _calc_pshare_sec(self):
        self._pshare_sec = [self._equity_price_sec[i] / (self._tot_wealth - self._Bh) for i in range(3)]

    def _calc_beta_vphi_omega(self):
        x0 = np.array([self._beta_guess, self._vphi_guess, self._vphi_guess, self._vphi_guess, self._chi1_guess])
        x, _ = utils.broyden_solver(self._res, x0, noisy=self._noisy)
        self._beta, vphi1, vphi2, vphi3, self._chi1 = x
        self._vphi = (vphi1, vphi2, vphi3)

    def _calc_labor_mk(self):
        self._labor_mk = [self._N_occ[i] -
                          self._labor_hours_hh[0][0] * (0 == i) -
                          self._labor_hours_hh[1][1] * (1 == i) -
                          self._labor_hours_hh[2][2] * (2 == i)
                          for i in range(3)]

    def _household_inc_to_ss(self):
        self._ss_list = [household_inc.ss(a_grid=self._a_grid,
                                          b_grid=self._b_grid,
                                          beta=self._beta,
                                          chi0=self._chi0,
                                          chi1=self._chi1,
                                          chi2=self._chi2,
                                          e_grid=self._e_grid,
                                          eis=self._eis,
                                          gamma_hh=self._gamma_hh[i],
                                          k_grid=self._k_grid,
                                          m=self._m[i],
                                          N_hh_occ=self._N_hh_occ[i],
                                          Pi=self._Pi,
                                          ra=self._ra,
                                          rb=self._rb,
                                          tax=self._tax,
                                          Va=self._Va[i],
                                          Vb=self._Vb[i],
                                          w_occ=self._w_occ)
                         for i in range(3)]

    def _calc_chi_hh(self):
        self._chi_hh = [Psi_fun(self._ss_list[i]['a'], self._a_grid, self._r, self._chi0, self._chi1, self._chi2)
                        for i in range(3)]

    def _calc_Chi(self):
        self._Chi = [np.vdot(self._ss_list[i]['D'], self._chi_hh[i]) for i in range(3)]

    # residual function
    def _res(self, x):
        beta_loc, vphi_loc1, vphi_loc2, vphi_loc3, chi1_loc = x
        if any((beta_loc > 0.999 / (1 + self._r),
                vphi_loc1 < 0.001,
                vphi_loc2 < 0.001,
                vphi_loc3 < 0.001,
                chi1_loc < 0.5)):
            raise ValueError("Clearly invalid inputs")

        vphi_loc = [vphi_loc1, vphi_loc2, vphi_loc3]

        out = [household_inc.ss(a_grid=self._a_grid,
                                b_grid=self._b_grid,
                                beta=beta_loc,
                                chi0=self._chi0,
                                chi1=chi1_loc,
                                chi2=self._chi2,
                                e_grid=self._e_grid,
                                eis=self._eis,
                                gamma_hh=self._gamma_hh[i],
                                k_grid=self._k_grid,
                                m=self._m[i],
                                N_hh_occ=self._N_hh_occ[i],
                                Pi=self._Pi,
                                ra=self._ra,
                                rb=self._rb,
                                tax=self._tax,
                                Va=self._Va[i],
                                Vb=self._Vb[i],
                                w_occ=self._w_occ)
               for i in range(3)]

        asset_mkt = (out[0]["A"] + out[1]["A"] + out[2]["A"] +
                     out[0]["B"] + out[1]["B"] + out[2]["B"] -
                     self._equity_price - self._Bg)

        intratemp_hh = [vphi_loc[i] * (self._labor_hours_hh[i][i] / self._m[i] / self._gamma_hh[i][i]) ** (1/self._frisch) -
                        (1 - self._tax) * self._w_occ[i] * out[i][f"U"] * self._gamma_hh[i][i] * self._m[i]
                        for i in range(3)]

        result = np.array([asset_mkt, intratemp_hh[0], intratemp_hh[1], intratemp_hh[2],
                           out[0]['B'] + out[1]['B'] + out[2]['B'] - self._Bh])
        return result

    def _update_ss_with_household_outcome(self):
        for char in "ABCU":
            self._ss.update({char: self._ss_list[0][char] + self._ss_list[1][char] + self._ss_list[2][char]})

        for i in range(3):
            j = i + 1
            v = {f"{key}{j}": self._ss_list[i][key] for key in ("Va", "Vb", "A", "B", "C", "U", "a", "b", "c", "u")}
            v.update({f"a{j}_grid": self._ss_list[i]['a_grid'], f"b{j}_grid": self._ss_list[i]['b_grid']})
            self._ss.update(v)

    def _update_ss_with_matrix_values(self):
        self._ss.update({f"alpha_occ_sec_{i + 1}_{j + 1}": self._alpha_sec_occ[j][i]
                         for i, j in product(range(3), repeat=2)})
        self._ss.update({f"gamma_hh_{i + 1}_{j + 1}": self._gamma_hh[i][j] for i, j in product(range(3), repeat=2)})
        self._ss.update({f"N_occ_sec_{i + 1}_{j + 1}": self._N_sec_occ[j][i] for i, j in product(range(3), repeat=2)})
        self._ss.update({f"N_hh_occ_{i + 1}_{j + 1}": self._N_hh_occ[i][j] for i, j in product(range(3), repeat=2)})

    def _update_ss_with_scalar_values(self, scalars):
        for s in scalars:
            self._ss.update({s: eval(f"self._{s}")})

    def _update_ss_with_vector_values(self, vectors, size):
        for i in range(size):
            for vec in vectors:
                self._ss.update({f"{vec}_{i+1}": eval(f"self._{vec}[{i}]")})

    def __init__(self,
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
                 beta_guess=0.976):
        self._amax = amax
        self._beta_guess = beta_guess
        self._beta_sir = 1.5
        self._Bg = Bg
        self._Bh = Bh
        self._bmax = bmax
        self._chi0 = chi0
        self._chi1_guess = chi1_guess
        self._chi2 = chi2
        self._covid_shock = 0
        self._delta = delta
        self._eis = eis
        self._epsI = epsI
        self._frisch = frisch
        self._G = G
        self._gamma_sir = 1
        self._infected = 0
        self._kappap = kappap
        self._kappaw = kappaw
        self._kmax = kmax
        self._m = [0.33, 0.33, 0.33]
        self._mup = 6.3863129
        self._muw = muw
        self._N = [0.33, 0.33, 0.33]
        self._N_hh_occ = [[0.33, 0, 0], [0, 0.33, 0], [0, 0, 0.33]]
        self._nA = nA
        self._nB = nB
        self._nK = nK
        self._nu_sec = [0.425027757883072, 0.538959443569183, 0.273549377918243]
        self._nZ = nZ
        self._omega = omega
        self._p = 1
        self._phi = phi
        self._possible_occupation = [0, 1, 2]
        self._psip_sec = [0, 0, 0]
        self._Q = 1
        self._Q_sec = [1, 1, 1]
        self._r = r
        self._recovered = 0
        self._rho_z = rho_z
        self._sigma_sec = [0.2, 0.2, 0.2]
        self._sigma_z = sigma_z
        self._ss = {}
        self._tot_wealth = tot_wealth
        self._vphi_guess = vphi_guess
        self._w_occ = [3, 8, 2]
        self._Y_sec = [0.260986566543579, 0.343330949544907, 0.398539662361145]

        self._set_up_gamma_hh()
        self._set_up_N_sec_occ()

    def hank_ss(self, noisy=True):
        """
        Solve steady state of full GE model.
        Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for (r, tot_wealth, Bh, K, Y=N=1).
        """
        self._noisy = noisy
        self._calc_grids()
        self._calc_eta()
        self._calc_Y()
        self._calc_alpha_sec_occ()
        self._calc_p_sec()
        self._recalc_N_sec_occ_by_labor_demand_calculation()
        self._calc_L_sec()
        self._calc_K_sec_and_K()
        self._calc_mc()
        self._calc_ra_and_rb()
        self._calc_I_sec_and_I()
        self._calc_productivity_sec()
        self._calc_N_sec_and_N_sum()
        self._calc_N_occ()
        self._calc_w_sec()
        self._calc_w()
        self._calc_tax()
        self._calc_div()
        self._calc_equity_price()
        self._calc_pshare()
        self._calc_z_grid()
        self._calc_Va()
        self._calc_Vb()
        self._calc_labor_hours_hh()
        self._calc_div_sec()
        self._calc_equity_price_sec()
        self._calc_pshare_sec()
        self._calc_beta_vphi_omega()
        self._calc_labor_mk()
        self._household_inc_to_ss()
        self._calc_chi_hh()
        self._calc_Chi()

        goods_mkt = (self._ss_list[0]['C'] + self._ss_list[1]['C'] + self._ss_list[2]['C'] +
                     self._I + self._G + self._Chi[0] + self._Chi[1] + self._Chi[2] +
                     self._omega * (self._ss_list[0]['B'] + self._ss_list[1]['B'] + self._ss_list[2]['B']) - self._Y)
        #    assert np.abs(goods_mkt) < 1E-7

        self._ss.update(self._ss_list[0])

        self._update_ss_with_household_outcome()
        self._update_ss_with_matrix_values()

        self._update_ss_with_scalar_values(["a_grid", "b_grid", "beta", "beta_sir", "Bg", "covid_shock", "delta",
                                            "div", "e_grid", "epsI", "eta", "frisch", "G", "gamma_sir", "I",
                                            "infected", "K", "k_grid", "kappap", "kappaw", "mc", "mup", "muw", "omega",
                                            "p", "phi", "Pi", "pshare", "Q", "r", "ra", "rb", "recovered", "tax",
                                            "w", "Y"])

        self._update_ss_with_vector_values(["div_sec", "div_sec", "equity_price_sec", "I_sec", "K_sec", "L_sec",
                                            "mc_sec", "N_occ", "N_sec", "nu_sec", "p_sec", "productivity_sec",
                                            "psip_sec", "possible_occupation", "Q_sec", "sigma_sec", "vphi", "w_occ",
                                            "w_sec", "Y_sec", ], 3)

        self._ss.update({'Chi': sum(self._Chi),
                         'chi': sum(self._chi_hh),
                         'i': self._r,
                         'm1': self._m[0],
                         'm2': self._m[1],
                         'm3': self._m[2],
                         'N': self._N_sum,
                         'pi': 0,
                         'piw': 0,
                         'pshare_1': self._pshare_sec[0],
                         'pshare_2': self._pshare_sec[1],
                         'pshare_3': self._pshare_sec[2],
                         'psip': 0,
                         'rstar': self._r,
                         'susceptible': 1,
                         'vphi': sum(self._vphi) ** (1 / 3),
                         "w1": self._w_occ[0],
                         "w2": self._w_occ[1],
                         "w3": self._w_occ[2],
                         'z_grid': sum(self._z_grid)})
        return self._ss
