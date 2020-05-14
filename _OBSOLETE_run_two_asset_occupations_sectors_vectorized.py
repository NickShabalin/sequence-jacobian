import numpy as np
import matplotlib.pyplot as plt

import utils
import jacobian as jac
from het_block import het
from simple_block import simple
from solved_block import solved
import _OBSOLETE_two_asset_sec_occ_v3
import determinacy as det
from simple_with_vector_args import SimpleWithVectorArgs

@simple
def taylor(rstar, pi, phi):
    i = rstar + phi * pi
    return i

@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax

@simple
def finance(i, pi, r, div_sec_1, div_sec_2, div_sec_3, omega, pshare_1, pshare_2, pshare_3, equity_price_sec_1, equity_price_sec_2, equity_price_sec_3):
    rb = r - omega
    ra = pshare_1 * (div_sec_1 + equity_price_sec_1) / equity_price_sec_1(-1) + pshare_2 * (div_sec_2 + equity_price_sec_2) / equity_price_sec_2(-1) + pshare_3 * (div_sec_3 + equity_price_sec_3) / equity_price_sec_3(-1) + (1 - pshare_1 - pshare_2 - pshare_3) * (1 + r) - 1
    #fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    fisher = 1 + i - (1 + r(+1)) * (1 + pi(+1))
    return rb, fisher, ra


@SimpleWithVectorArgs({"N_occ": 3, "N_hh_occ_1": 3, "N_hh_occ_2": 3, "N_hh_occ_3": 3, "gamma_hh_1" : 3, "gamma_hh_2": 3, "gamma_hh_3": 3})
def mkt_clearing(equity_price_sec_1, equity_price_sec_2, equity_price_sec_3, A, B, Bg, m1, m2, m3, N_occ, gamma_hh_1, gamma_hh_2, gamma_hh_3, N_hh_occ_1, N_hh_occ_2, N_hh_occ_3):
    asset_mkt = equity_price_sec_1 + equity_price_sec_2 + equity_price_sec_3 + Bg - B - A
    labor_mkt = N_occ - m1 * gamma_hh_1 * N_hh_occ_1 - m2 * gamma_hh_2 * N_hh_occ_2 - m3 * gamma_hh_3 * N_hh_occ_3
    return asset_mkt, labor_mkt

@SimpleWithVectorArgs({"div_sec": 3, "equity_price_sec": 3})
def arbitrage(div_sec, equity_price_sec, r):
    #equity = div_sec(+1) + equity_price_sec(+1) - equity_price_sec * (1 + r(+1))
    equity = div_sec + equity_price_sec - equity_price_sec(-1) * (1 + r)
    return equity

@SimpleWithVectorArgs({"Y_sec": 3, "K_sec": 3, "w_sec": 3, "N_sec": 3, "p_sec": 3})
def dividend(Y_sec, w_sec, N_sec, K_sec, p_sec, pi, mup, kappap, delta):
    psip_sec = mup / (mup - 1) / 2 / kappap * np.log(1 + pi) ** 2 * Y_sec
    I_sec = K_sec - (1 - delta) * K_sec(-1)
    div_sec = Y_sec * p_sec - w_sec * N_sec - I_sec - psip_sec
    return psip_sec, I_sec, div_sec

@simple
def dividend_agg(I_sec_1, I_sec_2, I_sec_3, psip_sec_1, psip_sec_2, psip_sec_3, div_sec_1, div_sec_2, div_sec_3):
    I = I_sec_1 + I_sec_2 + I_sec_3
    div = div_sec_1 + div_sec_2 + div_sec_3
    psip = psip_sec_1 + psip_sec_2 + psip_sec_3
    return I, div, psip


@simple
def pricing(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1/mup) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1)) - np.log(1 + pi)
    return nkpc

@SimpleWithVectorArgs({"K_sec": 3, "Q_sec": 3, "productivity_sec": 3, "L_sec": 3, "mc_sec": 3, "nu_sec": 3})
def investment(Q_sec, K_sec, r, L_sec, mc_sec, productivity_sec, delta, epsI, nu_sec):
    inv_sec = (K_sec / K_sec(-1) - 1) / (delta * epsI) + 1 - Q_sec
    val_sec = nu_sec * productivity_sec(+1) * (L_sec(+1) / K_sec) ** (1- nu_sec) * mc_sec(+1) - (K_sec(+1) / K_sec - (1-delta) + (K_sec(+1) / K_sec - 1)**2 / (2*delta*epsI)) + K_sec(+1) / K_sec * Q_sec(+1) - (1 + r(+1)) * Q_sec
    return inv_sec, val_sec



@simple
def output_aggregation(Y_sec_1, Y_sec_2, Y_sec_3, eta):
    power_y = eta / (eta-1)
    Y = (Y_sec_1 ** (1/power_y) + Y_sec_2 ** (1/power_y) + Y_sec_3 ** (1/power_y)) ** power_y
    return Y

@SimpleWithVectorArgs({"productivity_sec": 3, "L_sec": 3, "nu_sec": 3, "K_sec": 3, "Y_sec": 3, "p_sec": 3, "sigma_sec": 3, "alpha_occ_sec_1": 3, "alpha_occ_sec_2": 3, "alpha_occ_sec_3": 3, "N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3})
def production_sec(productivity_sec, L_sec, nu_sec, K_sec, Y_sec, p_sec, alpha_occ_sec_1, alpha_occ_sec_2, alpha_occ_sec_3, w_occ_1, w_occ_2, w_occ_3, sigma_sec, N_occ_sec_1, N_occ_sec_2, N_occ_sec_3):
    prod_sec = productivity_sec * L_sec ** (1 - nu_sec) * K_sec(-1) ** nu_sec - Y_sec
    labor_sec_occ_1 = p_sec * (1 - nu_sec) * alpha_occ_sec_1 / w_occ_1 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_1 * (1 - sigma_sec)
    labor_sec_occ_2 = p_sec * (1 - nu_sec) * alpha_occ_sec_2 / w_occ_2 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_2 * (1 - sigma_sec)
    labor_sec_occ_3 = p_sec * (1 - nu_sec) * alpha_occ_sec_3 / w_occ_3 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_3 * (1 - sigma_sec)
    return prod_sec, labor_sec_occ_1, labor_sec_occ_2, labor_sec_occ_3




@SimpleWithVectorArgs({"N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3, "alpha_occ_sec_1": 3, "alpha_occ_sec_2": 3, "alpha_occ_sec_3": 3, "sigma_sec": 3, "L_sec": 3, "Q_sec": 3, "K_sec": 3, "nu_sec": 3, "Y_sec": 3})
def marginal_costs(w_occ_1, w_occ_2, w_occ_3, N_occ_sec_1, N_occ_sec_2, N_occ_sec_3, sigma_sec, alpha_occ_sec_1, alpha_occ_sec_2, alpha_occ_sec_3, L_sec, epsI, Q_sec, delta, K_sec, nu_sec, Y_sec):
    mc_sec = (w_occ_1 * N_occ_sec_1 ** (1 - sigma_sec) / alpha_occ_sec_1 + w_occ_2 * N_occ_sec_2 ** (1 - sigma_sec) / alpha_occ_sec_2 + w_occ_3 * N_occ_sec_3 ** (1 - sigma_sec) / alpha_occ_sec_3) * L_sec ** sigma_sec / (1 - nu_sec) / Y_sec - (1 - delta) * Q_sec * K_sec(-1) / nu_sec / Y_sec - 1 / delta / epsI / nu_sec / Y_sec / 2 / K_sec(-1) * (K_sec ** 2 - K_sec(-1) ** 2)
    mc = mc_sec_1 + mc_sec_2 + mc_sec_3
    return mc_sec, mc

@simple
def wage_labor_aggregates(w_occ_1, w_occ_2, w_occ_3, N_occ_sec_1_1, N_occ_sec_1_2, N_occ_sec_1_3, N_occ_sec_2_1, N_occ_sec_2_2, N_occ_sec_2_3, N_occ_sec_3_1, N_occ_sec_3_2, N_occ_sec_3_3):

    N_occ_1 = N_occ_sec_1_1 + N_occ_sec_1_2 + N_occ_sec_1_3
    N_occ_2 = N_occ_sec_2_1 + N_occ_sec_2_2 + N_occ_sec_2_3
    N_occ_3 = N_occ_sec_3_1 + N_occ_sec_3_2 + N_occ_sec_3_3

    N_sec_1 = N_occ_sec_1_1 + N_occ_sec_2_1 + N_occ_sec_3_1
    N_sec_2 = N_occ_sec_1_2 + N_occ_sec_2_2 + N_occ_sec_3_2
    N_sec_3 = N_occ_sec_1_3 + N_occ_sec_2_3 + N_occ_sec_3_3
    N = N_sec_1 + N_sec_2 + N_sec_3


    w_sec_1 = (w_occ_1 * N_occ_sec_1_1 + w_occ_2 * N_occ_sec_2_1 + w_occ_3 * N_occ_sec_3_1) / N_sec_1
    w_sec_2 = (w_occ_1 * N_occ_sec_1_2 + w_occ_2 * N_occ_sec_2_2 + w_occ_3 * N_occ_sec_3_2) / N_sec_2
    w_sec_3 = (w_occ_1 * N_occ_sec_1_3 + w_occ_2 * N_occ_sec_2_3 + w_occ_3 * N_occ_sec_3_3) / N_sec_3
    w = (w_sec_1 * N_sec_1 + w_sec_2 * N_sec_2 + w_sec_3 * N_sec_3) / N

    return w, N, w_sec_1, w_sec_2, w_sec_3, N_sec_1, N_sec_2, N_sec_3, N_occ_1, N_occ_2, N_occ_3

@SimpleWithVectorArgs({"Y_sec": 3})
def pricing_intermediate(Y, Y_sec, eta):
    p_sec = (Y / Y_sec) ** (1/eta)
    return p_sec

'''
production = solved(block_list=[output_aggregation, production_sec, investment, marginal_costs,
                                wage_labor_aggregates, dividend, pricing, arbitrage, dividend_agg],
                    unknowns=['Q_sec', 'K_sec', 'pi', 'Y_sec', 'N_occ_sec_1', 'N_occ_sec_2', 'N_occ_sec_3', 'p_sec'],
                    targets=['inv_sec', 'val_sec', 'nkpc', 'prod_sec', 'labor_sec_occ_1', 'labor_sec_occ_2', 'labor_sec_occ_3', 'equity'])
'''
production = solved(block_list=[output_aggregation, production_sec, investment, marginal_costs,
                                wage_labor_aggregates, dividend, pricing, arbitrage, dividend_agg],
                    unknowns=['Q_sec_1', 'Q_sec_2', 'Q_sec_3', 'K_sec_1', 'K_sec_2', 'K_sec_3', 'pi',
                              'Y_sec_1', 'Y_sec_2', 'Y_sec_3', 'p_sec_1', 'p_sec_2', 'p_sec_3',
                              'N_occ_sec_1_1', 'N_occ_sec_1_2', 'N_occ_sec_1_3',
                              'N_occ_sec_2_1', 'N_occ_sec_2_2', 'N_occ_sec_2_3',
                              'N_occ_sec_3_1', 'N_occ_sec_3_2', 'N_occ_sec_3_3'],
                    targets=['inv_sec_1', 'inv_sec_2', 'inv_sec_3', 'val_sec_1', 'val_sec_2', 'val_sec_3', 'nkpc',
                             'prod_sec_1', 'prod_sec_2', 'prod_sec_3', 'equity_1', 'equity_2', 'equity_3',
                             'labor_sec_occ_1_1', 'labor_sec_occ_1_2', 'labor_sec_occ_1_3',
                             'labor_sec_occ_2_1', 'labor_sec_occ_2_2', 'labor_sec_occ_2_3',
                             'labor_sec_occ_3_1', 'labor_sec_occ_3_2', 'labor_sec_occ_3_3'])


@simple
def consumers_aggregator(C1, C2, C3, A1, A2, A3, B1, B2, B3, U1, U2, U3):
    C = C1 + C2 + C3
    A = A1 + A2 + A3
    B = B1 + B2 + B3
    U = U1 + U2 + U3
    return C, A, B, U

'''
@SimpleWithVectorArgs({"w_occ": 3, "gamma_hh_1": 3})
def income1(w_occ, gamma_hh_1, m1, tax, U1, vphi_1, frisch):

    N_hh_occ_1 = ((1 - tax) * w_occ * U1 * gamma_hh_1 * m1 / vphi_1) ** frisch * m1 * gamma_hh_1

    N_eff_hh_1 = N_hh_occ_1 * m1 * gamma_hh_1

    choice = N_eff_hh_1 * w_occ
    choices = [choice_1, choice_2, choice_3]

    occupation = choices.index(max(choices))
    number = np.array([0, 1, 2])

    N_hh_occ_1 = (occupation == number) * N_hh_occ_1

    return N_hh_occ_1


@SimpleWithVectorArgs({"w_occ": 3, "gamma_hh_2": 3})
def income2(w_occ, gamma_hh_2, m2, tax, U2, vphi_2, frisch):

    N_hh_occ_2 = ((1 - tax) * w_occ * U2 * gamma_hh_2 * m2 / vphi_2) ** frisch * m2 * gamma_hh_2

    N_eff_hh_2 = N_hh_occ_2 * m2 * gamma_hh_2

    choice = N_eff_hh_2 * w_occ
    choices = [choice_1, choice_2, choice_3]

    occupation = choices.index(max(choices))
    number = np.array([0, 1, 2])

    N_hh_occ_2 = (occupation == number) * N_hh_occ_2

    return N_hh_occ_2

@SimpleWithVectorArgs({"w_occ": 3, "gamma_hh_3": 3})
def income2(w_occ, gamma_hh_3, m3, tax, U3, vphi_3, frisch):

    N_hh_occ_3 = ((1 - tax) * w_occ * U3 * gamma_hh_3 * m3 / vphi_3) ** frisch * m3 * gamma_hh_3

    N_eff_hh_3 = N_hh_occ_3 * m3 * gamma_hh_3

    choice = N_eff_hh_3 * w_occ
    choices = [choice_1, choice_2, choice_3]

    occupation = choices.index(max(choices))
    number = np.array([0, 1, 2])

    N_hh_occ_3 = (occupation == number) * N_hh_occ_3

    return N_hh_occ_3
'''
@simple
def income1(w_occ_1, w_occ_2, w_occ_3, gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3, m1, N = 0.33):

    gamma_occ1 = gamma_hh_1_1
    gamma_occ2 = gamma_hh_1_2
    gamma_occ3 = gamma_hh_1_3

    N_occ1 = N * m1 * gamma_occ1
    N_occ2 = N * m1 * gamma_occ2
    N_occ3 = N * m1 * gamma_occ3


    choice1 = N_occ1 * w_occ_1
    choice2 = N_occ2 * w_occ_2
    choice3 = N_occ3 * w_occ_3
    choices = [choice1, choice2, choice3]

    occupation = choices.index(max(choices))

    N_hh_occ_1_1 = (occupation == 0) * N_occ1
    N_hh_occ_1_2 = (occupation == 1) * N_occ2
    N_hh_occ_1_3 = (occupation == 2) * N_occ3

    return N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3


@simple
def income2(w_occ_1, w_occ_2, w_occ_3, gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3, m2, N = 0.33):

    gamma_occ1 = gamma_hh_2_1
    gamma_occ2 = gamma_hh_2_2
    gamma_occ3 = gamma_hh_2_3

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

@simple
def income3(w_occ_1, w_occ_2, w_occ_3, gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3, m3, N = 0.33):

    gamma_occ1 = gamma_hh_3_1
    gamma_occ2 = gamma_hh_3_2
    gamma_occ3 = gamma_hh_3_3

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


ss = _OBSOLETE_two_asset_sec_occ_v3.hank_ss()

# DEBUG TODO: Delete this after refactoring
# import two_asset_sec_occ
# ss_1 = two_asset_sec_occ.hank_ss()
#
# for k, v in ss.items():
#     print(k)
#     if isinstance(v, np.ndarray):
#         assert np.array_equal(ss[k], ss_1[k])
#     else:
#         assert ss[k] == ss_1[k]

T = 300
block_list = [consumers_aggregator, _OBSOLETE_two_asset_sec_occ_v3.household_inc1, _OBSOLETE_two_asset_sec_occ_v3.household_inc2, _OBSOLETE_two_asset_sec_occ_v3.household_inc3,
              taylor, fiscal, finance, mkt_clearing, production, income1, income2, income3]
exogenous = ['rstar', 'productivity_sec_1', 'productivity_sec_2', 'productivity_sec_3', 'G']
unknowns = ['r', 'Bg', 'w_occ_1', 'w_occ_2', 'w_occ_3']
targets = ['asset_mkt', 'fisher', 'labor_mkt_1', 'labor_mkt_2', 'labor_mkt_3']

A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)
wn = det.winding_criterion(A)
print(f'Winding number: {wn}')

G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=True)

# create plot style
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('summer')
plt.rcParams["figure.titlesize"] = 'xx-large'

rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['rstar'] @ drstar
dN = G['N']['rstar'] @ drstar
di = G['i']['rstar'] @ drstar
dC = G['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 25 bp monetary policy shock (easing)')
axs[0,0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0,1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1,0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()

rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dC1 = G['C1']['rstar'] @ drstar
dC2 = G['C2']['rstar'] @ drstar
dC3 = G['C3']['rstar'] @ drstar
dC = G['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 25 bp monetary policy shock (easing)')
axs[0,0].plot(dC1[:25])
axs[0, 0].set_title('consumption 1 hh')
axs[0,1].plot(dC2[:25])
axs[0, 1].set_title('consumption 2 hh')
axs[1,0].plot(dC3[:25])
axs[1, 0].set_title('consumption 3 hh')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('consumption aggregate')
plt.show()


rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec1 = -0.07 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_1'] @ dproductivity_sec1
dN = G['N']['productivity_sec_1'] @ dproductivity_sec1
di = G['i']['productivity_sec_1'] @ dproductivity_sec1
dC = G['C']['productivity_sec_1'] @ dproductivity_sec1

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 1')
axs[0,0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0,1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1,0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec1 = -0.07 * rhos ** (np.arange(T)[:, np.newaxis])
dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_1'] @ dproductivity_sec1
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_1'] @ dproductivity_sec1
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_1'] @ dproductivity_sec1
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_1'] @ dproductivity_sec1
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_1'] @ dproductivity_sec1
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_1'] @ dproductivity_sec1
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_1'] @ dproductivity_sec1
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_1'] @ dproductivity_sec1
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_1'] @ dproductivity_sec1

fig, axs = plt.subplots(3, 3, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 1')
axs[0,0].plot(dN_1_1[:25])
axs[0, 0].set_title('hh sec 1 occ 1')
axs[0,1].plot(dN_1_2[:25])
axs[0, 1].set_title('hh sec 2 occ 1')
axs[0,2].plot(dN_1_3[:25])
axs[0, 2].set_title('hh sec 3 occ 1')

axs[1,0].plot(dN_2_1[:25])
axs[1, 0].set_title('hh sec 1 occ 2')
axs[1,1].plot(dN_2_2[:25])
axs[1, 1].set_title('hh sec 2 occ 2')
axs[1,2].plot(dN_2_3[:25])
axs[1, 2].set_title('hh sec 3 occ 2')


axs[2,0].plot(dN_3_1[:25])
axs[2, 0].set_title('hh sec 1 occ 3')
axs[2,1].plot(dN_3_2[:25])
axs[2, 1].set_title('hh sec 2 occ 3')
axs[2,2].plot(dN_3_3[:25])
axs[2, 2].set_title('hh sec 3 occ 3')
plt.show()

rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec2 = -0.04 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_2'] @ dproductivity_sec2
dN = G['N']['productivity_sec_2'] @ dproductivity_sec2
di = G['i']['productivity_sec_2'] @ dproductivity_sec2
dC = G['C']['productivity_sec_2'] @ dproductivity_sec2

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 2')
axs[0,0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0,1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1,0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec2 = -0.04 * rhos ** (np.arange(T)[:, np.newaxis])
dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_2'] @ dproductivity_sec2
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_2'] @ dproductivity_sec2
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_2'] @ dproductivity_sec2
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_2'] @ dproductivity_sec2
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_2'] @ dproductivity_sec2
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_2'] @ dproductivity_sec2
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_2'] @ dproductivity_sec2
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_2'] @ dproductivity_sec2
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_2'] @ dproductivity_sec2

fig, axs = plt.subplots(3, 3, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 2')
axs[0,0].plot(dN_1_1[:25])
axs[0, 0].set_title('hh sec 1 occ 1')
axs[0,1].plot(dN_1_2[:25])
axs[0, 1].set_title('hh sec 2 occ 1')
axs[0,2].plot(dN_1_3[:25])
axs[0, 2].set_title('hh sec 3 occ 1')

axs[1,0].plot(dN_2_1[:25])
axs[1, 0].set_title('hh sec 1 occ 2')
axs[1,1].plot(dN_2_2[:25])
axs[1, 1].set_title('hh sec 2 occ 2')
axs[1,2].plot(dN_2_3[:25])
axs[1, 2].set_title('hh sec 3 occ 2')


axs[2,0].plot(dN_3_1[:25])
axs[2, 0].set_title('hh sec 1 occ 3')
axs[2,1].plot(dN_3_2[:25])
axs[2, 1].set_title('hh sec 2 occ 3')
axs[2,2].plot(dN_3_3[:25])
axs[2, 2].set_title('hh sec 3 occ 3')
plt.show()

rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec3 = -0.06 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_3'] @ dproductivity_sec3
dN = G['N']['productivity_sec_3'] @ dproductivity_sec3
di = G['i']['productivity_sec_3'] @ dproductivity_sec3
dC = G['C']['productivity_sec_3'] @ dproductivity_sec3

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 3')
axs[0,0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0,1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1,0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec3 = -0.06 * rhos ** (np.arange(T)[:, np.newaxis])
dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_3'] @ dproductivity_sec3
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_3'] @ dproductivity_sec3
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_3'] @ dproductivity_sec3
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_3'] @ dproductivity_sec3
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_3'] @ dproductivity_sec3
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_3'] @ dproductivity_sec3
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_3'] @ dproductivity_sec3
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_3'] @ dproductivity_sec3
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_3'] @ dproductivity_sec3

fig, axs = plt.subplots(3, 3, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp negative productivity shock in sector 3')
axs[0,0].plot(dN_1_1[:25])
axs[0, 0].set_title('hh sec 1 occ 1')
axs[0,1].plot(dN_1_2[:25])
axs[0, 1].set_title('hh sec 2 occ 1')
axs[0,2].plot(dN_1_3[:25])
axs[0, 2].set_title('hh sec 3 occ 1')

axs[1,0].plot(dN_2_1[:25])
axs[1, 0].set_title('hh sec 1 occ 2')
axs[1,1].plot(dN_2_2[:25])
axs[1, 1].set_title('hh sec 2 occ 2')
axs[1,2].plot(dN_2_3[:25])
axs[1, 2].set_title('hh sec 3 occ 2')


axs[2,0].plot(dN_3_1[:25])
axs[2, 0].set_title('hh sec 1 occ 3')
axs[2,1].plot(dN_3_2[:25])
axs[2, 1].set_title('hh sec 2 occ 3')
axs[2,2].plot(dN_3_3[:25])
axs[2, 2].set_title('hh sec 3 occ 3')
plt.show()


rhos = np.array([0.8])
dG = 0.001 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['G'] @ dG
dN = G['N']['G'] @ dG
di = G['i']['G'] @ dG
dC = G['C']['G'] @ dG

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp government spending shocks')
axs[0,0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0,1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1,0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()

rhos = np.array([0.8])
dG = 0.001 * rhos ** (np.arange(T)[:, np.newaxis])
dC1 = G['C1']['G'] @ dG
dC2 = G['C2']['G'] @ dG
dC3 = G['C3']['G'] @ dG
dC = G['C']['G'] @ dG

fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig.suptitle(r'Output response to 1 pp government spending shocks')
axs[0,0].plot(dC1[:25])
axs[0, 0].set_title('consumption 1 hh')
axs[0,1].plot(dC2[:25])
axs[0, 1].set_title('consumption 2 hh')
axs[1,0].plot(dC3[:25])
axs[1, 0].set_title('consumption 3 hh')
axs[1,1].plot(dC[:25])
axs[1, 1].set_title('consumption aggregate')
plt.show()



plt.plot(ss['a1_grid'][:10], ss['c1'][0,25, :10].T)
plt.xlabel('Assets'), plt.ylabel('Consumption')
plt.show()

plt.plot(ss['a2_grid'][:10], ss['c2'][2,25, :10].T)
plt.xlabel('Assets'), plt.ylabel('Consumption')
plt.show()
