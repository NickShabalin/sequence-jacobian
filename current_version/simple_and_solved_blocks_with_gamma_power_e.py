from simple_block import simple, simple_with_vector_args
from solved_block import solved
import numpy as np

@simple
def taylor(rstar, pi, phi):
    i = rstar + phi * pi
    return i


@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax


@simple
def finance(i, pi, r, div_sec_1, div_sec_2, div_sec_3, omega, pshare_1, pshare_2, pshare_3, equity_price_sec_1,
            equity_price_sec_2, equity_price_sec_3):
    rb = r - omega
    ra = pshare_1 * (div_sec_1 + equity_price_sec_1) / equity_price_sec_1(-1) + pshare_2 * (div_sec_2 + equity_price_sec_2) / equity_price_sec_2(-1) + pshare_3 * (div_sec_3 + equity_price_sec_3) / equity_price_sec_3(-1) + (1 - pshare_1 - pshare_2 - pshare_3) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher


@simple_with_vector_args({"occ_1_1": 3, "occ_1_2": 3, "occ_1_3": 3, "occ_2_1": 3, "occ_2_2": 3, "occ_2_3": 3, "occ_3_1": 3, "occ_3_2": 3, "occ_3_3": 3, "occupation_vector_1_1": 3, "q": 3, "gamma_hh_1": 3, "gamma_hh_2": 3, "gamma_hh_3": 3, "w_occ": 3, "N_hh_occ_1": 3, "N_hh_occ_2": 3, "N_hh_occ_3": 3, "possible_occupation": 3, "labor_supply_1": 3, "labor_supply_2": 3, "labor_supply_3": 3, "pi_distribution": 3, "z_grid_1_1": 3, "z_grid_1_2": 3, "z_grid_1_3": 3, "z_grid_2_1": 3, "z_grid_2_2": 3, "z_grid_2_3": 3, "z_grid_3_1": 3, "z_grid_3_2": 3, "z_grid_3_3": 3})
def labor_supply(Y, q, pi_distribution, possible_occupation, m1, m2, m3, gamma_hh_1, gamma_hh_2, gamma_hh_3, w_occ, N_hh_occ_1, N_hh_occ_2, N_hh_occ_3, e_grid_1, e_grid_2, e_grid_3, infected):
    q_1 = 1 - Y * infected * 0.82
    q_2 = 1 - Y * infected * 0.9
    q_3 = 1 - Y * infected * 0.927
    z_grid_1_1 = (1 + gamma_hh_1) ** e_grid_1 * w_occ * N_hh_occ_1 * q
    z_grid_1_2 = (1 + gamma_hh_1) ** e_grid_2 * w_occ * N_hh_occ_1 * q
    z_grid_1_3 = (1 + gamma_hh_1) ** e_grid_3 * w_occ * N_hh_occ_1 * q
    z_grid_2_1 = (1 + gamma_hh_2) ** e_grid_1 * w_occ * N_hh_occ_2 * q
    z_grid_2_2 = (1 + gamma_hh_2) ** e_grid_2 * w_occ * N_hh_occ_2 * q
    z_grid_2_3 = (1 + gamma_hh_2) ** e_grid_3 * w_occ * N_hh_occ_2 * q
    z_grid_3_1 = (1 + gamma_hh_3) ** e_grid_1 * w_occ * N_hh_occ_3 * q
    z_grid_3_2 = (1 + gamma_hh_3) ** e_grid_2 * w_occ * N_hh_occ_3 * q
    z_grid_3_3 = (1 + gamma_hh_3) ** e_grid_3 * w_occ * N_hh_occ_3 * q
    all_grids_1_1 = [z_grid_1_1_1, z_grid_1_1_2, z_grid_1_1_3]
    all_grids_1_2 = [z_grid_1_2_1, z_grid_1_2_2, z_grid_1_2_3]
    all_grids_1_3 = [z_grid_1_3_1, z_grid_1_3_2, z_grid_1_3_3]
    occupation_1_1 = np.argmax(all_grids_1_1, axis=0)
    occupation_1_2 = np.argmax(all_grids_1_2, axis=0)
    occupation_1_3 = np.argmax(all_grids_1_3, axis=0)
    all_grids_2_1 = [z_grid_2_1_1, z_grid_2_1_2, z_grid_2_1_3]
    all_grids_2_2 = [z_grid_2_2_1, z_grid_2_2_2, z_grid_2_2_3]
    all_grids_2_3 = [z_grid_2_3_1, z_grid_2_3_2, z_grid_2_3_3]
    occupation_2_1 = np.argmax(all_grids_2_1, axis=0)
    occupation_2_2 = np.argmax(all_grids_2_2, axis=0)
    occupation_2_3 = np.argmax(all_grids_2_3, axis=0)
    all_grids_3_1 = [z_grid_3_1_1, z_grid_3_1_2, z_grid_3_1_3]
    all_grids_3_2 = [z_grid_3_2_1, z_grid_3_2_2, z_grid_3_2_3]
    all_grids_3_3 = [z_grid_3_3_1, z_grid_3_3_2, z_grid_3_3_3]
    occupation_3_1 = np.argmax(all_grids_3_1, axis=0)
    occupation_3_2 = np.argmax(all_grids_3_2, axis=0)
    occupation_3_3 = np.argmax(all_grids_3_3, axis=0)
    occupations1 = np.zeros(3)
    occupations1[occupation_1_1] = 1
    occ_1_1_1 = occupations1[0]
    occ_1_1_2 = occupations1[1]
    occ_1_1_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_1_2] = 1
    occ_1_2_1 = occupations1[0]
    occ_1_2_2 = occupations1[1]
    occ_1_2_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_1_3] = 1
    occ_1_3_1 = occupations1[0]
    occ_1_3_2 = occupations1[1]
    occ_1_3_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_2_1] = 1
    occ_2_1_1 = occupations1[0]
    occ_2_1_2 = occupations1[1]
    occ_2_1_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_2_2] = 1
    occ_2_2_1 = occupations1[0]
    occ_2_2_2 = occupations1[1]
    occ_2_2_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_2_3] = 1
    occ_2_3_1 = occupations1[0]
    occ_2_3_2 = occupations1[1]
    occ_2_3_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_3_1] = 1
    occ_3_1_1 = occupations1[0]
    occ_3_1_2 = occupations1[1]
    occ_3_1_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_3_2] = 1
    occ_3_2_1 = occupations1[0]
    occ_3_2_2 = occupations1[1]
    occ_3_2_3 = occupations1[2]
    occupations1 = np.zeros(3)
    occupations1[occupation_3_3] = 1
    occ_3_3_1 = occupations1[0]
    occ_3_3_2 = occupations1[1]
    occ_3_3_3 = occupations1[2]
    labor_supply_1_1 = (pi_distribution_1 * (1 + gamma_hh_1) ** e_grid_1) * m1 * N_hh_occ_1 * q * occ_1_1
    labor_supply_1_2 = (pi_distribution_2 * (1 + gamma_hh_1) ** e_grid_2) * m1 * N_hh_occ_1 * q * occ_1_2
    labor_supply_1_3 = (pi_distribution_3 * (1 + gamma_hh_1) ** e_grid_3) * m1 * N_hh_occ_1 * q * occ_1_3
    labor_supply_2_1 = (pi_distribution_1 * (1 + gamma_hh_2) ** e_grid_1) * m2 * N_hh_occ_2 * q * occ_2_1
    labor_supply_2_2 = (pi_distribution_2 * (1 + gamma_hh_2) ** e_grid_2) * m2 * N_hh_occ_2 * q * occ_2_2
    labor_supply_2_3 = (pi_distribution_3 * (1 + gamma_hh_2) ** e_grid_3) * m2 * N_hh_occ_2 * q * occ_2_3
    labor_supply_3_1 = (pi_distribution_1 * (1 + gamma_hh_3) ** e_grid_1) * m3 * N_hh_occ_3 * q * occ_3_1
    labor_supply_3_2 = (pi_distribution_2 * (1 + gamma_hh_3) ** e_grid_2) * m3 * N_hh_occ_3 * q * occ_3_2
    labor_supply_3_3 = (pi_distribution_3 * (1 + gamma_hh_3) ** e_grid_3) * m3 * N_hh_occ_3 * q * occ_3_3
    N_occ_1 = labor_supply_1_1_1 + labor_supply_1_2_1 + labor_supply_1_3_1 + labor_supply_2_1_1 + labor_supply_2_2_1 + labor_supply_2_3_1 + labor_supply_3_1_1 + labor_supply_3_2_1 + labor_supply_3_3_1
    N_occ_2 = labor_supply_1_1_2 + labor_supply_1_2_2 + labor_supply_1_3_2 + labor_supply_2_1_2 + labor_supply_2_2_2 + labor_supply_2_3_2 + labor_supply_3_1_2 + labor_supply_3_2_2 + labor_supply_3_3_2
    N_occ_3 = labor_supply_1_1_3 + labor_supply_1_2_3 + labor_supply_1_3_3 + labor_supply_2_1_3 + labor_supply_2_2_3 + labor_supply_2_3_3 + labor_supply_3_1_3 + labor_supply_3_2_3 + labor_supply_3_3_3
    return z_grid_1_1, z_grid_1_2, z_grid_1_3, z_grid_2_1, z_grid_2_2, z_grid_2_3, z_grid_3_1, z_grid_3_2, z_grid_3_3, N_occ_1, N_occ_2, N_occ_3, labor_supply_1_1, labor_supply_1_2, labor_supply_1_3, labor_supply_2_1, labor_supply_2_2, labor_supply_2_3, labor_supply_3_1, labor_supply_3_2, labor_supply_3_3



@simple_with_vector_args({"div_sec": 3, "equity_price_sec": 3})
def arbitrage(div_sec, equity_price_sec, r):
    equity = div_sec(+1) + equity_price_sec(+1) - equity_price_sec * (1 + r(+1))
    return equity


@simple_with_vector_args({"Y_sec": 3, "K_sec": 3, "w_sec": 3, "N_sec": 3, "p_sec": 3})
def dividend(Y_sec, w_sec, N_sec, K_sec, p_sec, pi, eta, kappap, delta):
    psip_sec = eta / 2 / kappap * np.log(1 + pi) ** 2 * Y_sec
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
def asset_mkt_clearing(equity_price_sec_1, equity_price_sec_2, equity_price_sec_3, I, C, A, B, psip, ra, rb, mup, Y, r, delta, K, Bg, G, w, N, tax, gamma_hh_1_2, gamma_hh_1_1, gamma_hh_2_3, gamma_hh_2_2, gamma_hh_3_2, gamma_hh_3_3, e_grid_1, e_grid_2, e_grid_3, w_occ_1, w_occ_2, w_occ_3, q_1, q_2, q_3, N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_2_2, N_hh_occ_2_3, N_hh_occ_3_2, N_hh_occ_3_3, m1, m2, m3):
    z_grid_1_1_2 = (1 + gamma_hh_1_2) ** e_grid_1 * w_occ_2 * N_hh_occ_1_2 * q_2
    z_grid_1_2_2 = (1 + gamma_hh_1_2) ** e_grid_2 * w_occ_2 * N_hh_occ_1_2 * q_2
    z_grid_1_3_1 = (1 + gamma_hh_1_1) ** e_grid_3 * w_occ_1 * N_hh_occ_1_1 * q_1
    z_grid_2_1_3 = (1 + gamma_hh_2_3) ** e_grid_1 * w_occ_3 * N_hh_occ_2_3 * q_3
    z_grid_2_2_2 = (1 + gamma_hh_2_2) ** e_grid_2 * w_occ_2 * N_hh_occ_2_2 * q_2
    z_grid_2_3_2 = (1 + gamma_hh_2_2) ** e_grid_3 * w_occ_2 * N_hh_occ_2_2 * q_2
    z_grid_3_1_2 = (1 + gamma_hh_3_2) ** e_grid_1 * w_occ_2 * N_hh_occ_3_2 * q_2
    z_grid_3_2_2 = (1 + gamma_hh_3_2) ** e_grid_2 * w_occ_2 * N_hh_occ_3_2 * q_2
    z_grid_3_3_3 = (1 + gamma_hh_3_3) ** e_grid_3 * w_occ_3 * N_hh_occ_3_3 * q_3
    income_all_hh = z_grid_1_1_2 * 0.25 * m1 + z_grid_1_2_2 * 0.5 * m1 + z_grid_1_3_1 * 0.25 * m1 + z_grid_2_1_3 * 0.25 * m2 + z_grid_2_2_2 * 0.5 * m2 + z_grid_2_3_2 * 0.25 * m2 + z_grid_3_1_2 * 0.25 * m3 + z_grid_3_2_2 * 0.5 * m3 + z_grid_3_3_3 * 0.25 * m3
    asset_mkt = equity_price_sec_1 + equity_price_sec_2 + equity_price_sec_3 + Bg - A - B
    #asset_mkt = w * N  - income_all_hh
    return asset_mkt


'''
@simple
def asset_mkt_clearing(equity_price_sec_1, equity_price_sec_2, equity_price_sec_3, A, B, Bg, gamma_hh_1_2, gamma_hh_1_1, gamma_hh_2_3, gamma_hh_2_2, gamma_hh_3_2, gamma_hh_3_3, e_grid_1, e_grid_2, e_grid_3, w_occ_1, w_occ_2, w_occ_3, q_1, q_2, q_3, N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_2_2, N_hh_occ_2_3, N_hh_occ_3_2, N_hh_occ_3_3, m1, m2, m3):
    z_grid_1_1_2 = (1 + gamma_hh_1_2) ** e_grid_1 * w_occ_2 * N_hh_occ_1_2 * q_2
    z_grid_1_2_2 = (1 + gamma_hh_1_2) ** e_grid_2 * w_occ_2 * N_hh_occ_1_2 * q_2
    z_grid_1_3_1 = (1 + gamma_hh_1_1) ** e_grid_3 * w_occ_1 * N_hh_occ_1_1 * q_1
    z_grid_2_1_3 = (1 + gamma_hh_2_3) ** e_grid_1 * w_occ_3 * N_hh_occ_2_3 * q_3
    z_grid_2_2_2 = (1 + gamma_hh_2_2) ** e_grid_2 * w_occ_2 * N_hh_occ_2_2 * q_2
    z_grid_2_3_2 = (1 + gamma_hh_2_2) ** e_grid_3 * w_occ_2 * N_hh_occ_2_2 * q_2
    z_grid_3_1_2 = (1 + gamma_hh_3_2) ** e_grid_1 * w_occ_2 * N_hh_occ_3_2 * q_2
    z_grid_3_2_2 = (1 + gamma_hh_3_2) ** e_grid_2 * w_occ_2 * N_hh_occ_3_2 * q_2
    z_grid_3_3_3 = (1 + gamma_hh_3_3) ** e_grid_3 * w_occ_3 * N_hh_occ_3_3 * q_3
    income_all_hh = z_grid_1_1_2 * 0.25 * m1 + z_grid_1_2_2 * 0.5 * m1 + z_grid_1_3_1 * 0.25 * m1 + z_grid_2_1_3 * 0.25 * m2 + z_grid_2_2_2 * 0.5 * m2 + z_grid_2_3_2 * 0.25 * m2 + z_grid_3_1_2 * 0.25 * m3 + z_grid_3_2_2 * 0.5 * m3 + z_grid_3_3_3 * 0.25 * m3
    asset_mkt = equity_price_sec_1 + equity_price_sec_2 + equity_price_sec_3 + Bg - income_all_hh * (1 - tax)
    return asset_mkt
'''

@simple
def labor_market_clearing(N_occ_sec_1_1, N_occ_sec_1_2, N_occ_sec_1_3,
                          N_occ_sec_2_1, N_occ_sec_2_2, N_occ_sec_2_3,
                          N_occ_sec_3_1, N_occ_sec_3_2, N_occ_sec_3_3,
                          N_occ_1, N_occ_2, N_occ_3):

    labor_market_clearing_1 = N_occ_sec_1_1 + N_occ_sec_1_2 + N_occ_sec_1_3 - N_occ_1
    labor_market_clearing_2 = N_occ_sec_2_1 + N_occ_sec_2_2 + N_occ_sec_2_3 - N_occ_2
    labor_market_clearing_3 = N_occ_sec_3_1 + N_occ_sec_3_2 + N_occ_sec_3_3 - N_occ_3

    return labor_market_clearing_1, labor_market_clearing_2, labor_market_clearing_3




@simple_with_vector_args({"p_sec": 3, "mc_sec": 3, "f_sec": 3})
def pricing(mc_sec, r, Y, kappap, mup, eta, p_sec, p, f_sec):
    nkpc_sec = kappap * (p_sec / p) ** (- eta) * f_sec * (mc_sec - 1 / mup * p_sec / p) + Y(+1) / Y * np.log(p_sec(+1) / p_sec) / (1 + r(+1)) - np.log(p_sec / p_sec(-1))
    return nkpc_sec


@simple_with_vector_args({"K_sec": 3, "Q_sec": 3, "productivity_sec": 3, "L_sec": 3, "p_sec": 3, "nu_sec": 3, "mc_sec": 3})
def investment(Q_sec, K_sec, r, L_sec, productivity_sec, delta, epsI, nu_sec, mc_sec):
    inv_sec = (K_sec / K_sec(-1) - 1) / (delta * epsI) + 1 - Q_sec
    val_sec = nu_sec * productivity_sec(+1) * (L_sec(+1) / K_sec) ** (1 - nu_sec) * mc_sec(+1) - (K_sec(+1) / K_sec - (1 - delta) + (K_sec(+1) / K_sec - 1) ** 2 / (2 * delta * epsI)) + K_sec(+1) / K_sec * Q_sec(+1) - (1 + r(+1)) * Q_sec
    return inv_sec, val_sec


@simple
def output_aggregation(Y_sec_1, Y_sec_2, Y_sec_3, eta, f_sec_1, f_sec_2, f_sec_3):
    power_y = eta / (eta - 1)
    Y = (f_sec_1 ** (1 / eta) * Y_sec_1 ** (1 / power_y) + f_sec_2 ** (1 / eta) * Y_sec_2 ** (1 / power_y) + f_sec_3 ** (1 / eta) * Y_sec_3 ** (1 / power_y)) ** power_y
    return Y

@simple_with_vector_args({"productivity_sec": 3, "L_sec": 3, "nu_sec": 3, "K_sec": 3, "Y_sec": 3, "mc_sec": 3, "sigma_sec": 3, "alpha_occ_sec_1": 3, "alpha_occ_sec_2": 3, "alpha_occ_sec_3": 3, "N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3})
def production_sec(productivity_sec, L_sec, nu_sec, K_sec, mc_sec, alpha_occ_sec_1, alpha_occ_sec_2, alpha_occ_sec_3, Y_sec, w_occ_1, w_occ_2, w_occ_3, sigma_sec, N_occ_sec_1, N_occ_sec_2, N_occ_sec_3):
    prod_sec = productivity_sec * L_sec ** (1 - nu_sec) * K_sec(-1) ** nu_sec - Y_sec
    labor_sec_occ_1 = mc_sec * (1 - nu_sec) * alpha_occ_sec_1 * Y_sec / L_sec * (L_sec / N_occ_sec_1) ** (1 - sigma_sec) - w_occ_1
    labor_sec_occ_2 = mc_sec * (1 - nu_sec) * alpha_occ_sec_2 * Y_sec / L_sec * (L_sec / N_occ_sec_2) ** (1 - sigma_sec) - w_occ_2
    labor_sec_occ_3 = mc_sec * (1 - nu_sec) * alpha_occ_sec_3 * Y_sec / L_sec * (L_sec / N_occ_sec_3) ** (1 - sigma_sec) - w_occ_3
    return prod_sec, labor_sec_occ_1, labor_sec_occ_2, labor_sec_occ_3


@simple
def wage_labor_aggregates(w_occ_1, w_occ_2, w_occ_3,
                          N_occ_sec_1_1, N_occ_sec_1_2, N_occ_sec_1_3,
                          N_occ_sec_2_1, N_occ_sec_2_2, N_occ_sec_2_3,
                          N_occ_sec_3_1, N_occ_sec_3_2, N_occ_sec_3_3,
                          sigma_sec_1, sigma_sec_2, sigma_sec_3,
                          alpha_occ_sec_1_1, alpha_occ_sec_1_2, alpha_occ_sec_1_3,
                          alpha_occ_sec_2_1, alpha_occ_sec_2_2, alpha_occ_sec_2_3,
                          alpha_occ_sec_3_1, alpha_occ_sec_3_2, alpha_occ_sec_3_3):

    N_sec_1 = N_occ_sec_1_1 + N_occ_sec_2_1 + N_occ_sec_3_1
    N_sec_2 = N_occ_sec_1_2 + N_occ_sec_2_2 + N_occ_sec_3_2
    N_sec_3 = N_occ_sec_1_3 + N_occ_sec_2_3 + N_occ_sec_3_3
    N = N_sec_1 + N_sec_2 + N_sec_3

    w_sec_1 = (w_occ_1 * N_occ_sec_1_1 + w_occ_2 * N_occ_sec_2_1 + w_occ_3 * N_occ_sec_3_1) / N_sec_1
    w_sec_2 = (w_occ_1 * N_occ_sec_1_2 + w_occ_2 * N_occ_sec_2_2 + w_occ_3 * N_occ_sec_3_2) / N_sec_2
    w_sec_3 = (w_occ_1 * N_occ_sec_1_3 + w_occ_2 * N_occ_sec_2_3 + w_occ_3 * N_occ_sec_3_3) / N_sec_3
    w = (w_sec_1 * N_sec_1 + w_sec_2 * N_sec_2 + w_sec_3 * N_sec_3) / N

    L_sec_1 = (alpha_occ_sec_1_1 * N_occ_sec_1_1 ** sigma_sec_1 + alpha_occ_sec_2_1 * N_occ_sec_2_1 ** sigma_sec_1 + alpha_occ_sec_3_1 * N_occ_sec_3_1 ** sigma_sec_1) ** (1 / sigma_sec_1)
    L_sec_2 = (alpha_occ_sec_1_2 * N_occ_sec_1_2 ** sigma_sec_2 + alpha_occ_sec_2_2 * N_occ_sec_2_2 ** sigma_sec_2 + alpha_occ_sec_3_2 * N_occ_sec_3_2 ** sigma_sec_2) ** (1 / sigma_sec_2)
    L_sec_3 = (alpha_occ_sec_1_3 * N_occ_sec_1_3 ** sigma_sec_3 + alpha_occ_sec_2_3 * N_occ_sec_2_3 ** sigma_sec_3 + alpha_occ_sec_3_3 * N_occ_sec_3_3 ** sigma_sec_3) ** (1 / sigma_sec_3)

    return w, N, w_sec_1, w_sec_2, w_sec_3, N_sec_1, N_sec_2, N_sec_3, L_sec_1, L_sec_2, L_sec_3


@simple_with_vector_args({"Y_sec": 3, "f_sec": 3})
def pricing_intermediate(Y, Y_sec, eta, p, f_sec):
    p_sec = (f_sec * Y / Y_sec) ** (1 / eta) * p
    pi = p / p(-1) - 1
    return p_sec, pi




production = solved(block_list=[output_aggregation, production_sec, investment, labor_market_clearing,
                                wage_labor_aggregates, pricing, pricing_intermediate, labor_supply],
                    unknowns=['Q_sec', 'K_sec', 'Y_sec', 'N_occ_sec_1', 'N_occ_sec_2', 'N_occ_sec_3', 'mc_sec', 'w_occ_1', 'w_occ_2', 'w_occ_3'],
                    targets=['inv_sec', 'val_sec', 'prod_sec', 'labor_sec_occ_1', 'labor_sec_occ_2', 'labor_sec_occ_3', 'nkpc_sec', 'labor_market_clearing_1', 'labor_market_clearing_2', 'labor_market_clearing_3'],
                    vector_arguments={"Q_sec": 3, "K_sec": 3, "Y_sec": 3,
                                      "N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3,
                                      "mc_sec": 3, "inv_sec": 3, "val_sec": 3, "prod_sec": 3,
                                      "labor_sec_occ_1": 3, "labor_sec_occ_2": 3, "labor_sec_occ_3": 3,
                                      "nkpc_sec": 3})


@simple
def consumers_aggregator(C1, C2, C3, A1, A2, A3, B1, B2, B3, U1, U2, U3, m1, m2, m3):
    C = m1 * C1 + m2 * C2 + m3 * C3
    A = m1 * A1 + m2 * A2 + m3 * A3
    B = m1 * B1 + m2 * B2 + m3 * B3
    U = m1 * U1 + m2 * U2 + m3 * U3
    return C, A, B, U


@simple
def sir_block(susceptible, infected, recovered, covid_shock, beta_sir, gamma_sir):
    sus_eq = susceptible - (1 - beta_sir * infected(-1) / (infected(-1) + recovered(-1) + susceptible(-1))
                            ) * susceptible(-1) + covid_shock

    inf_eq = infected - (1 - gamma_sir + beta_sir * susceptible(-1) / (infected(-1) + recovered(-1) + susceptible(-1))
                         ) * infected(-1) - covid_shock

    rec_eq = recovered - recovered(-1) - gamma_sir * infected(-1)

    # q_1 = 1 - infected * 0.5
    # q_2 = 1 - infected * 0.2
    # q_3 = 1 - infected
    #q = 1

    return sus_eq, inf_eq, rec_eq