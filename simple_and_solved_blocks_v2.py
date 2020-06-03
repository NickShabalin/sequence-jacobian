from simple_block import simple, simple_with_vector_args
from solved_block import solved
import math
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


@simple
def mkt_clearing(equity_price_sec_1, equity_price_sec_2, equity_price_sec_3, frisch,
                 A, B, Bg, vphi_1, vphi_2, vphi_3, tax, U1, U2, U3, p,
                 N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3,
                 N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3,
                 N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3,
                 gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3,
                 gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3,
                 gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3,
                 w_occ_1, w_occ_2, w_occ_3,
                 beta_sir, infected, susceptible, recovered,
                 occupation_mult_for_covid_1, occupation_mult_for_covid_2, occupation_mult_for_covid_3):

    asset_mkt = equity_price_sec_1 + equity_price_sec_2 + equity_price_sec_3 + Bg - B - A

    N_hh_occ_1 = [N_hh_occ_1_1, N_hh_occ_1_2, N_hh_occ_1_3]
    N_hh_occ_2 = [N_hh_occ_2_1, N_hh_occ_2_2, N_hh_occ_2_3]
    N_hh_occ_3 = [N_hh_occ_3_1, N_hh_occ_3_2, N_hh_occ_3_3]
    gamma_hh_1 = [gamma_hh_1_1, gamma_hh_1_2, gamma_hh_1_3]
    gamma_hh_2 = [gamma_hh_2_1, gamma_hh_2_2, gamma_hh_2_3]
    gamma_hh_3 = [gamma_hh_3_1, gamma_hh_3_2, gamma_hh_3_3]
    w_occ = [w_occ_1, w_occ_2, w_occ_3]

    # different covid multipliers
    #covid_mult = math.exp(beta_sir * infected(-1) / (infected(-1) + susceptible(-1) + recovered(-1)))
    covid_mult = (1 + beta_sir * infected / (infected + susceptible + recovered))
    #covid_mult2 = (1 + 2 * beta_sir * infected / (infected + susceptible + recovered))
    #covid_mult = 1
    # covid_mult_1 = (1 + 3* beta_sir * infected / (infected + susceptible + recovered))
    # covid_mult_2 = (1 + 1.5 * beta_sir * infected / (infected + susceptible + recovered))
    # covid_mult_3 = (1 + 1 * beta_sir * infected / (infected + susceptible + recovered))
    covid_mult_1 = (1 + occupation_mult_for_covid_1 * beta_sir * infected * max(N_hh_occ_1) / (infected + susceptible + recovered))
    covid_mult_2 = (1 + occupation_mult_for_covid_2 * beta_sir * infected * max(N_hh_occ_2) / (infected + susceptible + recovered))
    covid_mult_3 = (1 + occupation_mult_for_covid_3 * beta_sir * infected * max(N_hh_occ_3) / (infected + susceptible + recovered))


    intratemp_hh_1 = covid_mult_1 * vphi_1 * max(N_hh_occ_1) ** (1 / frisch) - (1 - tax) * w_occ[
        N_hh_occ_1.index(max(N_hh_occ_1))] * U1 * gamma_hh_1[N_hh_occ_1.index(max(N_hh_occ_1))] / p
    intratemp_hh_2 = covid_mult_2 * vphi_2 * max(N_hh_occ_2) ** (1 / frisch) - (1 - tax) * w_occ[
        N_hh_occ_2.index(max(N_hh_occ_2))] * U2 * gamma_hh_2[N_hh_occ_2.index(max(N_hh_occ_2))] / p
    intratemp_hh_3 = covid_mult_3 * vphi_3 * max(N_hh_occ_3) ** (1 / frisch) - (1 - tax) * w_occ[
        N_hh_occ_3.index(max(N_hh_occ_3))] * U3 * gamma_hh_3[N_hh_occ_3.index(max(N_hh_occ_3))] / p

    return asset_mkt, intratemp_hh_1, intratemp_hh_2, intratemp_hh_3


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




@simple_with_vector_args({"p_sec": 3, "mc_sec": 3})
def pricing(mc_sec, r, Y, kappap, mup, eta, p_sec, p):
    nkpc_sec = kappap * (p_sec / p) ** (- eta) * (mc_sec - 1 / mup * p_sec / p) + Y(+1) / Y * np.log(p_sec(+1) / p_sec) / (1 + r(+1)) - np.log(p_sec / p_sec(-1))
    return nkpc_sec


@simple_with_vector_args({"K_sec": 3, "Q_sec": 3, "productivity_sec": 3, "L_sec": 3, "p_sec": 3, "nu_sec": 3, "mc_sec": 3})
def investment(Q_sec, K_sec, r, L_sec, productivity_sec, delta, epsI, nu_sec, mc_sec):
    inv_sec = (K_sec / K_sec(-1) - 1) / (delta * epsI) + 1 - Q_sec
    val_sec = nu_sec * productivity_sec(+1) * (L_sec(+1) / K_sec) ** (1 - nu_sec) * mc_sec(+1) - (K_sec(+1) / K_sec - (1 - delta) + (K_sec(+1) / K_sec - 1) ** 2 / (2 * delta * epsI)) + K_sec(+1) / K_sec * Q_sec(+1) - (1 + r(+1)) * Q_sec
    return inv_sec, val_sec


@simple
def output_aggregation(Y_sec_1, Y_sec_2, Y_sec_3, eta):
    power_y = eta / (eta - 1)
    # Y = (Y_sec_1 ** (1 / power_y) + Y_sec_2 ** (1 / power_y) + Y_sec_3 ** (1 / power_y)) ** power_y
    Y = Y_sec_1 * Y_sec_2 * Y_sec_3
    return Y


@simple_with_vector_args({"productivity_sec": 3, "L_sec": 3, "nu_sec": 3, "K_sec": 3, "Y_sec": 3, "p_sec": 3, "sigma_sec": 3, "alpha_occ_sec_1": 3, "alpha_occ_sec_2": 3, "alpha_occ_sec_3": 3, "N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3})
def production_sec(productivity_sec, L_sec, nu_sec, K_sec, Y_sec, p_sec, alpha_occ_sec_1, alpha_occ_sec_2, alpha_occ_sec_3, w_occ_1, w_occ_2, w_occ_3, sigma_sec, N_occ_sec_1, N_occ_sec_2, N_occ_sec_3, p):
    prod_sec = productivity_sec * L_sec ** (1 - nu_sec) * K_sec(-1) ** nu_sec - Y_sec
    # labor_sec_occ_1 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_1 / w_occ_1 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_1 * (1 - sigma_sec)
    # labor_sec_occ_2 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_2 / w_occ_2 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_2 * (1 - sigma_sec)
    # labor_sec_occ_3 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_3 / w_occ_3 * (productivity_sec * K_sec(-1) ** nu_sec) ** (sigma_sec / (1 - nu_sec)) * Y_sec ** ((- sigma_sec + 1 - nu_sec) / (1 - nu_sec)) - N_occ_sec_3 * (1 - sigma_sec)
    labor_sec_occ_1 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_1 / w_occ_1 * Y_sec - N_occ_sec_1
    labor_sec_occ_2 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_2 / w_occ_2 * Y_sec - N_occ_sec_2
    labor_sec_occ_3 = p_sec / p * (1 - nu_sec) * alpha_occ_sec_3 / w_occ_3 * Y_sec - N_occ_sec_3
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

    # L_sec_1 = (alpha_occ_sec_1_1 * N_occ_sec_1_1 ** sigma_sec_1 + alpha_occ_sec_2_1 * N_occ_sec_2_1 ** sigma_sec_1 + alpha_occ_sec_3_1 * N_occ_sec_3_1 ** sigma_sec_1) ** (1 / sigma_sec_1)
    # L_sec_2 = (alpha_occ_sec_1_2 * N_occ_sec_1_2 ** sigma_sec_2 + alpha_occ_sec_2_2 * N_occ_sec_2_2 ** sigma_sec_2 + alpha_occ_sec_3_2 * N_occ_sec_3_2 ** sigma_sec_2) ** (1 / sigma_sec_2)
    # L_sec_3 = (alpha_occ_sec_1_3 * N_occ_sec_1_3 ** sigma_sec_3 + alpha_occ_sec_2_3 * N_occ_sec_2_3 ** sigma_sec_3 + alpha_occ_sec_3_3 * N_occ_sec_3_3 ** sigma_sec_3) ** (1 / sigma_sec_3)

    L_sec_1 = (N_occ_sec_1_1 ** alpha_occ_sec_1_1 * N_occ_sec_2_1 ** alpha_occ_sec_2_1 * N_occ_sec_3_1 ** alpha_occ_sec_3_1)
    L_sec_2 = (N_occ_sec_1_2 ** alpha_occ_sec_1_2 * N_occ_sec_2_2 ** alpha_occ_sec_2_2 * N_occ_sec_3_2 ** alpha_occ_sec_3_2)
    L_sec_3 = (N_occ_sec_1_3 ** alpha_occ_sec_1_3 * N_occ_sec_2_3 ** alpha_occ_sec_2_3 * N_occ_sec_3_3 ** alpha_occ_sec_3_3)

    # L_sec_1 = (N_occ_sec_1_1 * N_occ_sec_2_1 * N_occ_sec_3_1)
    # L_sec_2 = (N_occ_sec_1_2 * N_occ_sec_2_2 * N_occ_sec_3_2)
    # L_sec_3 = (N_occ_sec_1_3 * N_occ_sec_2_3 * N_occ_sec_3_3)

    return w, N, w_sec_1, w_sec_2, w_sec_3, N_sec_1, N_sec_2, N_sec_3, N_occ_1, N_occ_2, N_occ_3, L_sec_1, L_sec_2, L_sec_3


@simple_with_vector_args({"Y_sec": 3})
def pricing_intermediate(Y, Y_sec, eta, p):
    p_sec = (Y / Y_sec) * p
    pi = p / p(-1) - 1
    return p_sec, pi




production = solved(block_list=[output_aggregation, production_sec, investment,
                                wage_labor_aggregates, pricing, pricing_intermediate],
                    unknowns=['Q_sec', 'K_sec', 'Y_sec', 'N_occ_sec_1', 'N_occ_sec_2', 'N_occ_sec_3', 'mc_sec'],
                    targets=['inv_sec', 'val_sec', 'prod_sec', 'labor_sec_occ_1', 'labor_sec_occ_2', 'labor_sec_occ_3', 'nkpc_sec'],
                    vector_arguments={"Q_sec": 3, "K_sec": 3, "Y_sec": 3,
                                      "N_occ_sec_1": 3, "N_occ_sec_2": 3, "N_occ_sec_3": 3,
                                      "mc_sec": 3, "inv_sec": 3, "val_sec": 3, "prod_sec": 3,
                                      "labor_sec_occ_1": 3, "labor_sec_occ_2": 3, "labor_sec_occ_3": 3,
                                      "nkpc_sec": 3})
'''
production = solved(block_list=[output_aggregation, production_sec, investment,
                                wage_labor_aggregates, pricing, pricing_intermediate],
                    unknowns=['Q_sec_1', 'Q_sec_2', 'Q_sec_3', 'K_sec_1', 'K_sec_2', 'K_sec_3', 'Y_sec_1', 'Y_sec_2', 'Y_sec_3',
                              'N_occ_sec_1_1', 'N_occ_sec_1_2', 'N_occ_sec_1_3', 'N_occ_sec_2_1', 'N_occ_sec_2_2', 'N_occ_sec_2_3',
                              'N_occ_sec_3_1', 'N_occ_sec_3_2', 'N_occ_sec_3_3', 'mc_sec_1', 'mc_sec_2', 'mc_sec_3'],
                    targets=['inv_sec_1', 'inv_sec_2', 'inv_sec_3', 'val_sec_1', 'val_sec_2', 'val_sec_3', 'prod_sec_1', 'prod_sec_2', 'prod_sec_3',
                             'labor_sec_occ_1_1', 'labor_sec_occ_1_2', 'labor_sec_occ_1_3',
                             'labor_sec_occ_2_1', 'labor_sec_occ_2_2', 'labor_sec_occ_2_3',
                             'labor_sec_occ_3_1', 'labor_sec_occ_3_2', 'labor_sec_occ_3_3',
                             'nkpc_sec_1', 'nkpc_sec_2', 'nkpc_sec_3'])
'''
@simple
def consumers_aggregator(C1, C2, C3, A1, A2, A3, B1, B2, B3, U1, U2, U3):
    C = C1 + C2 + C3
    A = A1 + A2 + A3
    B = B1 + B2 + B3
    U = U1 + U2 + U3
    return C, A, B, U


@simple_with_vector_args({"N_occ": 3, "N_raw_1": 3, "N_raw_2": 3, "N_raw_3": 3, "gamma_hh_1": 3, "gamma_hh_2": 3, "gamma_hh_3": 3, "w_occ": 3, 'possible_occupation': 3, "occupation_mult_for_covid": 3})
def occupation_choice(N_occ, gamma_hh_1, gamma_hh_2, gamma_hh_3, m1, m2, m3, tax, frisch, vphi_1, vphi_2, vphi_3, w_occ, possible_occupation, p, infected, susceptible, recovered, beta_sir, occupation_mult_for_covid):
    N_raw_1 = N_occ / m1 / gamma_hh_1
    N_raw_2 = N_occ / m2 / gamma_hh_2
    N_raw_3 = N_occ / m3 / gamma_hh_3
    covid_mult_1 = (1 + occupation_mult_for_covid * beta_sir * infected * N_raw_1 / (infected + susceptible + recovered))
    covid_mult_2 = (1 + occupation_mult_for_covid * beta_sir * infected * N_raw_2 / (infected + susceptible + recovered))
    covid_mult_3 = (1 + occupation_mult_for_covid * beta_sir * infected * N_raw_3 / (infected + susceptible + recovered))
    u_c_1 = covid_mult_1 * vphi_1 * N_raw_1 ** (1 / frisch) / (1 - tax) / w_occ / gamma_hh_1 * p
    u_c_2 = covid_mult_2 * vphi_2 * N_raw_2 ** (1 / frisch) / (1 - tax) / w_occ / gamma_hh_2 * p
    u_c_3 = covid_mult_3 * vphi_3 * N_raw_3 ** (1 / frisch) / (1 - tax) / w_occ / gamma_hh_3 * p
    u_c_vector_1 = [u_c_1_1, u_c_1_2, u_c_1_3]
    u_c_vector_2 = [u_c_2_1, u_c_2_2, u_c_2_3]
    u_c_vector_3 = [u_c_3_1, u_c_3_2, u_c_3_3]
    occupation_1 = u_c_vector_1.index(min(u_c_vector_1))
    occupation_2 = u_c_vector_2.index(min(u_c_vector_2))
    occupation_3 = u_c_vector_3.index(min(u_c_vector_3))
    N_hh_occ_1 = (occupation_1 == possible_occupation) * N_raw_1
    N_hh_occ_2 = (occupation_2 == possible_occupation) * N_raw_2
    N_hh_occ_3 = (occupation_3 == possible_occupation) * N_raw_3
    return N_hh_occ_1, N_hh_occ_2, N_hh_occ_3, u_c_1, u_c_2, u_c_3


@simple
def sir_block(susceptible, infected, recovered, covid_shock, beta_sir, gamma_sir):
    sus_eq = susceptible - (1 - beta_sir * infected(-1) / (infected(-1) + recovered(-1) + susceptible(-1))
                            ) * susceptible(-1) + covid_shock

    inf_eq = infected - (1 - gamma_sir + beta_sir * susceptible(-1) / (infected(-1) + recovered(-1) + susceptible(-1))
                         ) * infected(-1) - covid_shock

    rec_eq = recovered - recovered(-1) - gamma_sir * infected(-1)

    return sus_eq, inf_eq, rec_eq