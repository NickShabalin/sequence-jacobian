import numpy as np
import matplotlib.pyplot as plt

import jacobian as jac
import household_blocks
import determinacy as det
from steady_state_v3 import hank_ss
import nonlinear

from simple_and_solved_blocks import taylor, fiscal, finance, mkt_clearing, arbitrage, \
                                        dividend, dividend_agg, production, consumers_aggregator, \
                                        occupation_choice, sir_block

# calculate steady state
ss = hank_ss()

# assign variables for A and G calculation
T = 300
block_list = [consumers_aggregator, household_blocks.household_inc1, household_blocks.household_inc2,
              household_blocks.household_inc3, dividend, arbitrage, dividend_agg,
              taylor, fiscal, finance, mkt_clearing, production, occupation_choice, sir_block]
exogenous = ['rstar', 'G', 'covid_shock', 'productivity_sec_1', 'productivity_sec_2', 'productivity_sec_3']
unknowns = ['r', 'p', 'w_occ_1', 'w_occ_2', 'w_occ_3', 'susceptible', 'recovered', 'infected', 'equity_price_sec_1',
            'equity_price_sec_2', 'equity_price_sec_3']
targets = ['asset_mkt', 'fisher', 'intratemp_hh_1', 'intratemp_hh_2', 'intratemp_hh_3', 'sus_eq', 'inf_eq', 'rec_eq',
           'equity_1', 'equity_2', 'equity_3']

# calculate A and G
# A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)
# wn = det.winding_criterion(A)
# print(f'Winding number: {wn}')
#
# G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=True)

# rhos = np.array([0.8])
# drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])

rhos = np.array([0.0, 0.1])
dcovid = 0.000001 * rhos ** (np.arange(T)[:, np.newaxis])

td_nonlin = nonlinear.td_solve(ss, block_list, unknowns, targets,
                               covid_shock=ss['covid_shock']+dcovid[:,0], use_saved=False)



##################################################################################################################
# plot results
# create plot style
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('summer')
plt.rcParams["figure.titlesize"] = 'xx-large'

# graphs: mp shock
rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['rstar'] @ drstar
dN = G['N']['rstar'] @ drstar
di = G['i']['rstar'] @ drstar
dC = G['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 25 bp natural interest rate shock (easing)')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()

rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dC1 = G['C1']['rstar'] @ drstar
dC2 = G['C2']['rstar'] @ drstar
dC3 = G['C3']['rstar'] @ drstar
dC = G['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 25 bp natural interest rate shock (easing)')
axs[0, 0].plot(dC1[:25])
axs[0, 0].set_title('consumption 1 hh')
axs[0, 1].plot(dC2[:25])
axs[0, 1].set_title('consumption 2 hh')
axs[1, 0].plot(dC3[:25])
axs[1, 0].set_title('consumption 3 hh')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('consumption aggregate')
plt.show()

# graphs: productivity shock in sec 1
rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec_1 = -0.07 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_1'] @ dproductivity_sec_1
dN = G['N']['productivity_sec_1'] @ dproductivity_sec_1
di = G['i']['productivity_sec_1'] @ dproductivity_sec_1
dC = G['C']['productivity_sec_1'] @ dproductivity_sec_1

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 1')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_1'] @ dproductivity_sec_1

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 1')
axs[0, 0].plot(dN_1_1[:25])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:25])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:25])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:25])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:25])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:25])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:25])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:25])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:25])
axs[2, 2].set_title('N sec 3 occ 3')
plt.show()

# graphs: productivity shock in sec 2
rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec_2 = -0.04 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_2'] @ dproductivity_sec_2
dN = G['N']['productivity_sec_2'] @ dproductivity_sec_2
di = G['i']['productivity_sec_2'] @ dproductivity_sec_2
dC = G['C']['productivity_sec_2'] @ dproductivity_sec_2

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 2')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_2'] @ dproductivity_sec_2

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 2')
axs[0, 0].plot(dN_1_1[:25])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:25])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:25])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:25])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:25])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:25])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:25])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:25])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:25])
axs[2, 2].set_title('N sec 3 occ 3')
plt.show()

# graphs: productivity shock in sec 3
rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec_3 = -0.06 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['productivity_sec_3'] @ dproductivity_sec_3
dN = G['N']['productivity_sec_3'] @ dproductivity_sec_3
di = G['i']['productivity_sec_3'] @ dproductivity_sec_3
dC = G['C']['productivity_sec_3'] @ dproductivity_sec_3

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 3')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()


dN_1_1 = G['N_occ_sec_1_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_1_2 = G['N_occ_sec_1_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_1_3 = G['N_occ_sec_1_3']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_1 = G['N_occ_sec_2_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_2 = G['N_occ_sec_2_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_3 = G['N_occ_sec_2_3']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_1 = G['N_occ_sec_3_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_2 = G['N_occ_sec_3_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_3 = G['N_occ_sec_3_3']['productivity_sec_3'] @ dproductivity_sec_3

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp negative productivity shock in sector 3')
axs[0, 0].plot(dN_1_1[:25])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:25])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:25])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:25])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:25])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:25])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:25])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:25])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:25])
axs[2, 2].set_title('N sec 3 occ 3')
plt.show()
'''
du_c_1_1 = G['u_c_number_1_1']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_1_1
du_c_1_2 = G['u_c_number_1_2']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_1_2
du_c_1_3 = G['u_c_number_1_3']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_1_3
du_c_2_1 = G['u_c_number_2_1']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_2_1
du_c_2_2 = G['u_c_number_2_2']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_2_2
du_c_2_3 = G['u_c_number_2_3']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_2_3
du_c_3_1 = G['u_c_number_3_1']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_3_1
du_c_3_2 = G['u_c_number_3_2']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_3_2
du_c_3_3 = G['u_c_number_3_3']['productivity_sec_3'] @ dproductivity_sec_3 + utility_number_3_3

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Disutility responses to covid shock')
axs[0, 0].plot(du_c_1_1[:25], label = "first occupation")
axs[0, 0].plot(du_c_1_2[:25], label = "second occupation")
axs[0, 0].plot(du_c_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(du_c_2_1[:25], label = "first occupation")
axs[0, 1].plot(du_c_2_2[:25], label = "second occupation")
axs[0, 1].plot(du_c_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(du_c_3_1[:25], label = "first occupation")
axs[1, 0].plot(du_c_3_2[:25], label = "second occupation")
axs[1, 0].plot(du_c_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()
'''
# graphs: government spending shock
rhos = np.array([0.8])
dG = 0.001 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['G'] @ dG
dN = G['N']['G'] @ dG
di = G['i']['G'] @ dG
dC = G['C']['G'] @ dG

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp government spending shocks')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()

rhos = np.array([0.8])
dG = 0.001 * rhos ** (np.arange(T)[:, np.newaxis])
dC1 = G['C1']['G'] @ dG
dC2 = G['C2']['G'] @ dG
dC3 = G['C3']['G'] @ dG
dC = G['C']['G'] @ dG

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 1 pp government spending shocks')
axs[0, 0].plot(dC1[:25])
axs[0, 0].set_title('consumption 1 hh')
axs[0, 1].plot(dC2[:25])
axs[0, 1].set_title('consumption 2 hh')
axs[1, 0].plot(dC3[:25])
axs[1, 0].set_title('consumption 3 hh')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('consumption aggregate')
plt.show()

# graphs: covid shock
rhos = np.array([0.0])
dcovid = 0.000001 * rhos ** (np.arange(T)[:, np.newaxis])
#dcovid = 0.1 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['covid_shock'] @ dcovid
dN = G['N']['covid_shock'] @ dcovid
di = G['i']['covid_shock'] @ dcovid
dC = G['C']['covid_shock'] @ dcovid

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dY[:25])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:25])
axs[0, 1].set_title('CB rate')
axs[1, 0].plot(dN[:25])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('Consumption')
plt.show()

# steady state values calculation
N_raw_1_1 = ss["N_occ_1"] / ss["m1"] / ss["gamma_hh_1_1"]
N_raw_1_2 = ss["N_occ_2"] / ss["m1"] / ss["gamma_hh_1_2"]
N_raw_1_3 = ss["N_occ_3"] / ss["m1"] / ss["gamma_hh_1_3"]
N_raw_2_1 = ss["N_occ_1"] / ss["m2"] / ss["gamma_hh_2_1"]
N_raw_2_2 = ss["N_occ_2"] / ss["m2"] / ss["gamma_hh_2_2"]
N_raw_2_3 = ss["N_occ_3"] / ss["m2"] / ss["gamma_hh_2_3"]
N_raw_3_1 = ss["N_occ_1"] / ss["m3"] / ss["gamma_hh_3_1"]
N_raw_3_2 = ss["N_occ_2"] / ss["m3"] / ss["gamma_hh_3_2"]
N_raw_3_3 = ss["N_occ_3"] / ss["m3"] / ss["gamma_hh_3_3"]

utility_number_1_1 = ss["vphi_1"] * N_raw_1_1 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_1"] / ss["gamma_hh_1_1"] / ss["m1"]
utility_number_1_2 = ss["vphi_1"] * N_raw_1_2 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_2"] / ss["gamma_hh_1_2"] / ss["m1"]
utility_number_1_3 = ss["vphi_1"] * N_raw_1_3 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_3"] / ss["gamma_hh_1_3"] / ss["m1"]
utility_number_2_1 = ss["vphi_2"] * N_raw_2_1 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_1"] / ss["gamma_hh_2_1"] / ss["m2"]
utility_number_2_2 = ss["vphi_2"] * N_raw_2_2 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_2"] / ss["gamma_hh_2_2"] / ss["m2"]
utility_number_2_3 = ss["vphi_2"] * N_raw_2_3 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_3"] / ss["gamma_hh_2_3"] / ss["m2"]
utility_number_3_1 = ss["vphi_3"] * N_raw_3_1 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_1"] / ss["gamma_hh_3_1"] / ss["m3"]
utility_number_3_2 = ss["vphi_3"] * N_raw_3_2 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_2"] / ss["gamma_hh_3_2"] / ss["m3"]
utility_number_3_3 = ss["vphi_3"] * N_raw_3_3 ** (1 / ss["frisch"]) / (1 - ss["tax"]) / ss["w_occ_3"] / ss["gamma_hh_3_3"] / ss["m3"]

# responses
du_c_1_1 = G['u_c_1_1']['covid_shock'] @ dcovid + utility_number_1_1
du_c_1_2 = G['u_c_1_2']['covid_shock'] @ dcovid + utility_number_1_2
du_c_1_3 = G['u_c_1_3']['covid_shock'] @ dcovid + utility_number_1_3
du_c_2_1 = G['u_c_2_1']['covid_shock'] @ dcovid + utility_number_2_1
du_c_2_2 = G['u_c_2_2']['covid_shock'] @ dcovid + utility_number_2_2
du_c_2_3 = G['u_c_2_3']['covid_shock'] @ dcovid + utility_number_2_3
du_c_3_1 = G['u_c_3_1']['covid_shock'] @ dcovid + utility_number_3_1
du_c_3_2 = G['u_c_3_2']['covid_shock'] @ dcovid + utility_number_3_2
du_c_3_3 = G['u_c_3_3']['covid_shock'] @ dcovid + utility_number_3_3

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Disutility responses to covid shock')
axs[0, 0].plot(du_c_1_1[:25], label = "first occupation")
axs[0, 0].plot(du_c_1_2[:25], label = "second occupation")
axs[0, 0].plot(du_c_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(du_c_2_1[:25], label = "first occupation")
axs[0, 1].plot(du_c_2_2[:25], label = "second occupation")
axs[0, 1].plot(du_c_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(du_c_3_1[:25], label = "first occupation")
axs[1, 0].plot(du_c_3_2[:25], label = "second occupation")
axs[1, 0].plot(du_c_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

dC1 = G['C1']['covid_shock'] @ dcovid
dC2 = G['C2']['covid_shock'] @ dcovid
dC3 = G['C3']['covid_shock'] @ dcovid
dC = G['C']['covid_shock'] @ dcovid

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dC1[:25])
axs[0, 0].set_title('consumption 1 hh')
axs[0, 1].plot(dC2[:25])
axs[0, 1].set_title('consumption 2 hh')
axs[1, 0].plot(dC3[:25])
axs[1, 0].set_title('consumption 3 hh')
axs[1, 1].plot(dC[:25])
axs[1, 1].set_title('consumption aggregate')
plt.show()


dsus = G['susceptible']['covid_shock'] @ dcovid
dinf = G['infected']['covid_shock'] @ dcovid
drec = G['recovered']['covid_shock'] @ dcovid

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dsus[:125])
axs[0, 0].set_title('susceptible')
axs[0, 1].plot(dinf[:125])
axs[0, 1].set_title('infected')
axs[1, 0].plot(drec[:125])
axs[1, 0].set_title('recovered')
plt.show()

dN_1_1 = G['N_occ_sec_1_1']['covid_shock'] @ dcovid
dN_1_2 = G['N_occ_sec_1_2']['covid_shock'] @ dcovid
dN_1_3 = G['N_occ_sec_1_3']['covid_shock'] @ dcovid
dN_2_1 = G['N_occ_sec_2_1']['covid_shock'] @ dcovid
dN_2_2 = G['N_occ_sec_2_2']['covid_shock'] @ dcovid
dN_2_3 = G['N_occ_sec_2_3']['covid_shock'] @ dcovid
dN_3_1 = G['N_occ_sec_3_1']['covid_shock'] @ dcovid
dN_3_2 = G['N_occ_sec_3_2']['covid_shock'] @ dcovid
dN_3_3 = G['N_occ_sec_3_3']['covid_shock'] @ dcovid

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dN_1_1[:25])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:25])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:25])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:25])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:25])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:25])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:25])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:25])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:25])
axs[2, 2].set_title('N sec 3 occ 3')
plt.show()

# graphs: distributions
plt.plot(ss['a1_grid'][:10], ss['c1'][0, 25, :10].T)
plt.xlabel('Assets'), plt.ylabel('Consumption')
plt.show()

plt.plot(ss['a2_grid'][:10], ss['c2'][2, 25, :10].T)
plt.xlabel('Assets'), plt.ylabel('Consumption')
plt.show()

plt.plot(ss['a3_grid'][:10], ss['c3'][2, 25, :10].T)
plt.xlabel('Assets'), plt.ylabel('Consumption')
plt.show()

