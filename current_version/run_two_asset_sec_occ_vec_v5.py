import numpy as np
import matplotlib.pyplot as plt

import jacobian as jac
from household_blocks_with_gamma_power_e import household_inc1, household_inc2, household_inc3
import determinacy as det
from steady_state_with_gamma_power_e import hank_ss
import nonlinear
from sir_calculation import SIR_with_loop_calc



from simple_and_solved_blocks_with_gamma_power_e import taylor, fiscal, finance, arbitrage, \
                                        dividend, dividend_agg, production, consumers_aggregator, \
                                        asset_mkt_clearing, sir_block


# calculate steady state
ss = hank_ss()

#calculate suseptible, infected, recovered
susceptible, infected, recovered = SIR_with_loop_calc()

# assign variables for A and G calculation
T = 300

block_list = [consumers_aggregator, household_inc1, household_inc2,
              household_inc3, dividend, arbitrage, dividend_agg,
              taylor, fiscal, finance, production, asset_mkt_clearing]

exogenous = ['rstar', 'G', 'productivity_sec_1', 'productivity_sec_2',
             'productivity_sec_3', 'susceptible', 'infected', 'recovered']

unknowns = ['r', 'p', 'equity_price_sec_1',
            'equity_price_sec_2', 'equity_price_sec_3']

targets = ['fisher',
           'equity_1', 'equity_2', 'equity_3', 'asset_mkt']

#calculate A and G
A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)
wn = det.winding_criterion(A)
print(f'Winding number: {wn}')

G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=False)

rhos = np.array([0.8])
#drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
drstar = 0.0 * rhos ** (np.arange(T)[:, np.newaxis])

# rhos = np.array([0.0, 0.1])
# dcovid = 0.00001 * rhos ** (np.arange(T)[:, np.newaxis])
#
td_nonlin = nonlinear.td_solve(ss, block_list, unknowns, targets,
                               rstar=ss['rstar']+drstar[:,0], use_saved=False)





# td_nonlin = nonlinear.td_solve(ss, block_list, unknowns, targets,
#                                covid_shock=ss['covid_shock']+dcovid[:,0], use_saved=True)


##################################################################################################################
# plot results
# create plot style
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('summer')
plt.rcParams["figure.titlesize"] = 'xx-large'

# graphs: mp shock
rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dY = td_nonlin['Y']['rstar'] @ drstar
dN = td_nonlin['N']['rstar'] @ drstar
di = td_nonlin['i']['rstar'] @ drstar
dC = td_nonlin['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 25 bp monetary policy shock (easing)')
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
dC1 = td_nonlin['C1']['rstar'] @ drstar
dC2 = td_nonlin['C2']['rstar'] @ drstar
dC3 = td_nonlin['C3']['rstar'] @ drstar
dC = td_nonlin['C']['rstar'] @ drstar

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to 25 bp monetary policy shock (easing)')
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
dY = td_nonlin['Y']['productivity_sec_1'] @ dproductivity_sec_1
dN = td_nonlin['N']['productivity_sec_1'] @ dproductivity_sec_1
di = td_nonlin['i']['productivity_sec_1'] @ dproductivity_sec_1
dC = td_nonlin['C']['productivity_sec_1'] @ dproductivity_sec_1

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


dN_1_1 = td_nonlin['N_occ_sec_1_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_1_2 = td_nonlin['N_occ_sec_1_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_1_3 = td_nonlin['N_occ_sec_1_3']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_1 = td_nonlin['N_occ_sec_2_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_2 = td_nonlin['N_occ_sec_2_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_2_3 = td_nonlin['N_occ_sec_2_3']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_1 = td_nonlin['N_occ_sec_3_1']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_2 = td_nonlin['N_occ_sec_3_2']['productivity_sec_1'] @ dproductivity_sec_1
dN_3_3 = td_nonlin['N_occ_sec_3_3']['productivity_sec_1'] @ dproductivity_sec_1

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


dz_grid_1_1_1 = td_nonlin['z_grid_1_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = td_nonlin['z_grid_1_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = td_nonlin['z_grid_1_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = td_nonlin['z_grid_2_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = td_nonlin['z_grid_2_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = td_nonlin['z_grid_2_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = td_nonlin['z_grid_3_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = td_nonlin['z_grid_3_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = td_nonlin['z_grid_3_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = td_nonlin['z_grid_1_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = td_nonlin['z_grid_1_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = td_nonlin['z_grid_1_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = td_nonlin['z_grid_2_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = td_nonlin['z_grid_2_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = td_nonlin['z_grid_2_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = td_nonlin['z_grid_3_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = td_nonlin['z_grid_3_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = td_nonlin['z_grid_3_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = td_nonlin['z_grid_1_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = td_nonlin['z_grid_1_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = td_nonlin['z_grid_1_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = td_nonlin['z_grid_2_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = td_nonlin['z_grid_2_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = td_nonlin['z_grid_2_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = td_nonlin['z_grid_3_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = td_nonlin['z_grid_3_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = td_nonlin['z_grid_3_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

# graphs: productivity shock in sec 2
rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec_2 = -0.04 * rhos ** (np.arange(T)[:, np.newaxis])
dY = td_nonlin['Y']['productivity_sec_2'] @ dproductivity_sec_2
dN = td_nonlin['N']['productivity_sec_2'] @ dproductivity_sec_2
di = td_nonlin['i']['productivity_sec_2'] @ dproductivity_sec_2
dC = td_nonlin['C']['productivity_sec_2'] @ dproductivity_sec_2

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


dN_1_1 = td_nonlin['N_occ_sec_1_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_1_2 = td_nonlin['N_occ_sec_1_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_1_3 = td_nonlin['N_occ_sec_1_3']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_1 = td_nonlin['N_occ_sec_2_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_2 = td_nonlin['N_occ_sec_2_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_2_3 = td_nonlin['N_occ_sec_2_3']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_1 = td_nonlin['N_occ_sec_3_1']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_2 = td_nonlin['N_occ_sec_3_2']['productivity_sec_2'] @ dproductivity_sec_2
dN_3_3 = td_nonlin['N_occ_sec_3_3']['productivity_sec_2'] @ dproductivity_sec_2

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

dz_grid_1_1_1 = td_nonlin['z_grid_1_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = td_nonlin['z_grid_1_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = td_nonlin['z_grid_1_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = td_nonlin['z_grid_2_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = td_nonlin['z_grid_2_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = td_nonlin['z_grid_2_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = td_nonlin['z_grid_3_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = td_nonlin['z_grid_3_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = td_nonlin['z_grid_3_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = td_nonlin['z_grid_1_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = td_nonlin['z_grid_1_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = td_nonlin['z_grid_1_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = td_nonlin['z_grid_2_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = td_nonlin['z_grid_2_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = td_nonlin['z_grid_2_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = td_nonlin['z_grid_3_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = td_nonlin['z_grid_3_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = td_nonlin['z_grid_3_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = td_nonlin['z_grid_1_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = td_nonlin['z_grid_1_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = td_nonlin['z_grid_1_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = td_nonlin['z_grid_2_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = td_nonlin['z_grid_2_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = td_nonlin['z_grid_2_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = td_nonlin['z_grid_3_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = td_nonlin['z_grid_3_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = td_nonlin['z_grid_3_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

# graphs: productivity shock in sec 3
rhos = np.array([0.2, 0.6, 0.8])
dproductivity_sec_3 = -0.06 * rhos ** (np.arange(T)[:, np.newaxis])
dY = td_nonlin['Y']['productivity_sec_3'] @ dproductivity_sec_3
dN = td_nonlin['N']['productivity_sec_3'] @ dproductivity_sec_3
di = td_nonlin['i']['productivity_sec_3'] @ dproductivity_sec_3
dC = td_nonlin['C']['productivity_sec_3'] @ dproductivity_sec_3

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


dN_1_1 = td_nonlin['N_occ_sec_1_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_1_2 = td_nonlin['N_occ_sec_1_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_1_3 = td_nonlin['N_occ_sec_1_3']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_1 = td_nonlin['N_occ_sec_2_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_2 = td_nonlin['N_occ_sec_2_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_2_3 = td_nonlin['N_occ_sec_2_3']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_1 = td_nonlin['N_occ_sec_3_1']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_2 = td_nonlin['N_occ_sec_3_2']['productivity_sec_3'] @ dproductivity_sec_3
dN_3_3 = td_nonlin['N_occ_sec_3_3']['productivity_sec_3'] @ dproductivity_sec_3


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


dz_grid_1_1_1 = td_nonlin['z_grid_1_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = td_nonlin['z_grid_1_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = td_nonlin['z_grid_1_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = td_nonlin['z_grid_2_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = td_nonlin['z_grid_2_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = td_nonlin['z_grid_2_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = td_nonlin['z_grid_3_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = td_nonlin['z_grid_3_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = td_nonlin['z_grid_3_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = td_nonlin['z_grid_1_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = td_nonlin['z_grid_1_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = td_nonlin['z_grid_1_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = td_nonlin['z_grid_2_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = td_nonlin['z_grid_2_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = td_nonlin['z_grid_2_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = td_nonlin['z_grid_3_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = td_nonlin['z_grid_3_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = td_nonlin['z_grid_3_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = td_nonlin['z_grid_1_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = td_nonlin['z_grid_1_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = td_nonlin['z_grid_1_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = td_nonlin['z_grid_2_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = td_nonlin['z_grid_2_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = td_nonlin['z_grid_2_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = td_nonlin['z_grid_3_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = td_nonlin['z_grid_3_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = td_nonlin['z_grid_3_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

# graphs: government spending shock
rhos = np.array([0.8])
dG = 0.001 * rhos ** (np.arange(T)[:, np.newaxis])
dY = td_nonlin['Y']['G'] @ dG
dN = td_nonlin['N']['G'] @ dG
di = td_nonlin['i']['G'] @ dG
dC = td_nonlin['C']['G'] @ dG

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
dC1 = td_nonlin['C1']['G'] @ dG
dC2 = td_nonlin['C2']['G'] @ dG
dC3 = td_nonlin['C3']['G'] @ dG
dC = td_nonlin['C']['G'] @ dG

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

dY = td_nonlin['Y']['infected'] @ infected
dN = td_nonlin['N']['infected'] @ infected
di = td_nonlin['I']['infected'] @ infected
dC = td_nonlin['C']['infected'] @ infected

# graphs: covid shock
'''
rhos = np.array([0.0])
dcovid = 0.000001 * rhos ** (np.arange(T)[:, np.newaxis])
#dcovid = 0.1 * rhos ** (np.arange(T)[:, np.newaxis])
dY = td_nonlin['Y']['covid_shock'] @ dcovid
dN = td_nonlin['N']['covid_shock'] @ dcovid
di = td_nonlin['I']['covid_shock'] @ dcovid
dC = td_nonlin['C']['covid_shock'] @ dcovid
'''
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dY[:100])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:100])
axs[0, 1].set_title('Investments')
axs[1, 0].plot(dN[:100])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:100])
axs[1, 1].set_title('Consumption')
plt.show()



# responses
dz_grid_1_1_1 = td_nonlin['z_grid_1_1_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = td_nonlin['z_grid_1_1_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = td_nonlin['z_grid_1_1_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = td_nonlin['z_grid_2_1_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = td_nonlin['z_grid_2_1_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = td_nonlin['z_grid_2_1_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = td_nonlin['z_grid_3_1_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = td_nonlin['z_grid_3_1_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = td_nonlin['z_grid_3_1_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = td_nonlin['z_grid_1_2_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = td_nonlin['z_grid_1_2_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = td_nonlin['z_grid_1_2_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = td_nonlin['z_grid_2_2_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = td_nonlin['z_grid_2_2_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = td_nonlin['z_grid_2_2_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = td_nonlin['z_grid_3_2_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = td_nonlin['z_grid_3_2_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = td_nonlin['z_grid_3_2_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']

dz_grid_1_3_1 = td_nonlin['z_grid_1_3_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = td_nonlin['z_grid_1_3_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = td_nonlin['z_grid_1_3_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = td_nonlin['z_grid_2_3_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = td_nonlin['z_grid_2_3_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = td_nonlin['z_grid_2_3_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = td_nonlin['z_grid_3_3_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = td_nonlin['z_grid_3_3_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = td_nonlin['z_grid_3_3_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']


#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


dC1 = td_nonlin['C1']['infected'] @ infected
dC2 = td_nonlin['C2']['infected'] @ infected
dC3 = td_nonlin['C3']['infected'] @ infected
dC = td_nonlin['C']['infected'] @ infected

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dC1[:100])
axs[0, 0].set_title('consumption 1 hh')
axs[0, 1].plot(dC2[:100])
axs[0, 1].set_title('consumption 2 hh')
axs[1, 0].plot(dC3[:100])
axs[1, 0].set_title('consumption 3 hh')
axs[1, 1].plot(dC[:100])
axs[1, 1].set_title('consumption aggregate')
plt.show()

dw1 = td_nonlin['w_occ_1']['infected'] @ infected
dw2 = td_nonlin['w_occ_2']['infected'] @ infected
dw3 = td_nonlin['w_occ_3']['infected'] @ infected
dw = td_nonlin['w']['infected'] @ infected

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dw1[:100])
axs[0, 0].set_title('wage in the first occupation')
axs[0, 1].plot(dw2[:100])
axs[0, 1].set_title('wage in the second occupation')
axs[1, 0].plot(dw3[:100])
axs[1, 0].set_title('wage in the third occupation')
axs[1, 1].plot(dw[:100])
axs[1, 1].set_title('wage aagregate')
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(susceptible[:100])
axs[0, 0].set_title('susceptible')
axs[0, 1].plot(infected[:100])
axs[0, 1].set_title('infected')
axs[1, 0].plot(recovered[:100])
axs[1, 0].set_title('recovered')
plt.show()

dN_1_1 = td_nonlin['N_occ_sec_1_1']['infected'] @ infected
dN_1_2 = td_nonlin['N_occ_sec_1_2']['infected'] @ infected
dN_1_3 = td_nonlin['N_occ_sec_1_3']['infected'] @ infected
dN_2_1 = td_nonlin['N_occ_sec_2_1']['infected'] @ infected
dN_2_2 = td_nonlin['N_occ_sec_2_2']['infected'] @ infected
dN_2_3 = td_nonlin['N_occ_sec_2_3']['infected'] @ infected
dN_3_1 = td_nonlin['N_occ_sec_3_1']['infected'] @ infected
dN_3_2 = td_nonlin['N_occ_sec_3_2']['infected'] @ infected
dN_3_3 = td_nonlin['N_occ_sec_3_3']['infected'] @ infected

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dN_1_1[:100])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:100])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:100])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:100])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:100])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:100])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:100])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:100])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:100])
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
fig.suptitle(r'Responses to 25 bp monetary policy shock (easing)')
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
fig.suptitle(r'Responses to 25 bp monetary policy shock (easing)')
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


dz_grid_1_1_1 = G['z_grid_1_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = G['z_grid_1_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = G['z_grid_1_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = G['z_grid_2_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = G['z_grid_2_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = G['z_grid_2_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = G['z_grid_3_1_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = G['z_grid_3_1_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = G['z_grid_3_1_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = G['z_grid_1_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = G['z_grid_1_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = G['z_grid_1_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = G['z_grid_2_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = G['z_grid_2_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = G['z_grid_2_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = G['z_grid_3_2_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = G['z_grid_3_2_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = G['z_grid_3_2_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = G['z_grid_1_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = G['z_grid_1_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = G['z_grid_1_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = G['z_grid_2_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = G['z_grid_2_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = G['z_grid_2_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = G['z_grid_3_3_1']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = G['z_grid_3_3_2']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = G['z_grid_3_3_3']['productivity_sec_1'] @ dproductivity_sec_1[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
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

dz_grid_1_1_1 = G['z_grid_1_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = G['z_grid_1_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = G['z_grid_1_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = G['z_grid_2_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = G['z_grid_2_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = G['z_grid_2_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = G['z_grid_3_1_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = G['z_grid_3_1_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = G['z_grid_3_1_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = G['z_grid_1_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = G['z_grid_1_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = G['z_grid_1_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = G['z_grid_2_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = G['z_grid_2_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = G['z_grid_2_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = G['z_grid_3_2_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = G['z_grid_3_2_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = G['z_grid_3_2_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = G['z_grid_1_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = G['z_grid_1_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = G['z_grid_1_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = G['z_grid_2_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = G['z_grid_2_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = G['z_grid_2_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = G['z_grid_3_3_1']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = G['z_grid_3_3_2']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = G['z_grid_3_3_3']['productivity_sec_2'] @ dproductivity_sec_2[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 2 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
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


dz_grid_1_1_1 = G['z_grid_1_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = G['z_grid_1_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = G['z_grid_1_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = G['z_grid_2_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = G['z_grid_2_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = G['z_grid_2_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = G['z_grid_3_1_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = G['z_grid_3_1_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = G['z_grid_3_1_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = G['z_grid_1_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = G['z_grid_1_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = G['z_grid_1_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = G['z_grid_2_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = G['z_grid_2_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = G['z_grid_2_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = G['z_grid_3_2_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = G['z_grid_3_2_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = G['z_grid_3_2_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']


dz_grid_1_3_1 = G['z_grid_1_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = G['z_grid_1_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = G['z_grid_1_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = G['z_grid_2_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = G['z_grid_2_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = G['z_grid_2_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = G['z_grid_3_3_1']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = G['z_grid_3_3_2']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = G['z_grid_3_3_3']['productivity_sec_3'] @ dproductivity_sec_3[:,2] + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation choices responses to negative productivity shock in sector 3 of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:25], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:25], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:25], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:25], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:25], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:25], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:25], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:25], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:25], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

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

dY = G['Y']['infected'] @ infected
dN = G['N']['infected'] @ infected
di = G['I']['infected'] @ infected
dC = G['C']['infected'] @ infected

# graphs: covid shock
'''
rhos = np.array([0.0])
dcovid = 0.000001 * rhos ** (np.arange(T)[:, np.newaxis])
#dcovid = 0.1 * rhos ** (np.arange(T)[:, np.newaxis])
dY = G['Y']['covid_shock'] @ dcovid
dN = G['N']['covid_shock'] @ dcovid
di = G['I']['covid_shock'] @ dcovid
dC = G['C']['covid_shock'] @ dcovid
'''
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dY[:100])
axs[0, 0].set_title('GDP')
axs[0, 1].plot(di[:100])
axs[0, 1].set_title('Investments')
axs[1, 0].plot(dN[:100])
axs[1, 0].set_title('Labor hours')
axs[1, 1].plot(dC[:100])
axs[1, 1].set_title('Consumption')
plt.show()



# responses
dz_grid_1_1_1 = G['z_grid_1_1_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_1']
dz_grid_1_1_2 = G['z_grid_1_1_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_1']
dz_grid_1_1_3 = G['z_grid_1_1_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_1']
dz_grid_2_1_1 = G['z_grid_2_1_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_1']
dz_grid_2_1_2 = G['z_grid_2_1_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_1']
dz_grid_2_1_3 = G['z_grid_2_1_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_1']
dz_grid_3_1_1 = G['z_grid_3_1_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_1']
dz_grid_3_1_2 = G['z_grid_3_1_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_1']
dz_grid_3_1_3 = G['z_grid_3_1_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_1']

dz_grid_1_2_1 = G['z_grid_1_2_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_2']
dz_grid_1_2_2 = G['z_grid_1_2_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_2']
dz_grid_1_2_3 = G['z_grid_1_2_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_2']
dz_grid_2_2_1 = G['z_grid_2_2_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_2']
dz_grid_2_2_2 = G['z_grid_2_2_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_2']
dz_grid_2_2_3 = G['z_grid_2_2_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_2']
dz_grid_3_2_1 = G['z_grid_3_2_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_2']
dz_grid_3_2_2 = G['z_grid_3_2_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_2']
dz_grid_3_2_3 = G['z_grid_3_2_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_2']

dz_grid_1_3_1 = G['z_grid_1_3_1']['infected'] @ infected + ss['N_hh_occ_1_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_1_1']) ** ss['e_grid_3']
dz_grid_1_3_2 = G['z_grid_1_3_2']['infected'] @ infected + ss['N_hh_occ_1_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_1_2']) ** ss['e_grid_3']
dz_grid_1_3_3 = G['z_grid_1_3_3']['infected'] @ infected + ss['N_hh_occ_1_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_1_3']) ** ss['e_grid_3']
dz_grid_2_3_1 = G['z_grid_2_3_1']['infected'] @ infected + ss['N_hh_occ_2_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_2_1']) ** ss['e_grid_3']
dz_grid_2_3_2 = G['z_grid_2_3_2']['infected'] @ infected + ss['N_hh_occ_2_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_2_2']) ** ss['e_grid_3']
dz_grid_2_3_3 = G['z_grid_2_3_3']['infected'] @ infected + ss['N_hh_occ_2_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_2_3']) ** ss['e_grid_3']
dz_grid_3_3_1 = G['z_grid_3_3_1']['infected'] @ infected + ss['N_hh_occ_3_1'] * ss['w_occ_1'] * (1 + ss['gamma_hh_3_1']) ** ss['e_grid_3']
dz_grid_3_3_2 = G['z_grid_3_3_2']['infected'] @ infected + ss['N_hh_occ_3_2'] * ss['w_occ_2'] * (1 + ss['gamma_hh_3_2']) ** ss['e_grid_3']
dz_grid_3_3_3 = G['z_grid_3_3_3']['infected'] @ infected + ss['N_hh_occ_3_3'] * ss['w_occ_3'] * (1 + ss['gamma_hh_3_3']) ** ss['e_grid_3']


#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the least productive')
axs[0, 0].plot(dz_grid_1_1_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_1_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_1_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_1_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_1_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_1_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_1_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_1_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_1_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the middle one')
axs[0, 0].plot(dz_grid_1_2_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_2_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_2_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_2_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_2_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_2_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_2_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_2_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_2_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Occupation responses to covid shock of the most productive')
axs[0, 0].plot(dz_grid_1_3_1[:100], label = "first occupation")
axs[0, 0].plot(dz_grid_1_3_2[:100], label = "second occupation")
axs[0, 0].plot(dz_grid_1_3_3[:100], label = "third occupation")
axs[0, 0].set_title('1 type of consumers')
axs[0, 1].plot(dz_grid_2_3_1[:100], label = "first occupation")
axs[0, 1].plot(dz_grid_2_3_2[:100], label = "second occupation")
axs[0, 1].plot(dz_grid_2_3_3[:100], label = "third occupation")
axs[0, 1].set_title('2 type of consumers')
axs[1, 0].plot(dz_grid_3_3_1[:100], label = "first occupation")
axs[1, 0].plot(dz_grid_3_3_2[:100], label = "second occupation")
axs[1, 0].plot(dz_grid_3_3_3[:100], label = "third occupation")
axs[1, 0].set_title('3 type of consumers')
leg = axs[0,0].legend()
leg = axs[0,1].legend()
leg = axs[1,0].legend()
plt.show()


dC1 = G['C1']['infected'] @ infected
dC2 = G['C2']['infected'] @ infected
dC3 = G['C3']['infected'] @ infected
dC = G['C']['infected'] @ infected

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dC1[:100])
axs[0, 0].set_title('consumption 1 hh')
axs[0, 1].plot(dC2[:100])
axs[0, 1].set_title('consumption 2 hh')
axs[1, 0].plot(dC3[:100])
axs[1, 0].set_title('consumption 3 hh')
axs[1, 1].plot(dC[:100])
axs[1, 1].set_title('consumption aggregate')
plt.show()

dw1 = G['w_occ_1']['infected'] @ infected
dw2 = G['w_occ_2']['infected'] @ infected
dw3 = G['w_occ_3']['infected'] @ infected
dw = G['w']['infected'] @ infected

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dw1[:100])
axs[0, 0].set_title('wage in the first occupation')
axs[0, 1].plot(dw2[:100])
axs[0, 1].set_title('wage in the second occupation')
axs[1, 0].plot(dw3[:100])
axs[1, 0].set_title('wage in the third occupation')
axs[1, 1].plot(dw[:100])
axs[1, 1].set_title('wage aagregate')
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(susceptible[:100])
axs[0, 0].set_title('susceptible')
axs[0, 1].plot(infected[:100])
axs[0, 1].set_title('infected')
axs[1, 0].plot(recovered[:100])
axs[1, 0].set_title('recovered')
plt.show()

dN_1_1 = G['N_occ_sec_1_1']['infected'] @ infected
dN_1_2 = G['N_occ_sec_1_2']['infected'] @ infected
dN_1_3 = G['N_occ_sec_1_3']['infected'] @ infected
dN_2_1 = G['N_occ_sec_2_1']['infected'] @ infected
dN_2_2 = G['N_occ_sec_2_2']['infected'] @ infected
dN_2_3 = G['N_occ_sec_2_3']['infected'] @ infected
dN_3_1 = G['N_occ_sec_3_1']['infected'] @ infected
dN_3_2 = G['N_occ_sec_3_2']['infected'] @ infected
dN_3_3 = G['N_occ_sec_3_3']['infected'] @ infected

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(r'Responses to covid shock')
axs[0, 0].plot(dN_1_1[:100])
axs[0, 0].set_title('N sec 1 occ 1')
axs[0, 1].plot(dN_1_2[:100])
axs[0, 1].set_title('N sec 2 occ 1')
axs[0, 2].plot(dN_1_3[:100])
axs[0, 2].set_title('N sec 3 occ 1')

axs[1, 0].plot(dN_2_1[:100])
axs[1, 0].set_title('N sec 1 occ 2')
axs[1, 1].plot(dN_2_2[:100])
axs[1, 1].set_title('N sec 2 occ 2')
axs[1, 2].plot(dN_2_3[:100])
axs[1, 2].set_title('N sec 3 occ 2')

axs[2, 0].plot(dN_3_1[:100])
axs[2, 0].set_title('N sec 1 occ 3')
axs[2, 1].plot(dN_3_2[:100])
axs[2, 1].set_title('N sec 2 occ 3')
axs[2, 2].plot(dN_3_3[:100])
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

