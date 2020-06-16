import numpy as np
import matplotlib.pyplot as plt

import jacobian as jac
import household_blocks
import determinacy as det
from steady_state import hank_ss
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

#calculate A and G
A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)
wn = det.winding_criterion(A)
print(f'Winding number: {wn}')

G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=True)

rhos = np.array([0.8])
drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])


td_nonlin = nonlinear.td_solve(ss, block_list, unknowns, targets,
                               rstar=ss['rstar']+drstar[:,0], use_saved=False)

