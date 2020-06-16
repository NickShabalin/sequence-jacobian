import numpy as np

def SIR_with_loop_calc():
    # initialization of the variables
    susceptible_1 = 1
    infected_1 = 0
    recovered_1 = 0
    covid_shock = 0.000001
    # covid_shock = 0.01
    beta_sir = 1.5
    gamma_sir = 0.8
    n = 300
    S = R = I = np.array([])

    S = np.append(S, 1)
    I = np.append(I, 0)
    R = np.append(R, 0)

    # calucalation of infected, susceptible and recovered
    for ThisT in range(n-1):
        # susceptible = (1 - beta_sir * infected_1 * (N / (infected_1 + recovered_1 + susceptible_1))) * susceptible_1 - covid_shock ;
        susceptible = (1 - beta_sir * infected_1) * susceptible_1 - covid_shock
        # infected = (1 - gamma_sir + beta_sir * susceptible_1 * (N / (infected_1 + recovered_1 + susceptible_1))) * infected_1 + covid_shock ;
        infected = (1 - gamma_sir + beta_sir * susceptible_1) * infected_1 + covid_shock

        recovered = recovered_1 + gamma_sir * infected_1

        ## move ahead
        susceptible_1 = susceptible
        infected_1 = infected
        recovered_1 = recovered
        covid_shock = 0

        ## track evolution of stocks
        S = np.append(S, susceptible)
        I = np.append(I, infected)
        R = np.append(R, recovered)

    return S, I, R


