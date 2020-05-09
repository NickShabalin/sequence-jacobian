import numpy as np


def post_decision_vfun(Va_p, Vb_p, Pi, beta):
    Wb = (Vb_p.T @ (beta * Pi.T)).T
    Wa = (Va_p.T @ (beta * Pi.T)).T
    return Wb, Wa


def Psi_fun(ap, a, ra, chi0, chi1, chi2):
    return chi1 / chi2 * np.abs((ap - (1 + ra) * a)) ** chi2 / ((1 + ra) * a + chi0)


def Psi1_fun(ap, a, ra, chi0, chi1, chi2):
    return np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)


def Psi2_fun(ap, a, ra, chi0, chi1, chi2):
    Psi1 = np.sign(ap - (1 + ra) * a) * chi1 * np.abs((ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** (chi2 - 1)
    return -(1 + ra) * (Psi1 + chi1 * (chi2 - 1) / chi2 * (np.abs(ap - (1 + ra) * a) / ((1 + ra) * a + chi0)) ** chi2)