import numpy as np
import pytest
from assertpy import assert_that

import determinacy as det
import jacobian as jac
from simple_block import simple


@pytest.fixture
def ss():
    return {'phi': 1.5, 'i': 0, 'pi': 0, 'k': 0.5, 'y': 0, 'zeta': 0.1, 'eps': 0}


@pytest.fixture
def expected_A():
    expected_A = np.zeros((1199, 2, 2))
    expected_A[599][0][0] = -1.0
    expected_A[599] = [[-1, 1.5], [0, 1]]
    return expected_A


@pytest.fixture
def expected_i_y():
    expected_i_y = np.zeros((300, 300))
    for i in range(300):
        expected_i_y[i][i] = 0.75
    return expected_i_y


@pytest.fixture
def expected_i_eps():
    expected_i_eps = np.zeros((300, 300))
    for i in range(300):
        expected_i_eps[i][i] = 1.0
    return expected_i_eps


@pytest.fixture
def expected_pi_y():
    expected_pi_y = np.negative(np.zeros((300, 300)))
    for i in range(300):
        expected_pi_y[i][i] = 0.5
    return expected_pi_y


@pytest.fixture
def expected_pi_eps():
    expected_pi_eps = np.negative(np.zeros((300, 300)))
    return expected_pi_eps


def test_two_blocks_scalars_only(ss, expected_A, expected_i_y, expected_i_eps, expected_pi_y, expected_pi_eps):
    # source blocks -----------------------------------------------------------

    @simple
    def taylor(pi, phi, i, eps):
        taylor_eq = phi * pi - i + eps
        return taylor_eq

    @simple
    def phillips(y, k, pi):
        nkpc = -k * y + pi
        return nkpc

    # Collect data ------------------------------------------------------------

    block_list = [taylor, phillips]
    unknowns = ['i', 'pi']
    targets = ['taylor_eq', 'nkpc']
    T = 300

    # Verify A ----------------------------------------------------------------

    A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)

    assert_that(A).is_not_none().is_instance_of(np.ndarray)
    assert_that(A.shape).is_equal_to((1199, 2, 2))
    assert_that(np.array_equal(A, expected_A)).is_true()

    # Verify Winding number ---------------------------------------------------

    wn = det.winding_criterion(A)
    assert_that(wn).is_zero()

    # Verify G ----------------------------------------------------------------

    G = jac.get_G(block_list=block_list, exogenous=['y', 'eps'], unknowns=unknowns, targets=targets, T=300, ss=ss)
    assert_that(G).described_as("G").is_not_none().is_instance_of(dict).is_not_empty().contains_only("i", "pi")

    i = G["i"]
    assert_that(i).described_as("i").is_not_none().is_instance_of(dict).is_not_empty().contains_only("y", "eps")

    i_y = G["i"]["y"]
    assert_that(i_y).described_as("i_y").is_not_none().is_instance_of(np.ndarray)
    assert_that(i_y.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(i_y, expected_i_y)).is_true()

    i_eps = G["i"]["eps"]
    assert_that(i_eps).described_as("i_eps").is_not_none().is_instance_of(np.ndarray)
    assert_that(i_eps.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(i_eps, expected_i_eps)).is_true()

    pi = G["pi"]
    assert_that(pi).described_as("pi").is_not_none().is_instance_of(dict).is_not_empty().contains_only("y", "eps")

    pi_y = G["pi"]["y"]
    assert_that(pi_y).described_as("pi_y").is_not_none().is_instance_of(np.ndarray)
    assert_that(pi_y.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(pi_y, expected_pi_y)).is_true()

    pi_eps = G["pi"]["eps"]
    assert_that(pi_eps).described_as("pi_eps").is_not_none().is_instance_of(np.ndarray)
    assert_that(pi_eps.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(pi_eps, expected_pi_eps)).is_true()
