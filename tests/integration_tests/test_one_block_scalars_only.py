import numpy as np
import pytest
from assertpy import assert_that

import determinacy as det
import jacobian as jac
from simple_block import simple


@pytest.fixture
def ss():
    return {'phi': 1.5, 'i': 0, 'pi': 0}


@pytest.fixture
def expected_A():
    expected_A = np.zeros((1199, 1, 1))
    expected_A[599][0][0] = -1.0
    return expected_A


@pytest.fixture
def expected_pi():
    expected_pi = np.zeros((300, 300))
    for i in range(300):
        expected_pi[i][i] = 1.5
    return expected_pi


def test_one_block_scalars_only(ss, expected_A, expected_pi):
    # source block ------------------------------------------------------------

    @simple
    def taylor(pi, phi, i):
        taylor_eq = phi * pi - i
        return taylor_eq

    # Collect data ------------------------------------------------------------

    block_list = [taylor]
    unknowns = ['i']
    targets = ['taylor_eq']
    T = 300

    # Verify A ----------------------------------------------------------------

    A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=True)

    assert_that(A).is_not_none().is_instance_of(np.ndarray)
    assert_that(A.shape).is_equal_to((1199, 1, 1))
    assert_that(np.array_equal(A, expected_A)).is_true()

    # Verify Winding number ---------------------------------------------------

    wn = det.winding_criterion(A)
    assert_that(wn).is_zero()

    # Verify G ----------------------------------------------------------------

    G = jac.get_G(block_list=block_list, exogenous=['pi'], unknowns=unknowns, targets=targets, T=300, ss=ss)
    assert_that(G).described_as("G").is_not_none().is_instance_of(dict).is_not_empty().contains_only("i")

    i = G["i"]
    assert_that(i).described_as("i").is_not_none().is_instance_of(dict).is_not_empty().contains_only("pi")

    pi = i["pi"]
    assert_that(pi).described_as("pi").is_not_none().is_instance_of(np.ndarray)
    assert_that(pi.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(pi, expected_pi)).is_true()
