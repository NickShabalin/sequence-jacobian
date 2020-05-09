import numpy as np
import pytest
from assertpy import assert_that

import determinacy as det
import jacobian as jac
from simple_with_vector_args import SimpleWithVectorArgs


@pytest.fixture
def ss():
    return {'phi_1': 1.5, "phi_2": 1.2, 'i_1': 0, "i_2": 0, 'pi': 0}


@pytest.fixture
def expected_A():
    expected_A = np.zeros((1199, 2, 2))
    expected_A[599] = [[-1, 0], [0, -1]]
    return expected_A


@pytest.fixture
def expected_i_1_pi():
    expected_i_1_pi = np.zeros((300, 300))
    for i in range(300):
        expected_i_1_pi[i][i] = 1.5
    return expected_i_1_pi
    pass

@pytest.fixture
def expected_i_2_pi():
    expected_i_2_pi = np.zeros((300, 300))
    for i in range(300):
        expected_i_2_pi[i][i] = 1.2
    return expected_i_2_pi
    pass


def test_one_block_with_vectors(ss, expected_A, expected_i_1_pi, expected_i_2_pi):
    # source block ------------------------------------------------------------

    @SimpleWithVectorArgs({"phi": 2, "i": 2})
    def taylor(pi, phi, i):
        taylor_eq = phi * pi - i
        return taylor_eq

    # Collect data ------------------------------------------------------------

    block_list = [taylor]
    unknowns = ["i_1", "i_2"]
    targets = ["taylor_eq_1", "taylor_eq_2"]
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

    G = jac.get_G(block_list=block_list,
                  exogenous=['pi'],
                  unknowns=unknowns,
                  targets=targets,
                  T=300, ss=ss)

    assert_that(G).described_as("G").is_not_none().is_instance_of(dict).is_not_empty().contains_only("i_1", "i_2")

    i_1 = G["i_1"]
    assert_that(i_1).described_as("i_1").is_not_none().is_instance_of(dict).is_not_empty().contains_only("pi")

    i_1_pi = i_1["pi"]
    assert_that(i_1_pi).described_as("i_1_pi").is_not_none().is_instance_of(np.ndarray)
    assert_that(i_1_pi.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(i_1_pi, expected_i_1_pi)).is_true()

    i_2 = G["i_2"]
    assert_that(i_2).described_as("i_2").is_not_none().is_instance_of(dict).is_not_empty().contains_only("pi")

    i_2_pi = i_2["pi"]
    assert_that(i_2_pi).described_as("i_2_pi").is_not_none().is_instance_of(np.ndarray)
    assert_that(i_2_pi.shape).is_equal_to((300, 300))
    assert_that(np.array_equal(i_2_pi, expected_i_2_pi)).is_true()
