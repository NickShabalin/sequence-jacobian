import numpy as np
import pytest

from simple_block._ignore import IgnoreFactory, IgnoreScalar, IgnoreVector


# IGNORE CLASSES ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, index", [(0, 1), (-10, 0), (5, -17), (3.5, 2325), (0.005554, 232)])
def test_ignore_scalar(value, index):
    ignore_scalar = IgnoreScalar(value)
    assert ignore_scalar(index) == value


@pytest.mark.parametrize("value, index", [([1, 2], 1),
                                          ([3, 6, 1.55], 0),
                                          ([0.00443, 0, 0], -1),
                                          ([0.003, 0], 23),
                                          ([[1, 2], [0, 0], [3.5, 0.00023]], -2)])
def test_ignore_vector(value, index):
    nd_arr = np.array(value)
    ignore_vector = IgnoreVector(nd_arr)
    assert np.array_equal(ignore_vector(index), nd_arr)


# IGNORE FACTORY ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1, 0, -3, 0.000034, 2344342.3])
def test_ignore_factory_scalar(value):
    ignore_obj = IgnoreFactory().ignore(value)
    assert isinstance(ignore_obj, IgnoreScalar)


@pytest.mark.parametrize("value", [[0, 0], [-1, 0.00032], [0, -32.444, 2343242]])
def test_ignore_factory_vector(value):
    ignore_obj = IgnoreFactory().ignore(np.array(value))
    assert isinstance(ignore_obj, IgnoreVector)


def test_ignore_factory_unsupported_type():
    with pytest.raises(NotImplementedError):
        IgnoreFactory().ignore("%^&*DS")
