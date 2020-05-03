from functools import singledispatchmethod

import numpy as np


class IgnoreScalar(float):
    """This class ignores time displacements of a scalar."""

    def __call__(self, index):
        return self


class IgnoreVector(np.ndarray):
    """This class ignores time displacements of a vector."""

    def __new__(cls, value):
        return super().__new__(cls,
                               shape=value.shape,
                               buffer=value,
                               dtype=value.dtype)

    def __call__(self, index):
        return self


class IgnoreFactory:
    """Factory for Ignore classes creation"""

    @singledispatchmethod
    def ignore(self, arg):
        raise NotImplementedError

    @ignore.register
    def _(self, arg: int) -> IgnoreScalar:
        return IgnoreScalar(arg)

    @ignore.register
    def _(self, arg: float) -> IgnoreScalar:
        return IgnoreScalar(arg)

    @ignore.register
    def _(self, arg: np.ndarray) -> IgnoreVector:
        return IgnoreVector(arg)
