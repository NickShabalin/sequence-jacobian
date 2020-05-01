from functools import singledispatchmethod

import numpy as np


class PerturbScalar(float):
    """This class uses the shared set to perturb each x[i] separately, starting at steady-state values.
    Needed for differentiation in SimpleBlock.jac()"""

    def __new__(cls, value, h, index):
        if index == 0:
            return float.__new__(cls, value + h)
        else:
            return float.__new__(cls, value)

    def __init__(self, value, h, index):
        self.h = h
        self.index = index

    def __call__(self, index):
        if self.index == 0:
            if index == 0:
                return self
            else:
                return self - self.h
        else:
            if self.index == index:
                return self + self.h
            else:
                return self


class PerturbVector(np.ndarray):
    """The same as above, but for vector"""

    def __new__(cls, value, h, index):
        value = value + h if index == 0 else value
        obj = super().__new__(cls,
                              shape=(len(value),),
                              buffer=value,
                              dtype=value.dtype)
        obj.h = h
        obj.index = index
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.h = getattr(obj, 'h', None)
        self.index = getattr(obj, 'index', None)

    def __call__(self, index):
        if self.index == 0:
            if index == 0:
                return self
            else:
                return self - self.h
        else:
            if self.index == index:
                return self + self.h
            else:
                return self


class PerturbFactory:
    """Factory for Perturb classes creation"""""

    @singledispatchmethod
    def perturb(self, value, h, index):
        raise NotImplementedError

    @perturb.register
    def _(self, value: int, h, index) -> PerturbScalar:
        return PerturbScalar(value, h, index)

    @perturb.register
    def _(self, value: float, h, index) -> PerturbScalar:
        return PerturbScalar(value, h, index)

    @perturb.register
    def _(self, value: np.ndarray, h, index) -> PerturbVector:
        return PerturbVector(value, h, index)
