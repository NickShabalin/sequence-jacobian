from functools import singledispatchmethod

import numpy as np


class ReporterScalar(float):
    """This class adds to a shared set to tell us what x[i] are accessed.
    Needed for differentiation in SimpleBlock.jac()"""

    def __init__(self, value):
        self.myset = set()

    def __call__(self, index):
        self.myset.add(index)
        return self


class ReporterVector(np.ndarray):
    """The same as above, but for vector"""

    def __new__(cls, value):
        obj = super().__new__(cls,
                              shape=(len(value),),
                              buffer=value,
                              dtype=value.dtype)
        obj.myset = set()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.myset = getattr(obj, 'myset', set())

    def __call__(self, index):
        self.myset.add(index)
        return self


class ReporterFactory:
    """Factory for Reporter classes creation"""""

    @singledispatchmethod
    def reporter(self, value):
        raise NotImplementedError

    @reporter.register
    def _(self, value: int) -> ReporterScalar:
        return ReporterScalar(value)

    @reporter.register
    def _(self, value: float) -> ReporterScalar:
        return ReporterScalar(value)

    @reporter.register
    def _(self, value: np.ndarray) -> ReporterVector:
        return ReporterVector(value)
