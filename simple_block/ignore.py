from functools import singledispatch

import numpy as np


class Ignore(float):
    """This class ignores time displacements of a scalar."""

    def __call__(self, index):
        return self


class IgnoreFactory:
    """Factory for Ignore class production"""""

    def __init__(self):
        self.ignore = singledispatch(self.ignore)
        self.ignore.register(np.ndarray, self._ignore_ndarray)

    # common case
    def ignore(self, arg) -> Ignore:
        return Ignore(arg)

    # nd array case
    def _ignore_ndarray(self, arg) -> np.ndarray:
        result = arg.copy().astype(Ignore)
        for i in range(len(arg)):
            result[i] = Ignore(i)
        return result
