import numpy as np
from numba import njit

import asymptotic
from ._simple_sparse_abstract import SimpleSparseAbstract


class SimpleSparseVector(SimpleSparseAbstract):
    """
    SimpleSparse class implementation for the Vector objects.
    See parent class documentation for the more details
    """


    @property
    def asymptotic_time_invariant(self):
        raise NotImplementedError

    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""
        raise NotImplementedError

    # magic methods -----------------------------------------------------------

    def __add__(self, A):
        raise NotImplementedError

    def __eq__(self, s):
        if self.elements.keys() != s.elements.keys():
            return False
        return all([all(self.elements[k] == s.elements[k]) for k in self.elements.keys()])



    def __matmul__(self, A):
        raise NotImplementedError

    def __mul__(self, a):
        raise NotImplementedError

    def __radd__(self, A):
        raise NotImplementedError

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x}' for (i, m), x in self.elements.items()) + '}'
        return f'SimpleSparse({formatted})'

    def __rmatmul__(self, A):
        # multiplication rule when this object is on right (will only be called when left is matrix)
        # for simplicity, just use transpose to reduce this to previous cases
        raise NotImplementedError


def multiply_basis(t1, t2):
    """Matrix multiplication operation mapping two sparse basis elements to another."""
    # equivalent to formula in Proposition 2 of Sequence Space Jacobian paper, but with
    # signs of i and j flipped to reflect different sign convention used here
    raise NotImplementedError


def multiply_rs_rs(s1, s2):
    """Matrix multiplication operation on two SimpleSparse objects."""
    # iterate over all pairs (i, m) -> x and (j, n) -> y in objects,
    # add all pairwise products to get overall product
    raise NotImplementedError


@njit
def multiply_rs_matrix(indices, xs, A):
    """Matrix multiplication of SimpleSparse object ('indices' and 'xs') and matrix A.
    Much more computationally demanding than multiplying two SimpleSparse (which is almost
    free with simple analytical formula), so we implement as jitted function."""
    raise NotImplementedError
