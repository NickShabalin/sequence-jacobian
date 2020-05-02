from copy import deepcopy

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
        indices, xs = self.array()
        tau = np.max(np.abs(indices[:, 0])) + 1
        v = np.zeros((2 * tau - 1, len(xs[0])))
        t = -indices[:, 0] + tau - 1
        v[-indices[:, 0] + tau - 1] = xs
        return asymptotic.AsymptoticTimeInvariant(v)

    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""
        raise NotImplementedError


    # magic methods -----------------------------------------------------------

    def __add__(self, A):
        if isinstance(A, SimpleSparseVector):
            # add SimpleSparse to SimpleSparse, combining dicts, summing x when (i, m) overlap
            elements = deepcopy(self.elements)
            for im, x in A.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if all(abs(elements[im]) < 1E-14):
                        del elements[im]
                else:
                    elements[im] = x
            return SimpleSparseVector(elements)
        else:
            raise NotImplementedError

    def __eq__(self, s):
        if self.elements.keys() != s.elements.keys(): return False
        return all([all(self.elements[k] == s.elements[k]) for k in self.elements.keys()])

    def __matmul__(self, A):
        if isinstance(A, SimpleSparseVector):
            # multiply SimpleSparse by SimpleSparse, simple analytical rules in multiply_rs_rs
            return multiply_rs_rs(self, A)
        elif isinstance(A, np.ndarray):
            # multiply SimpleSparse by matrix or vector, multiply_rs_matrix uses slicing
            indices, xs = self.array()
            if A.ndim == 3:
                return multiply_rs_matrix_3d(indices, xs, A)
            elif A.ndim == 4:
                return multiply_rs_matrix_4d(indices, xs, A)
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x}' for (i, m), x in self.elements.items()) + '}'
        return f'SimpleSparseVector({formatted})'


def multiply_basis(t1, t2):
    """Matrix multiplication operation mapping two sparse basis elements to another."""
    # equivalent to formula in Proposition 2 of Sequence Space Jacobian paper, but with
    # signs of i and j flipped to reflect different sign convention used here
    i, m = t1
    j, n = t2
    k = i + j
    if i >= 0:
        if j >= 0:
            l = max(m, n - i)
        elif k >= 0:
            l = max(m, n - k)
        else:
            l = max(m + k, n)
    else:
        if j <= 0:
            l = max(m + j, n)
        else:
            l = max(m, n) + min(-i, j)
    return k, l


def multiply_rs_rs(s1, s2):
    """Matrix multiplication operation on two SimpleSparse objects."""
    # iterate over all pairs (i, m) -> x and (j, n) -> y in objects,
    # add all pairwise products to get overall product
    elements = {}
    for im, x in s1.elements.items():
        for jn, y in s2.elements.items():
            kl = multiply_basis(im, jn)
            if kl in elements:
                elements[kl] += x @ y
            else:
                elements[kl] = x @ y
    return SimpleSparseVector(elements)


@njit
def multiply_rs_matrix_3d(indices, xs, A):
    raise NotImplementedError


@njit
def multiply_rs_matrix_4d(indices, xs, A):
    raise NotImplementedError
