import numpy as np
from numba import njit

import asymptotic
from ._simple_sparse_abstract import SimpleSparseAbstract


class SimpleSparseScalar(SimpleSparseAbstract):
    """
    SimpleSparse class implementation for the Scalar objects.
    See parent class documentation for the more details
    """

    @property
    def asymptotic_time_invariant(self):
        indices, xs = self.array()
        tau = np.max(np.abs(indices[:, 0])) + 1  # how far out do we go?
        v = np.zeros(2 * tau - 1)
        # v[indices[:, 0]+tau-1] = xs
        v[-indices[:, 0] + tau - 1] = xs  # switch from asymptotic ROW to asymptotic COLUMN
        return asymptotic.AsymptoticTimeInvariant(v)

    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""
        return self + np.zeros((T, T))

    # magic methods -----------------------------------------------------------

    def __add__(self, A):
        if isinstance(A, SimpleSparseScalar):
            # add SimpleSparse to SimpleSparse, combining dicts, summing x when (i, m) overlap
            elements = self.elements.copy()
            for im, x in A.elements.items():
                if im in elements:
                    elements[im] += x
                    # safeguard to retain sparsity: disregard extremely small elements (num error)
                    if abs(elements[im]) < 1E-14:
                        del elements[im]
                else:
                    elements[im] = x
            return SimpleSparseScalar(elements)
        else:
            # add SimpleSparse to T*T matrix
            if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
                return NotImplemented
            T = A.shape[0]

            # fancy trick to do this efficiently by writing A as flat vector
            # then (i, m) can be mapped directly to NumPy slicing!
            A = A.flatten()  # use flatten, not ravel, since we'll modify A and want a copy
            for (i, m), x in self.elements.items():
                if i < 0:
                    A[T * (-i) + (T + 1) * m::T + 1] += x
                else:
                    A[i + (T + 1) * m:(T - i) * T:T + 1] += x
            return A.reshape((T, T))

    def __eq__(self, s):
        return self.elements == s.elements

    def __matmul__(self, A):
        if isinstance(A, SimpleSparseScalar):
            # multiply SimpleSparse by SimpleSparse, simple analytical rules in multiply_rs_rs
            return multiply_rs_rs(self, A)
        elif isinstance(A, np.ndarray):
            # multiply SimpleSparse by matrix or vector, multiply_rs_matrix uses slicing
            indices, xs = self.array()
            if A.ndim == 2:
                return multiply_rs_matrix(indices, xs, A)
            elif A.ndim == 1:
                return multiply_rs_matrix(indices, xs, A[:, np.newaxis])[:, 0]
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return SimpleSparseScalar({im: a * x for im, x in self.elements.items()})

    def __radd__(self, A):
        try:
            return self + A
        except Exception:
            print(self)
            print(A)
            raise

    def __repr__(self):
        formatted = '{' + ', '.join(f'({i}, {m}): {x:.3f}' for (i, m), x in self.elements.items()) + '}'
        return f'SimpleSparse({formatted})'

    def __rmatmul__(self, A):
        # multiplication rule when this object is on right (will only be called when left is matrix)
        # for simplicity, just use transpose to reduce this to previous cases
        return (self.T @ A.T).T


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
                elements[kl] += x * y
            else:
                elements[kl] = x * y
    return SimpleSparseScalar(elements)


@njit
def multiply_rs_matrix(indices, xs, A):
    """Matrix multiplication of SimpleSparse object ('indices' and 'xs') and matrix A.
    Much more computationally demanding than multiplying two SimpleSparse (which is almost
    free with simple analytical formula), so we implement as jitted function."""
    n = indices.shape[0]
    T = A.shape[0]
    S = A.shape[1]
    Aout = np.zeros((T, S))

    for count in range(n):
        # for Numba to jit easily, SimpleSparse with basis elements '(i, m)' with coefs 'x'
        # was stored in 'indices' and 'xs'
        i = indices[count, 0]
        m = indices[count, 1]
        x = xs[count]

        # loop faster than vectorized when jitted
        # directly use def of basis element (i, m), displacement of i and ignore first m
        if i == 0:
            for t in range(m, T):
                for s in range(S):
                    Aout[t, s] += x * A[t, s]
        elif i > 0:
            for t in range(m, T - i):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
        else:
            for t in range(m - i, T):
                for s in range(S):
                    Aout[t, s] += x * A[t + i, s]
    return Aout
