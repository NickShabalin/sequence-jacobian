"""Part 2: SimpleSparse class to represent and work with sparse Jacobians of SimpleBlocks"""

import abc
import logging

import numpy as np


class SimpleSparseAbstract(abc.ABC):
    """
    Efficient representation of sparse linear operators, which are linear combinations of basis
    operators represented by pairs (i, m), where i is the index of diagonal on which there are 1s
    (measured by # above main diagonal) and m is number of initial entries missing.

    Examples of such basis operators:
        - (0, 0) is identity operator
        - (0, 2) is identity operator with first two '1's on main diagonal missing
        - (1, 0) has 1s on diagonal above main diagonal: "left-shift" operator
        - (-1, 1) has 1s on diagonal below main diagonal, except first column

    The linear combination of these basis operators that makes up a given SimpleSparse object is
    stored as a dict 'elements' mapping (i, m) -> x.

    The Jacobian of a SimpleBlock is a SimpleSparse operator combining basis elements (i, 0). We need
    the more general basis (i, m) to ensure closure under multiplication.

    These (i, m) correspond to the Q_(-i, m) operators defined for Proposition 2 of the Sequence Space
    Jacobian paper. The flipped sign in the code is so that the index 'i' matches the k(i) notation
    for writing SimpleBlock functions.

    The "dunder" methods x.__add__(y), x.__matmul__(y), x.__rsub__(y), etc. in Python implement infix
    operations x + y, x @ y, y - x, etc. Defining these allows us to use these more-or-less
    interchangeably with ordinary NumPy matrices.
    """

    # when performing binary operations on SimpleSparse and a NumPy array, use SimpleSparse's rules
    __array_priority__ = 1000

    # Constructors ------------------------------------------------------------

    def __init__(self, elements):
        self._log = logging.getLogger(f"{self.__class__.__name__}")
        self._log.debug(f"Initializing. Elements: {elements}")
        self.elements = elements
        self.indices, self.xs = None, None

    @classmethod
    def from_simple_diagonals(cls, elements):
        """Take dict i -> x, i.e. from SimpleBlock differentiation, convert to SimpleSparse (i, 0) -> x"""
        return cls({(i, 0): x for i, x in elements.items()})

    # non - abstract methods --------------------------------------------------

    def array(self):
        """
        Rewrite dict (i, m) -> x as pair of NumPy arrays, one size-N*2 array of ints with rows (i, m)
        and one size-N array of floats with entries x.
        This is needed for Numba to take as input. Cache for efficiency.
        """
        if self.indices is not None:
            return self.indices, self.xs
        else:
            indices, xs = zip(*self.elements.items())
            self.indices, self.xs = np.array(indices), np.array(xs)
            return self.indices, self.xs

    @property
    def T(self):
        """Transpose"""
        return type(self)({(-i, m): x for (i, m), x in self.elements.items()})

    # abstract methods --------------------------------------------------------

    @property
    @abc.abstractmethod
    def asymptotic_time_invariant(self):
        pass

    @abc.abstractmethod
    def matrix(self, T):
        """Return matrix giving first T rows and T columns of matrix representation of SimpleSparse"""

    # non abstract magic methods ----------------------------------------------

    def __mul__(self, a):
        if not np.isscalar(a): return NotImplemented
        return type(self)({im: a * x for im, x in self.elements.items()})

    def __neg__(self):
        return type(self)({im: -x for im, x in self.elements.items()})

    def __pos__(self):
        return self

    # TODO: log errors, fix broad exception
    def __radd__(self, A):
        try:
            return self + A
        except:
            print(self)
            print(A)
            raise

    def __rmatmul__(self, A):
        # multiplication rule when this object is on right (will only be called when left is matrix)
        # for simplicity, just use transpose to reduce this to previous cases
        return (self.T @ A.T).T

    def __rmul__(self, a):
        return self * a

    def __rsub__(self, A):
        return -self + A

    def __sub__(self, A):
        # slightly inefficient implementation with temporary for simplicity
        return self + (-A)

    # abstract magic methods --------------------------------------------------

    @abc.abstractmethod
    def __add__(self, A):
        pass

    @abc.abstractmethod
    def __eq__(self, s):
        pass

    @abc.abstractmethod
    def __matmul__(self, A):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass
