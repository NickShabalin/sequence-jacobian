import numpy as np

from ._simple_sparse_abstract import SimpleSparseAbstract
from ._simple_sparse_scalar import SimpleSparseScalar
from ._simple_sparse_vector import SimpleSparseVector


class SimpleSparseFactory:
    # TODO: validate if all elements are of one type and the same shape\size (for vectors)
    _SCALAR_FACTORY = 0
    _VECTOR_FACTORY = 1

    def __init__(self, elements: dict):
        self._elements = elements

    @property
    def _factory_type(self) -> int:
        first_element_val = tuple(self._elements.values())[0]
        if np.isscalar(first_element_val):
            return self._SCALAR_FACTORY
        elif isinstance(first_element_val, np.ndarray):
            return self._VECTOR_FACTORY
        else:
            raise TypeError

    def build(self) -> SimpleSparseAbstract:
        if self._factory_type == self._SCALAR_FACTORY:
            return SimpleSparseScalar(self._elements)
        elif self._factory_type == self._VECTOR_FACTORY:
            return SimpleSparseVector(self._elements)
        else:
            raise NotImplementedError

    def build_from_dimple_diagonals(self) -> SimpleSparseAbstract:
        if self._factory_type == self._SCALAR_FACTORY:
            return SimpleSparseScalar.from_simple_diagonals(self._elements)
        elif self._factory_type == self._VECTOR_FACTORY:
            return SimpleSparseVector.from_simple_diagonals(self._elements)
        else:
            raise NotImplementedError
