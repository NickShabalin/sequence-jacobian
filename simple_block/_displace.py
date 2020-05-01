import numpy as np


class Displace(np.ndarray):
    """This class makes time displacements of a time path, given the steady-state value.
    Needed for SimpleBlock.td()"""

    def __new__(cls, x, ss=None, name='UNKNOWN'):
        obj = np.asarray(x).view(cls)
        obj.ss = ss
        obj.name = name
        return obj

    def __call__(self, index):
        if index != 0:
            if self.ss is None:
                raise KeyError(f'Trying to call {self.name}({index}), but steady-state {self.name} not given!')
            newx = np.empty_like(self)
            if index > 0:
                newx[:-index] = self[index:]
                newx[-index:] = self.ss
            else:
                newx[-index:] = self[:index]
                newx[:-index] = self.ss
            return newx
        else:
            return self