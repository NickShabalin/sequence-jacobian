"""Part 1: SimpleBlock class and @simple decorator to generate it"""

import logging

import numpy as np

import utils
from ._displace import Displace
from ._ignore import IgnoreFactory
from ._perturb import PerturbFactory
from ._reporter import ReporterFactory
from ._simple_sparse import SimpleSparseFactory


def simple(f):
    return SimpleBlock(f)


class SimpleBlock:
    """Generated from simple block written in Dynare-ish style and decorated with @simple, e.g.
    
    @simple
    def production(Z, K, L, alpha):
        Y = Z * K(-1) ** alpha * L ** (1 - alpha)
        return Y

    which is a SimpleBlock that takes in Z, K, L, and alpha, all of which can be either constants
    or series, and implements a Cobb-Douglas production function, noting that for production today
    we use the capital K(-1) determined yesterday.
    
    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        self._log = logging.getLogger(f"{self.__class__.__name__}::{f.__name__}")
        self._log.debug(f"Initializing")

        self.f = f
        self.input_list = utils.input_list(f)
        self.output_list = utils.output_list(f)
        self.inputs = set(self.input_list)
        self.outputs = set(self.output_list)

    def __repr__(self):
        return f"<SimpleBlock '{self.f.__name__}'>"

    def ss(self, *args, **kwargs):
        self._log.debug("ss method call")
        args = [IgnoreFactory().ignore(x) for x in args]
        kwargs = {k: IgnoreFactory().ignore(v) for k, v in kwargs.items()}
        return self.f(*args, **kwargs)

    def td(self, ss, **kwargs):
        self._log.debug("td method call")
        kwargs_new = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            kwargs_new[k] = Displace(v, ss=ss.get(k, None), name=k)

        for k in self.input_list:
            if k not in kwargs_new:
                kwargs_new[k] = IgnoreFactory().ignore(ss[k])

        return dict(zip(self.output_list, utils.make_tuple(self.f(**kwargs_new))))

    def jac(self, ss, T=None, shock_list=None, h=1E-5):
        """Assemble nested dict of Jacobians

        Parameters
        ----------
        ss : dict,
            steady state values
        T : int, optional
            number of time periods for explicit T*T Jacobian
            if omitted, more efficient SimpleSparse objects returned
        shock_list : list of str, optional
            names of input variables to differentiate wrt; if omitted, assume all inputs
        h : float, optional
            radius for symmetric numerical differentiation

        Returns
        -------
        J : dict of {str: dict of {str: array(T,T)}}
            J[o][i] for output o and input i gives Jacobian of o with respect to i
            This Jacobian is a SimpleSparse object or, if T specific, a T*T matrix, omitted by convention
            if zero
        """
        self._log.debug("jac method call")
        if shock_list is None:
            shock_list = self.input_list

        raw_derivatives = {o: {} for o in self.output_list}

        # initialize dict of default inputs k on which we'll evaluate simple blocks
        # each element is 'Ignore' object containing ss value of input k that ignores
        # time displacement, i.e. k(3) in a simple block will evaluate to just ss k
        # x_ss_new = {k: Ignore(ss[k]) for k in self.input_list} TODO: uncomment and delete following lines
        x_ss_new = {}
        for k in self.input_list:
            x_ss_new[k] = IgnoreFactory().ignore(ss[k])

        # loop over all inputs k which we want to differentiate
        for k in shock_list:
            # detect all non-zero time displacements i with which k(i) appears in f
            # wrap steady-state values in Reporter class (similar to Ignore but adds any time
            # displacements to shared set), then feed into f
            reporter = ReporterFactory().reporter(ss[k])
            x_ss_new[k] = reporter
            self.f(**x_ss_new)
            relevant_displacements = reporter.myset

            # add zero by default (Reporter can't detect, since no explicit call k(0) required)
            relevant_displacements.add(0)

            # evaluate derivative with respect to input at each displacement i
            for i in relevant_displacements:
                # perturb k(i) up by +h from steady state and evaluate f
                x_ss_new[k] = PerturbFactory().perturb(ss[k], h, i)
                y_up_all = utils.make_tuple(self.f(**x_ss_new))

                # perturb k(i) down by -h from steady state and evaluate f
                x_ss_new[k] = PerturbFactory().perturb(ss[k], -h, i)
                y_down_all = utils.make_tuple(self.f(**x_ss_new))

                # for each output o of f, if affected, store derivative in rawderivatives[o][k][i]
                # this builds up Jacobian rawderivatives[o][k] of output o with respect to input k
                # which is 'sparsederiv' dict mapping time displacements 'i' to derivatives
                for y_up, y_down, o in zip(y_up_all, y_down_all, self.output_list):
                    if utils.compare_y_up_down(y_up, y_down):
                        sparsederiv = raw_derivatives[o].setdefault(k, {})
                        sparsederiv[i] = (y_up - y_down) / (2 * h)

            # replace our Perturb object for k with Ignore object, so we can run with other k
            x_ss_new[k] = IgnoreFactory().ignore(ss[k])

        # process raw_derivatives to return either SimpleSparse objects or (if T provided) matrices
        J = {o: {} for o in self.output_list}
        for o in self.output_list:
            for k in raw_derivatives[o].keys():
                if T is None:
                    J[o][k] = SimpleSparseFactory(raw_derivatives[o][k]).build_from_dimple_diagonals()
                else:
                    J[o][k] = SimpleSparseFactory(raw_derivatives[o][k]).build_from_dimple_diagonals().matrix(T)

        return J
