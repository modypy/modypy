"""
States represent the memory of a dynamical system. In MoDyPy, states are always
real-valued, and may be multi-dimensional.

For each state, a derivative function may be defined which describes the
evolution of the value of the state over time. The value of that derivative
function may depend on the value of any system state or signal, which are made
accessible by the :class:`DataProvider <modypy.model.evaluation.DataProvider>`
object passed to it.

States may also be updated by :mod:`event listeners <modypy.model.events>`.

States are represented as instances of the :class:`State` class. In addition,
:class:`SignalState` instances are states that are also signals.
"""
import functools
import operator

import numpy as np

from modypy.model import Signal


class State:
    """A state describes a portion of the state of a block.

    Args:
        owner: The owner of the state
        derivative_function: The derivative function of the state
            (Default: 0)
        shape: The shape of the state (Default: 1)
        initial_condition: The initial value of the state (Default: 0)
    """

    def __init__(self,
                 owner,
                 derivative_function=None,
                 shape=1,
                 initial_condition=None):
        self.owner = owner
        self.derivative_function = derivative_function
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        if initial_condition is None:
            self.initial_condition = np.zeros(self.shape)
        else:
            self.initial_condition = np.asarray(initial_condition)

        self.size = functools.reduce(operator.mul, self.shape, 1)

        self.state_index = self.owner.system.allocate_state_lines(self.size)
        self.owner.system.states.append(self)

    @property
    def state_slice(self):
        """A slice that can be used to index state vectors"""
        return slice(self.state_index,
                     self.state_index + self.size)


class SignalState(State, Signal):
    """A state that also provides itself as an output signal."""

    def __init__(self,
                 owner,
                 derivative_function=None,
                 shape=1,
                 initial_condition=None):
        State.__init__(self,
                       owner,
                       derivative_function,
                       shape,
                       initial_condition)
        Signal.__init__(self, owner, shape, value=self.output_function)

    def output_function(self, data):
        """The output function that returns the state"""
        return data.states[self]
