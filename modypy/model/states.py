"""
Provides classes for defining states.
"""
import functools
import operator

import numpy as np

from modypy.model import Signal


class State:
    """A state describes a portion of the state of a block."""

    def __init__(self,
                 owner,
                 derivative_function,
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
    def __init__(self, owner, derivative_function=None, shape=1, initial_condition=None):
        State.__init__(self, owner, derivative_function, shape, initial_condition)
        Signal.__init__(self, owner, shape, value=self.output_function)

    def output_function(self, data):
        """The output function that returns the state"""
        return data.states[self]
