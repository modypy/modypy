"""
Defines meta-properties for states.
"""
from functools import cached_property
import math

import numpy as np


class State:
    """
    A state describes a portion of the state of a block.
    """
    def __init__(self, owner, derivative_function, shape=1, initial_condition=None):
        self.owner = owner
        self.derivative_function = derivative_function
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        if initial_condition is None:
            self.initial_condition = np.zeros(self.shape)
        else:
            self.initial_condition = initial_condition

        self.state_index = self.owner.system.allocate_state_lines(self.size)
        self.owner.system.states.add(self)

    @cached_property
    def size(self):
        """The size of the state. Equivalent to the product of the dimensions
        of the state."""
        return math.prod(self.shape)
