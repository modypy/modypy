"""
Provides classes for constructing systems and block hierarchies.
"""
from functools import cached_property

import numpy as np


class System:
    """
    A system is a group of interacting ``Block`` instances.
    """
    def __init__(self):
        self.signal_line_count = 0
        self.signals = set()
        self.state_line_count = 0
        self.states = set()

    @property
    def system(self):
        """The system itself"""
        return self

    @property
    def initial_condition(self):
        """The initial condition vector for the state of this system"""
        initial = np.zeros(self.state_line_count)
        for state in self.states:
            start_index = state.state_index
            end_index = start_index + state.size
            initial[start_index:end_index] = state.initial_condition.flatten()
        return initial

    def allocate_signal_lines(self, count):
        """
        Allocate a sequence of consecutive signal lines.

        :param count: The number of signal lines to allocate
        :return: The index of the first signal line allocated
        """
        start_index = self.signal_line_count
        self.signal_line_count += count
        return start_index

    def allocate_state_lines(self, count):
        """
        Allocate a sequence of consecutive state lines.

        :param count: The number of state lines to allocate
        :return: The index of the first state line allocated
        """
        start_index = self.state_line_count
        self.state_line_count += count
        return start_index


class Block:
    """
    A block describes the interface and behaviour of a part of a system.

    The interface of a block is mainly defined by ports, by which signal
    connections can be established to other parts of the system.

    The behaviour of a block is defined by states.
    """
    def __init__(self, parent):
        self.parent = parent

    @cached_property
    def system(self):
        """The system this block is part of"""
        return self.parent.system
