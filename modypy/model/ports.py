"""
Provides classes for creating ports and signals.
"""
import functools
import operator
from typing import Union, Sequence

import numpy as np


ShapeType = Union[int, Sequence[int]]


class Port:
    """
    A port is a structural element of a system that can be connected to a
    signal.
    """

    def __init__(self, owner, shape: ShapeType = 1):
        self.owner = owner
        self.reference = self

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        self.size = functools.reduce(operator.mul, self.shape, 1)

    @property
    def signal(self):
        """The signal this port is connected to or None."""
        if self.reference == self:
            return None
        return self.reference.signal

    @property
    def signal_slice(self):
        """A slice object that can be used to index signal vectors"""
        return self.signal.signal_slice

    def connect(self, other):
        """
        Connect this port to another port.

        :param other: The other port to connect to
        :raises ShapeMismatchError: if the shapes of the ports do not match
        :raises MultipleSignalsError: if both ports are already connected to
            different signals
        """
        if self.shape != other.shape:
            # It is an error if the shapes of the ports do not match.
            raise ShapeMismatchError()
        if self.signal is not None and other.signal is not None:
            # Both ports are already connected to a signal.
            # It is an error if they are not connected to the same signal.
            if self.signal != other.signal:
                raise MultipleSignalsError()
        else:
            if self.signal is None:
                # We are not yet connected to a signal, so we take the reference
                # from the other port.
                self.reference.reference = other.reference
            else:
                # The other port is not yet connected to a signal, so we update
                # the reference of the other port.
                other.reference.reference = self.reference


class MultipleSignalsError(RuntimeError):
    """This exception is raised if two ports shall be connected to each other
    that are already connected to different signals."""


class ShapeMismatchError(RuntimeError):
    """This exception is raised if two ports with incompatible shapes shall be
    connected to each other."""


class OutputPort(Port):
    """
    An ``OutputPort`` is a special port that is considered to be an output of
    the system. In simulation or steady-state determination, output ports play
    a special role.
    """
    def __init__(self, owner, shape=1):
        Port.__init__(self, owner, shape)
        self.output_index = self.owner.system.allocate_output_lines(self.size)
        self.owner.system.outputs.append(self)

    @property
    def output_slice(self):
        """
        A slice object that represents the indices of this port in the outputs
        vector.
        """
        return slice(self.output_index,
                     self.output_index+self.size)


class Signal(Port):
    """
    A signal provides the value for all ports connected to it.
    """

    def __init__(self, owner, shape: ShapeType = 1, value=0):
        Port.__init__(self, owner, shape)
        self.signal_index = self.owner.system.allocate_signal_lines(self.size)
        self.owner.system.signals.append(self)
        if callable(value):
            self.value = value
        else:
            self.value = np.atleast_1d(value)

    @property
    def signal(self):
        """The signal this port is connected. As this is a signal, it returns
        itself."""
        return self

    @property
    def signal_slice(self):
        """A slice object that can be used to index signal vectors"""
        return slice(self.signal_index,
                     self.signal_index+self.size)


class InputSignal(Signal):
    """
    An ``InputSignal`` is a special kind of signal that is considered an input
    into the system. In simulation and linearization, input signals play a
    special role.
    """
    def __init__(self, owner, shape: ShapeType = 1, value=0):
        Signal.__init__(self, owner, shape, value)
        self.input_index = self.owner.system.allocate_input_lines(self.size)
        self.owner.system.inputs.append(self)

    @property
    def input_slice(self):
        """A slice object that can be used to index input vectors"""
        return slice(self.input_index,
                     self.input_index+self.size)
