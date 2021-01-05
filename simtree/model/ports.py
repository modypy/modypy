"""
Provides classes for creating ports and signals.
"""
from functools import cached_property
import math
from typing import Union, Sequence


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

    @cached_property
    def size(self):
        """The size of the port. Equivalent to the product of the dimensions of
        the port."""
        return math.prod(self.shape)

    @property
    def signal(self):
        """The signal this port is connected to or None."""
        if self.reference == self:
            return None
        return self.reference.signal

    @property
    def slice(self):
        return self.signal.indices

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


class Signal(Port):
    """
    A signal provides the value for all ports connected to it.
    """

    def __init__(self, owner, shape: ShapeType, function: callable):
        Port.__init__(self, owner, shape)
        self.signal_index = self.owner.system.allocate_signal_lines(self.size)
        self.owner.system.signals.add(self)
        self.function = function

    @property
    def signal(self):
        """The signal this port is connected. As this is a signal, it returns
        itself."""
        return self

    @property
    def slice(self):
        return slice(self.signal_index,
                     self.signal_index+self.size)
