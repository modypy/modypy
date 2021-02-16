"""
Values in a model are transported by signals. In MoDyPy signals are always
real-valued, and may be multi-dimensional.

The value of a signal is defined by a value function, which may depend on the
value of any system state or signal, which are made accessible by the
:class:`DataProvider <modypy.model.evaluation.DataProvider>` object passed to
it.

Signals differ from states in that they do not have their own memory - although
they may be based on the values of states, which represent memory.

Ports serve as symbolic placeholders for signals, and may be connected to each
other and to signals using the ``connect`` method. In fact, each signal is also
a port.
"""
import functools
import operator
from typing import Union, Sequence, Tuple

import numpy as np


ShapeType = Union[int, Sequence[int], Tuple[int]]


class Port:
    """A port is a structural element of a system that can be connected to a
    signal."""

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
        """A slice object that represents the indices of this port in the
        signals vector."""

        return self.signal.signal_slice

    @property
    def signal_range(self):
        """A range object that represents the indices of this port in the
        signals vector."""

        return self.signal.signal_range

    def connect(self, other):
        """Connect this port to another port.

        Args:
          other: The other port to connect to

        Raises:
          ShapeMismatchError: if the shapes of the ports do not match
          MultipleSignalsError: if both ports are already connected to
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
    """An ``OutputPort`` is a special port that is considered to be an output of
    the system. In simulation or steady-state determination, output ports play
    a special role."""
    def __init__(self, owner, shape: ShapeType = 1):
        Port.__init__(self, owner, shape)
        self.output_index = self.owner.system.allocate_output_lines(self.size)
        self.owner.system.outputs.append(self)

    @property
    def output_slice(self):
        """A slice object that represents the indices of this output in the
        outputs vector."""
        return slice(self.output_index,
                     self.output_index+self.size)

    @property
    def output_range(self):
        """A range object that represents the indices of this output in the
        outputs vector."""
        return range(self.output_index,
                     self.output_index+self.size)


class Signal(Port):
    """A signal provides the value for all ports connected to it."""

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
        """A slice object that represents the indices of this signal in the
        signals vector."""
        return slice(self.signal_index,
                     self.signal_index+self.size)

    @property
    def signal_range(self):
        """A range object that represents the indices of this signal in the
        signals vector."""
        return range(self.signal_index,
                     self.signal_index+self.size)


class InputSignal(Signal):
    """An ``InputSignal`` is a special kind of signal that is considered an
    input into the system. In simulation and linearization, input signals play a
    special role."""
    def __init__(self, owner, shape: ShapeType = 1, value=0):
        Signal.__init__(self, owner, shape, value)
        self.input_index = self.owner.system.allocate_input_lines(self.size)
        self.owner.system.inputs.append(self)

    @property
    def input_slice(self):
        """A slice object that represents the indices of this input in the
        inputs vector."""

        return slice(self.input_index,
                     self.input_index+self.size)

    @property
    def input_range(self):
        """A range object that represents the indices of this input in the
        inputs vector."""

        return range(self.input_index,
                     self.input_index+self.size)
