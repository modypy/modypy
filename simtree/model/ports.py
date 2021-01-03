"""
Defines meta-properties for ports and signals.
"""

from typing import Union, Iterable
import math

from .block import MetaProperty

class PortConnectionError(RuntimeError):
    """
    Base class for all exceptions occurring in the context of connecting ports.
    """


class MultipleSignalsError(PortConnectionError):
    """
    Exception raised in case two ports are to be connected which are already
    connected to different signals.
    """


class ShapeMismatchError(PortConnectionError):
    """
    Exception raised in case two ports with incompatible shapes are to be
    connected.
    """


class Port(MetaProperty):
    """
    A port describes a channel by which a block can communicate with the outside
    world.

    It has a shape, which is either an integer or a tuple of integers giving
    the dimensions of the port along one or more axes. For example, a 3x3-matrix
    has the shape ``(3,3)``.
    """
    def __init__(self, shape: Union[int, Iterable[int]], *args, **kwargs):
        MetaProperty.__init__(self, *args, **kwargs)
        self.shape = shape
        if isinstance(self.shape, int):
            self.size = self.shape
        else:
            self.size = math.prod(self.shape)

    def __get__(self, block, owner):
        return self.get_instance(block)

    def __set__(self, block, other):
        port_instance = self.get_instance(block)
        port_instance.connect(other)

    def create_instance(self, block):
        return PortInstance(block, self)

    def get_instance(self, block):
        return getattr(block, "_port_%s" % self.name)

    def register_instance(self, block, instance):
        setattr(block, "_port_%s" % self.name, instance)


class PortInstance:
    """
    An instance of a port.
    """
    def __init__(self, block, port):
        self.block = block
        self.port = port
        self.reference = self

    @property
    def signal(self):
        """The signal this port is connected to, or ``None`` if it is not yet
        connected to any signal."""
        if self.reference == self:
            return None
        return self.reference.signal

    def connect(self, other):
        """
        Connect this port instance to another port instance.

        :param other: The other port instance.
        :raises ShapeMismatchError: If there is a mismatch in the shapes of the
            ports.
        :raises MultipleSignalsError: If the ports are both already connected
            to different signals.
        """
        if self.port.shape != other.port.shape:
            # There is a mismatch in shape between the ports.
            raise ShapeMismatchError()
        if self.signal is not None and other.signal is not None:
            # Both ports are already connected to a signal,
            # so we need to ensure that they are connected to the same
            if self.reference.signal != other.reference.signal:
                raise MultipleSignalsError()
        else:
            if other.signal is not None:
                # Set our reference to the other reference
                self.reference.reference = other.reference
            else:
                other.reference.reference = self.reference


class Signal(Port):
    """
    A signal is the actual source of the value associated with a port.
    By connecting to a signal, ports get their values.
    """
    def __init__(self, *args, **kwargs):
        Port.__init__(self, *args, **kwargs)

    def create_instance(self, block):
        signal_index = block.context.allocate_signals(self.size)
        return SignalInstance(block, self, signal_index)


class SignalInstance(PortInstance):
    """
    An instance of a signal.
    """
    def __init__(self, block, port, signal_index):
        PortInstance.__init__(self, block, port)
        self.signal_index = signal_index

    @property
    def signal(self):
        """
        The signal itself.
        """
        return self


class OutputSignal(Signal):
    """
    A signal that is the output of a block.

    Such a signal is associated with a method that will be used to determine
    the value of the signal.

    An output signal can be easily created using the ``output_signal``
    decorator on the respective function.
    """
    def __init__(self, method, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.method = method


def output_signal(shape=1):
    """Decorator to create an output signal.

    .. code-block:: python
        @output_signal(shape=3)
        def rpm(self, time, state, inputs):
            return state[self.omega]*30/math.pi
    """
    def output_signal_wrapper(function):
        return OutputSignal(function, shape)
    return output_signal_wrapper

# TODO: InputSignal for signals generated from trajectories
