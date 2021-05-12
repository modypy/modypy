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


class PortNotConnectedError(RuntimeError):
    """This exception is raised when a port is evaluated that is not connected
    to a signal"""


class MultipleSignalsError(RuntimeError):
    """This exception is raised if two ports shall be connected to each other
    that are already connected to different signals."""


class ShapeMismatchError(RuntimeError):
    """This exception is raised if two ports with incompatible shapes shall be
    connected to each other."""


class Port:
    """A port is a structural element of a system that can be connected to a
    signal."""

    def __init__(self, shape: ShapeType = (1,)):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.size = functools.reduce(operator.mul, self.shape, 1)
        self._reference = self

    @property
    def reference(self):
        """The port referenced by this port"""
        if self._reference is not self:
            # Try to further shorten the reference path
            self._reference = self._reference.reference
        return self._reference

    @reference.setter
    def reference(self, value):
        self._reference = value

    @property
    def signal(self):
        """The signal referenced by this port or ``None`` if this port is not
        connected to any signal."""
        if self._reference is not self:
            return self.reference.signal
        return None

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

    def __call__(self, *args, **kwargs):
        if self.size == 0:
            return np.empty(self.shape)
        if self.signal is None:
            raise PortNotConnectedError()
        return self.signal(*args, **kwargs)


class AbstractSignal(Port):
    """An signal is a terminal port with a defined value.

    It is connected to itself, i.e., it can only be connected to other,
    unconnected ports or to ports that are already connected to itself."""

    @property
    def signal(self):
        """The signal this signal is connected to, i.e., itself."""
        # A signal is always connected to itself
        return self


class Signal(AbstractSignal):
    """A signal is a port for which the value is defined by a callable or a
    constant."""

    def __init__(self, *args, value=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def __call__(self, *args, **kwargs):
        if callable(self.value):
            return self.value(*args, **kwargs)
        return self.value


def decorator(func):
    """Helper function to create decorators with optional arguments"""
    def _wrapper(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # We only have the function as parameter, so directly call the
            # decorator function
            return func(*args, **kwargs)

        # We have parameters, so we define a functor to use as decorator
        def _functor(user_func):
            return func(user_func, *args, **kwargs)

        return _functor

    # Be a well-behaved decorator by copying name, documentation and attributes
    _wrapper.__name__ = func.__name__
    _wrapper.__doc__ = func.__doc__
    _wrapper.__dict__.update(func.__dict__)
    return _wrapper


@decorator
def signal_function(user_function, *args, **kwargs):
    """Transform a function into a ``Signal``"""
    the_signal = Signal(user_function, *args, **kwargs)

    # Be a well-behaved decorator by copying name, documentation and attributes
    the_signal.__doc__ = user_function.__doc__
    the_signal.__name__ = user_function.__name__
    the_signal.__dict__.update(user_function.__dict__)
    return the_signal


@decorator
def signal_method(user_function, *args, **kwargs):
    """Transform a method into a ``Signal``

    The return value is a descriptor object that creates a ``Signal`` instance
    for each instance of the containing class."""

    class _SignalDescriptor:
        def __init__(self, function):
            self.name = None
            self.function = function

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, instance, owner):
            if instance is None:
                return self
            signal_name = '__signal_%s' % self.name
            the_signal = getattr(instance, signal_name, None)
            if the_signal is None:
                the_signal = Signal(self.function.__get__(instance, owner),
                                    *args, **kwargs)
                the_signal.__name__ = self.function.__name__
                the_signal.__doc__ = self.function.__doc__
                the_signal.__dict__.update(self.function.__dict__)
                setattr(instance, signal_name, the_signal)
            return the_signal

    descriptor = _SignalDescriptor(user_function)
    descriptor.__name__ = user_function.__name__
    descriptor.__doc__ = user_function.__doc__
    descriptor.__dict__.update(user_function.__dict__)
    return descriptor


class InputSignal(AbstractSignal):
    """An ``InputSignal`` is a special kind of signal that is considered an
    input into the system. In steady-state identification and linearization,
    input signals play a special role."""

    def __init__(self, owner, shape: ShapeType = (1,), value=None):
        super().__init__(shape)
        self.owner = owner
        self.input_index = self.owner.system.allocate_input_lines(self.size)
        self.owner.system.inputs.append(self)
        if value is None:
            value = np.zeros(shape)
        self.value = value

    @property
    def input_slice(self):
        """A slice object that represents the indices of this input in the
        inputs vector."""

        return slice(self.input_index,
                     self.input_index + self.size)

    @property
    def input_range(self):
        """A range object that represents the indices of this input in the
        inputs vector."""

        return range(self.input_index,
                     self.input_index + self.size)

    def __call__(self, system_state):
        return system_state.get_input_value(self)
