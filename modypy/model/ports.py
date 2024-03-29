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
from abc import abstractmethod

import numpy as np
import operator
from typing import Optional, Sequence, Tuple, Union

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


class AbstractSignal:
    """A signal is a function of the system state.

    Args:
        shape: The shape of the signal, which is either an empty tuple for a
            scalar signal (the default), an integer for a vector signal of the
            given dimension or a tuple of integers for a multi-dimensional
            signal.
    """

    def __init__(self, shape: ShapeType = ()):
        try:
            len(shape)
        except TypeError:
            shape = (shape,)
        self.shape = shape
        self.size = functools.reduce(operator.mul, self.shape, 1)

    def connect(self, other: "AbstractSignal"):
        """Connect this signal to another signal.

        Args:
          other: The signal to connect to

        Raises:
          ShapeMismatchError: raised if the shapes of the signals do not match
          MultipleSignalsError: raised if both sides of the connection are
            already connected to different signals
        """
        other.reference = self

    @property
    def reference(self) -> "AbstractSignal":
        return self

    @reference.setter
    def reference(self, other: "AbstractSignal"):
        if other.reference is not self:
            raise MultipleSignalsError

    @property
    def signal(self) -> "AbstractSignal":
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Evaluate the value of the signal"""


class Port(AbstractSignal):
    """A port is a placeholder for a signal that can be connected to other ports
    or a signal."""

    def __init__(self, shape: ShapeType = ()):
        AbstractSignal.__init__(self, shape)
        self._reference = self

    @property
    def reference(self):
        """The port referenced by this port"""
        if self._reference is not self:
            # Try to further shorten the reference path
            self._reference = self._reference.reference
        return self._reference

    @reference.setter
    def reference(self, other):
        if self._reference is self:
            self._reference = other
        elif other.reference is other:
            other.reference = self
        else:
            self._reference.reference = other

    @property
    def signal(self) -> Optional[AbstractSignal]:
        """The signal referenced by this port or ``None`` if this port is not
        connected to any signal."""
        if self._reference is not self:
            return self.reference.signal
        return None

    def connect(self, other: "AbstractSignal"):
        """Connect this port to a signal.

        Args:
          other: The signal to connect to

        Raises:
          ShapeMismatchError: raised if the shapes of the signals do not match
          MultipleSignalsError: raised if both sides of the connection are
            already connected to different signals
        """
        if self.shape != other.shape:
            # It is an error if the shapes of the ports do not match.
            raise ShapeMismatchError(
                "Shape (%s) of left port does not match "
                "shape (%s) of right port" % (self.shape, other.shape)
            )
        self.reference = other

    def __call__(self, *args, **kwargs):
        if self.size == 0:
            return np.empty(self.shape)
        if self.signal is None:
            raise PortNotConnectedError()
        # pylint does not recognize that self.signal is either None or Port,
        # which _is_ callable.
        # pylint: disable=not-callable
        return self.signal(*args, **kwargs)


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
    the_signal = Signal(*args, value=user_function, **kwargs)

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
        """Descriptor that will return itself when accessed on a class, but a
        unique Signal instance when accessed on a class instance."""

        def __init__(self, function):
            self.name = None
            self.function = function

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, instance, owner):
            if instance is None:
                return self
            signal_name = "__signal_%s" % self.name
            the_signal = getattr(instance, signal_name, None)
            if the_signal is None:
                the_signal = Signal(
                    *args,
                    value=self.function.__get__(instance, owner),
                    **kwargs
                )
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

    def __init__(self, owner, shape: ShapeType = (), value=None):
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

        return slice(self.input_index, self.input_index + self.size)

    @property
    def input_range(self):
        """A range object that represents the indices of this input in the
        inputs vector."""

        return range(self.input_index, self.input_index + self.size)

    def __call__(self, system_state):
        return system_state.get_input_value(self)
