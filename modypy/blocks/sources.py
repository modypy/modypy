"""Provides some simple source blocks"""
import numpy as np
from modypy.model import Signal, signal_function
from modypy.model.ports import AbstractSignal, ShapeType


def constant(value):
    """
    Create a constant signal

    Args:
      value: The value of the signal

    Returns:
      A signal with the required constant value
    """

    return Signal(shape=np.shape(value), value=value)


@signal_function
def time(system_state):
    """Signal returning the current time

    The value of this signal is the value of the `time` property of the
    system state."""
    return system_state.time


class FunctionSignal(AbstractSignal):
    """Signal whose value is determined by evaluating a function on the value
    of a sequence of other signals.

    Args:
        function: The function to be called for calculating the value of the
            signal.
        signals: A single signal or a sequence of signals to be passed as
            arguments to the function.
        shape: The shape of the signal."""

    def __init__(self, function, signals, shape: ShapeType = ()):
        super().__init__(shape)
        self.function = function
        try:
            len(signals)
        except TypeError:
            signals = [signals]
        self.signals = signals

    def __call__(self, system_state):
        signal_values = [signal(system_state) for signal in self.signals]
        return self.function(*signal_values)
