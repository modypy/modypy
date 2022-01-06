"""Provides some simple source blocks"""
import numpy as np
from modypy.model import Signal, signal_function


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
