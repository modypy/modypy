"""Provides some simple source blocks"""
import numpy as np

from modypy.model import Signal


def constant(value):
    """
    Create a constant signal

    Args:
      value: The value of the signal

    Returns:
      A signal with the required constant value
    """

    value = np.atleast_1d(value)
    return Signal(shape=value.shape,
                  value=value)
