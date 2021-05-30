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

    return Signal(shape=np.shape(value),
                  value=value)
