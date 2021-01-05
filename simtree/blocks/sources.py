"""Provides some simple source blocks"""
import numpy as np

from simtree.model import Signal


def constant(parent, value):
    """Return a signal that provides a constant value."""
    value = np.atleast_1d(value)
    return Signal(parent,
                  shape=value.shape,
                  value=value)
