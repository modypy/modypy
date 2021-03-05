"""Blocks for modelling discontinuities"""
from functools import partial

import numpy as np

from modypy.model import Signal


def _saturation_function(input_signal, lower_limit, upper_limit, data):
    """Function to return a saturated signal"""

    return np.minimum(np.maximum(input_signal(data), lower_limit), upper_limit)


def saturation(owner, input_signal, lower_limit, upper_limit):
    """
    Limit a signal to lower and upper limits.

    Args:
        owner: The owner for the output signal
        input_signal: The input signal
        lower_limit: The lower limit
        upper_limit: The upper limit

    Returns:
        The output signal with the limited value
    """

    return Signal(owner=owner,
                  shape=input_signal.shape,
                  value=partial(_saturation_function,
                                input_signal,
                                lower_limit,
                                upper_limit))
