"""Blocks for linear, time-invariant systems"""
from functools import partial

import numpy as np
from modypy.model import Block, Port, State, Signal, SignalState


class LTISystem(Block):
    """Implementation of a linear, time-invariant block of the following format:

      dx/dt = system_matrix * x +        input_matrix * u
      y     = output_matrix * x + feed_through_matrix * u

    The matrices ``system_matrix``, ``input_matrix``, ``output_matrix`` and
    ``feed_through_matrix`` define the state and output behaviour of the block.
    """

    def __init__(self,
                 parent,
                 system_matrix,
                 input_matrix,
                 output_matrix,
                 feed_through_matrix,
                 initial_condition=None):
        Block.__init__(self, parent)

        system_matrix = np.atleast_2d(system_matrix)
        input_matrix = np.atleast_2d(input_matrix)
        output_matrix = np.atleast_2d(output_matrix)
        feed_through_matrix = np.atleast_2d(feed_through_matrix)

        if system_matrix.shape[0] != system_matrix.shape[1]:
            raise ValueError("The system matrix must be square")
        if system_matrix.shape[0] != input_matrix.shape[0]:
            raise ValueError(
                "The height of the system matrix and the input matrix "
                "must be the same")
        if output_matrix.shape[0] != feed_through_matrix.shape[0]:
            raise ValueError(
                "The height of the output matrix and the "
                "feed-through matrix feed_through_matrix must be the same")
        if system_matrix.shape[1] != output_matrix.shape[1]:
            raise ValueError(
                "The width of the system matrix and the output matrix"
                "must be the same")
        if input_matrix.shape[1] != feed_through_matrix.shape[1]:
            raise ValueError(
                "The width of the output matrix and the feed-through "
                "matrix must be the same")

        self.system_matrix = system_matrix
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.feed_through_matrix = feed_through_matrix

        self.input = Port(self, shape=self.input_matrix.shape[1])
        self.state = State(self,
                           shape=self.system_matrix.shape[0],
                           derivative_function=self.state_derivative,
                           initial_condition=initial_condition)
        self.output = Signal(self,
                             shape=self.output_matrix.shape[0],
                             value=self.output_function)

    def state_derivative(self, data):
        """Calculates the state derivative for the system"""
        return ((self.system_matrix @ self.state(data))
                + (self.input_matrix @ self.input(data)))

    def output_function(self, data):
        """Calculates the output for the system"""
        return ((self.output_matrix @ self.state(data))
                + (self.feed_through_matrix @ self.input(data)))


class Gain(Block):
    """A simple linear gain block.

    Provides the input scaled by the constant gain as output.

    This class is deprecated. Use ``gain`` instead.
    """

    def __init__(self, parent, k):
        Block.__init__(self, parent)
        self.k = np.atleast_2d(k)

        self.input = Port(self, shape=self.k.shape[0])
        self.output = Signal(self,
                             shape=self.k.shape[1],
                             value=self.output_function)

    def output_function(self, data):
        """Calculates the output for the system

        Args:
          data: The current time, states and signals for the system.

        Returns: The input multiplied by the gain
        """
        return self.k @ self.input(data)


def _gain_function(gain_matrix, input_signal, data):
    """
    Calculate the product of the given gain matrix and the value of the signal.

    Args:
        gain_matrix: The gain (matrix) to apply
        input_signal: The input signal
        data: The data provider

    Returns:
        The product of the gain matrix and the value of the signal
    """

    return gain_matrix @ input_signal(data)


def gain(owner, gain_matrix, input_signal):
    """
    Create a signal that represents the product of the given gain matrix and the
    value of the given input signal.

    Args:
        owner: The owner of the result signal
        gain_matrix: The gain matrix
        input_signal: The input signal to consider

    Returns:
        A signal that represents the product of the gain matrix and the value of
        the input signal.
    """

    # Determine the shape of the output signal
    output_shape = (gain_matrix @ np.zeros(input_signal.shape)).shape
    return Signal(owner=owner,
                  shape=output_shape,
                  value=partial(_gain_function,
                                gain_matrix,
                                input_signal))


class Sum(Block):
    """A linear weighted sum block.

    This block may have a number of inputs which are interpreted as vectors of
    common dimension. The output of the block is calculated as the weighted
    sum of the inputs.

    The ``channel_weights`` give the factors by which the individual channels
    are weighted in the sum.

    This class is deprecated. Use ``sum_signal`` instead.
    """

    def __init__(self,
                 parent,
                 channel_weights,
                 output_size=1):
        Block.__init__(self, parent)

        self.channel_weights = np.asarray(channel_weights)
        self.output_size = output_size

        self.inputs = [Port(self, shape=self.output_size)
                       for _ in range(self.channel_weights.shape[0])]
        self.output = Signal(self,
                             shape=self.output_size,
                             value=self.output_function)

    def output_function(self, data):
        """Calculates the output for the system

        Args:
          data: The time, states and signals of the system

        Returns:
            The sum of the input signals
        """
        inputs = np.empty((len(self.inputs), self.output_size))
        for port_idx in range(len(self.inputs)):
            inputs[port_idx] = self.inputs[port_idx](data)
        return self.channel_weights @ inputs


def _sum_function(signals, gains, data):
    """
    Calculate the sum of the values of the given signals multiplied by the
    given gains.

    Args:
        signals: A tuple of signals
        gains: A tuple of gains
        data: The data provider

    Returns:
        The sum of the values of the given signals multiplied by the given
        gains
    """

    signal_sum = 0
    for signal, gain_value in zip(signals, gains):
        signal_sum = signal_sum + gain_value * signal(data)
    return signal_sum


def sum_signal(owner, input_signals, gains=None):
    """
    Create a signal that represents the sum of the input signals multiplied by
    the given gains.

    The signals must have the same shape and there must be exactly as many
    entries in the ``gains`` tuple as there are input signals.

    Args:
        owner: The owner of the sum signal
        input_signals: A tuple of input signals
        gains:  A tuple of gains for the input signals. Optional: Default value
            is all ones.

    Returns:
        A signal that represents the sum of the input signals
    """

    if gains is None:
        gains = np.ones(len(input_signals))

    shape = input_signals[0].shape
    if any(signal.shape != shape for signal in input_signals):
        raise ValueError("The shapes of the input signals do not match")
    if len(input_signals) != len(gains):
        raise ValueError("There must be as many gains as there are "
                         "input signals")

    return Signal(owner=owner,
                  shape=shape,
                  value=partial(_sum_function,
                                input_signals,
                                gains))


def _integrator_derivative(input_signal, data):
    """Derivative function for an integrator"""

    return input_signal(data)


def integrator(owner, input_signal, initial_condition=None):
    """
    Create a state-signal that provides the integrated value of the input
    signal.

    The resulting signal will have the same shape as the input signal.

    Args:
        owner: The owner of the integrator
        input_signal: The input signal to integrate
        initial_condition: The initial condition of the integrator
            (default: ``None``)

    Returns:
        A state-signal that provides the integrated value of the input signal
    """

    return SignalState(owner,
                       shape=input_signal.shape,
                       derivative_function=partial(_integrator_derivative,
                                                   input_signal),
                       initial_condition=initial_condition)
