"""Blocks for linear, time-invariant systems"""
import numpy as np
from functools import partial
from modypy.model import Block, Port, Signal, SignalState, State


class InvalidLTIException(RuntimeError):
    """An exception which is raised when the specification of an LTI is invalid.
    """


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

        # Determine the number of states and the shape of the state
        if np.isscalar(system_matrix):
            self.state_shape = ()
            num_states = 1
        else:
            system_matrix = np.asarray(system_matrix)
            if (system_matrix.ndim == 2 and
                    system_matrix.shape[0] > 0 and
                    system_matrix.shape[0] == system_matrix.shape[1]):
                self.state_shape = system_matrix.shape[0]
                num_states = self.state_shape
            else:
                raise InvalidLTIException('The system matrix must be a scalar '
                                          'or a non-empty square matrix')

        # Determine the number of inputs and the shape of the input signal
        if np.isscalar(input_matrix):
            if num_states > 1:
                raise InvalidLTIException('There is more than one state, but '
                                          'the input matrix is neither empty, '
                                          'nor a vector or a matrix')
            num_inputs = 1
            self.input_shape = ()
        else:
            input_matrix = np.asarray(input_matrix)
            if input_matrix.ndim == 1:
                # The input matrix is a column vector => one input
                if num_states != input_matrix.shape[0]:
                    raise InvalidLTIException('The height of the input matrix '
                                              'must match the number of states')
                num_inputs = 1
            elif input_matrix.ndim == 2:
                # The input matrix is a matrix
                if num_states != input_matrix.shape[0]:
                    raise InvalidLTIException('The height of the input matrix '
                                              'does not match the number of '
                                              'states')
                num_inputs = input_matrix.shape[1]
            else:
                raise InvalidLTIException('The input matrix must be empty,'
                                          'a scalar, a vector or a matrix')
            self.input_shape = num_inputs

        # Determine the number of outputs and the shape of the output array
        if np.isscalar(output_matrix):
            if num_states > 1:
                raise InvalidLTIException('There is more than one state, but '
                                          'the output matrix is neither an '
                                          'empty, a vector nor a matrix')
            num_outputs = 1
            self.output_shape = ()
        else:
            output_matrix = np.asarray(output_matrix)
            if output_matrix.ndim == 1:
                # The output matrix is a row vector => one output
                if num_states != output_matrix.shape[0]:
                    raise InvalidLTIException('The width of the output matrix '
                                              'does not match the number of '
                                              'states')
                num_outputs = 1
            elif output_matrix.ndim == 2:
                # The output matrix is a matrix
                if num_states != output_matrix.shape[1]:
                    raise InvalidLTIException('The width of the output matrix '
                                              'does not match the number of '
                                              'states')
                num_outputs = output_matrix.shape[0]
            else:
                raise InvalidLTIException('The output matrix must be empty, a'
                                          'scalar, a vector or a matrix')
            self.output_shape = num_outputs

        if np.isscalar(feed_through_matrix):
            if not (num_inputs == 1 and num_outputs == 1):
                raise InvalidLTIException('A scalar feed-through matrix is '
                                          'only allowed for systems with '
                                          'exactly one input and one output')
        else:
            feed_through_matrix = np.asarray(feed_through_matrix)
            if feed_through_matrix.ndim == 1:
                # A vector feed_through_matrix is interpreted as row vector,
                # so there must be exactly one output.
                if num_outputs == 0:
                    raise InvalidLTIException('The feed-through matrix for a '
                                              'system without outputs must be'
                                              'empty')
                elif num_outputs > 1:
                    raise InvalidLTIException('The feed-through matrix for a '
                                              'system with more than one '
                                              'output must be a matrix')
                if feed_through_matrix.shape[0] != num_inputs:
                    raise InvalidLTIException('The width of the feed-through '
                                              'matrix must match the number of '
                                              'inputs')
            elif feed_through_matrix.ndim == 2:
                if feed_through_matrix.shape[0] != num_outputs:
                    raise InvalidLTIException('The height of the feed-through '
                                              'matrix must match the number of '
                                              'outputs')
                if feed_through_matrix.shape[1] != num_inputs:
                    raise InvalidLTIException('The width of the feed-through '
                                              'matrix must match the number of '
                                              'inputs')
            else:
                raise InvalidLTIException('The feed-through matrix must be '
                                          'empty, a scalar, a vector or a '
                                          'matrix')

        self.system_matrix = system_matrix
        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.feed_through_matrix = feed_through_matrix

        self.input = Port(shape=self.input_shape)
        self.state = State(self,
                           shape=self.state_shape,
                           derivative_function=self.state_derivative,
                           initial_condition=initial_condition)
        self.output = Signal(shape=self.output_shape,
                             value=self.output_function)

    def state_derivative(self, data):
        """Calculates the state derivative for the system"""
        if self.state.shape == ():
            derivative = self.system_matrix * self.state(data)
        else:
            derivative = np.matmul(self.system_matrix, self.state(data))
        if self.input.shape == ():
            derivative += self.input_matrix * self.input(data)
        elif self.input.size > 0:
            derivative += np.matmul(self.input_matrix, self.input(data))
        return derivative

    def output_function(self, data):
        """Calculates the output for the system"""
        if self.state.shape == ():
            output = self.output_matrix * self.state(data)
        else:
            output = np.matmul(self.output_matrix, self.state(data))
        if self.input.shape == ():
            output += self.feed_through_matrix * self.input(data)
        elif self.input.size > 0:
            output += np.matmul(self.feed_through_matrix, self.input(data))
        return output


class Gain(Block):
    """A simple linear gain block.

    Provides the input scaled by the constant gain as output.

    This class is deprecated. Use ``gain`` instead.
    """

    def __init__(self, parent, k):
        Block.__init__(self, parent)
        self.k = np.atleast_2d(k)

        self.input = Port(shape=self.k.shape[0])
        self.output = Signal(shape=self.k.shape[1],
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

    return np.matmul(gain_matrix, input_signal(data))


def gain(gain_matrix, input_signal):
    """
    Create a signal that represents the product of the given gain matrix and the
    value of the given input signal.

    Args:
        gain_matrix: The gain matrix
        input_signal: The input signal to consider

    Returns:
        A signal that represents the product of the gain matrix and the value of
        the input signal.
    """

    # Determine the shape of the output signal
    output_shape = (gain_matrix @ np.zeros(input_signal.shape)).shape
    return Signal(shape=output_shape,
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

        self.inputs = [Port(shape=self.output_size)
                       for _ in range(self.channel_weights.shape[0])]
        self.output = Signal(shape=self.output_size,
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
        signal_sum = signal_sum + np.dot(gain_value, signal(data))
    return signal_sum


def sum_signal(input_signals, gains=None):
    """
    Create a signal that represents the sum of the input signals multiplied by
    the given gains.

    The signals must have the same shape and there must be exactly as many
    entries in the ``gains`` tuple as there are input signals.

    Args:
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
        raise ValueError('The shapes of the input signals do not match')
    if len(input_signals) != len(gains):
        raise ValueError('There must be as many gains as there are '
                         'input signals')

    return Signal(shape=shape,
                  value=partial(_sum_function,
                                input_signals,
                                gains))


def integrator(owner, input_signal, initial_condition=None):
    """
    Create a state-signal that provides the integrated value of the input
    callable.

    The resulting signal will have the same shape as the input callable.

    Args:
        owner: The owner of the integrator
        input_signal: A callable accepting an object implementing the system
            object access protocol and providing the value of the derivative
        initial_condition: The initial condition of the integrator
            (default: ``None``)

    Returns:
        A state-signal that provides the integrated value of the input signal
    """

    return SignalState(owner,
                       shape=input_signal.shape,
                       derivative_function=input_signal,
                       initial_condition=initial_condition)
