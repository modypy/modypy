"""Blocks for linear, time-invariant systems"""
import numpy as np
from modypy.model import Block, Port, State, Signal


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
        state = data.states[self.state]
        inputs = data.signals[self.input]
        return (self.system_matrix @ state) + (self.input_matrix @ inputs)

    def output_function(self, data):
        """Calculates the output for the system"""
        state = data.states[self.state]
        inputs = data.signals[self.input]
        return (self.output_matrix @ state) \
            + (self.feed_through_matrix @ inputs)


class Gain(Block):
    """A simple linear gain block.

    Provides the input scaled by the constant gain as output.
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
        return self.k @ data.signals[self.input]


class Sum(Block):
    """A linear weighted sum block.

    This block may have a number of inputs which are interpreted as vectors of
    common dimension. The output of the block is calculated as the weighted
    sum of the inputs.

    The ``channel_weights`` give the factors by which the individual channels
    are weighted in the sum.
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
            inputs[port_idx] = data.signals[self.inputs[port_idx]]
        return self.channel_weights @ inputs
