"""Blocks for linear, time-invariant systems"""
import numpy as np
from simtree.blocks import LeafBlock


class LTISystem(LeafBlock):
    """
    Implementation of a linear, time-invariant block of the format

      dx/dt = system_matrix * x +        input_matrix * u
      y     = output_matrix * x + feed_through_matrix * u

    The matrices system_matrix, input_matrix, output_matrix and
    feed_through_matrix define the state and output behaviour of the block.
    """

    def __init__(self,
                 system_matrix,
                 input_matrix,
                 output_matrix,
                 feed_through_matrix,
                 **kwargs):
        system_matrix = np.atleast_2d(system_matrix)
        input_matrix = np.atleast_2d(input_matrix)
        output_matrix = np.atleast_2d(output_matrix)
        feed_through_matrix = np.atleast_2d(feed_through_matrix)
        LeafBlock.__init__(self,
                           num_states=system_matrix.shape[1],
                           num_inputs=input_matrix.shape[1],
                           num_outputs=output_matrix.shape[0],
                           **kwargs)
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

    def state_update_function(self, time, *args):
        del time  # unused
        if self.num_states > 0:
            dxdt = np.matmul(self.system_matrix, args[0])
            if self.num_inputs > 0:
                dxdt += np.matmul(self.input_matrix, args[1])
            return dxdt
        return np.empty(0)

    def output_function(self, time, *args):
        del time  # unused
        if self.num_states > 0 and self.num_inputs > 0:
            return np.matmul(self.output_matrix, args[0]) + \
                   np.matmul(self.feed_through_matrix, args[1])
        if self.num_states > 0:
            return np.matmul(self.output_matrix, args[0])
        if self.num_inputs > 0:
            return np.matmul(self.feed_through_matrix, args[0])
        return np.zeros(self.num_outputs)


class Gain(LeafBlock):
    """
    A simple linear gain block.

    Provides the input scaled by the constant gain as output.
    """

    def __init__(self, k, **kwargs):
        k = np.asarray(k)
        LeafBlock.__init__(
            self, num_inputs=k.shape[1], num_outputs=k.shape[0], **kwargs)
        self.k = k

    def output_function(self, time, inputs):
        del time  # unused
        return np.matmul(self.k, inputs)


class Sum(LeafBlock):
    """
    A linear weighted sum block.

    This block may have a number of inputs which are interpreted a vectors of
    common dimension. The output of the block is calculated as the weighted
    sum of the inputs.

    The block has `channel_dim` outputs, which represent the elements of the
    weighted vector sum of the inputs.
    In that case, inputs `0:channel_dim` represent the first vector, inputs
    `channel_dim:2*channel_dim` represent the second vector, etc.

    The `channel_weights` give the factors by which the individual channels are
    weighted in the sum.
    """

    def __init__(self,
                 channel_weights,
                 channel_dim=1,
                 **kwargs):
        channel_weights = np.asarray(channel_weights)
        LeafBlock.__init__(self,
                           num_inputs=channel_weights.size * channel_dim,
                           num_outputs=channel_dim,
                           **kwargs)
        self.channel_weights = channel_weights
        self.channel_dim = channel_dim

    def output_function(self, time, inputs):
        del time  # unused
        inputs = np.asarray(inputs).reshape(-1, self.channel_dim)
        return np.matmul(self.channel_weights, inputs)
