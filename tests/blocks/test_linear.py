# pylint: disable=missing-module-docstring
import numpy as np
import pytest
from modypy.blocks.linear import (
    Gain,
    InvalidLTIException,
    LTISystem,
    Sum,
    gain,
    sum_signal,
)
from modypy.blocks.sources import constant
from modypy.model import InputSignal, Signal, System, SystemState
from numpy import testing as npt


def _make_array_or_scalar(shape):
    if shape == ():
        return np.float_()
    return np.empty(shape=shape)


@pytest.mark.parametrize(
    "input_shape,system_shape,output_shape,feed_through_shape",
    [
        # More than two dimensions on system matrix
        (0, (2, 2, 2), 0, 0),
        # Empty system matrix
        (0, (0, 2), 0, 0),
        # Non-square system matrix
        (0, (2, 3), 0, 0),
        # Scalar system matrix with mismatching inputs
        (2, (), 0, 0),
        # Scalar input matrix with mismatching states
        ((), (2, 2), 0, 0),
        # Column input vector with mismatching states
        (2, (3, 3), (), ()),
        # Input matrix with mismatching states
        ((2, 2), (3, 3), (), ()),
        # Three-dimensional input matrix
        ((2, 2, 2), (3, 3), (), ()),
        # Scalar output matrix with multiple states
        ((2, 1), (2, 2), (), 0),
        # Row output vector with mismatching states
        ((2, 1), (2, 2), 3, ()),
        # Output matrix with mismatching states
        ((2, 1), (2, 2), (3, 3), ()),
        # Non-square output matrix
        ((2, 1), (2, 2), (3, 3, 3), ()),
        # Scalar feed-through matrix with more than one input
        ((2, 2), (2, 2), (2, 2), ()),
        # Scalar feed-through matrix with more than one output
        (2, (2, 2), (2, 2), ()),
        # Row feed-through vector without outputs
        (2, (2, 2), (0, 2), 1),
        # Row feed-through vector with more than one output
        (2, (2, 2), (2, 2), 1),
        # Row feed-through vector mismatching number of inputs
        ((2, 2), (2, 2), (1, 2), 1),
        # Feed-through matrix height mismatching number of outputs
        ((2, 2), (2, 2), (1, 2), (2, 2)),
        # Feed-through matrix width mismatching number of inputs
        ((2, 2), (2, 2), (1, 2), (1, 1)),
        # Feed-through matrix with invalid number of dimensions
        ((2, 2), (2, 2), (1, 2), (1, 1, 1)),
    ],
)
def test_invalid_lti_configurations(
    input_shape, system_shape, output_shape, feed_through_shape
):
    """Test the detection of invalid LTI matrix configurations"""
    input_matrix = _make_array_or_scalar(shape=input_shape)
    system_matrix = _make_array_or_scalar(shape=system_shape)
    output_matrix = _make_array_or_scalar(shape=output_shape)
    feed_through_matrix = _make_array_or_scalar(shape=feed_through_shape)

    system = System()
    with pytest.raises(InvalidLTIException):
        LTISystem(
            parent=system,
            input_matrix=input_matrix,
            system_matrix=system_matrix,
            output_matrix=output_matrix,
            feed_through_matrix=feed_through_matrix,
        )


def test_gain_class():
    system = System()
    gain_block = Gain(system, k=[[1, 2], [3, 4]])
    gain_in = constant(value=[3, 4])
    gain_block.input.connect(gain_in)

    npt.assert_almost_equal(gain_block.output(None), [11, 25])


def test_scalar_gain_function():
    gain_in = constant(value=[[3, 4], [5, 6]])
    gain_signal = gain(gain_matrix=3, input_signal=gain_in)

    npt.assert_almost_equal(gain_signal(None), [[9, 12], [15, 18]])


def test_gain_function():
    gain_in = constant(value=[3, 4])
    gain_signal = gain(gain_matrix=[[1, 2], [3, 4]], input_signal=gain_in)

    npt.assert_almost_equal(gain_signal(None), [11, 25])


@pytest.mark.parametrize(
    "channel_weights, output_size, inputs, expected_output",
    [
        ([1, 1], 1, [1, -1], [0]),
        ([1, 2], 2, [[1, 2], [3, 4]], [7, 10]),
        ([1, 2, 3], 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [30, 36, 42]),
    ],
)
def test_sum_block(channel_weights, output_size, inputs, expected_output):
    system = System()
    sum_block = Sum(
        system, channel_weights=channel_weights, output_size=output_size
    )
    for idx in range(len(inputs)):
        input_signal = InputSignal(system, shape=output_size, value=inputs[idx])
        sum_block.inputs[idx].connect(input_signal)

    system_state = SystemState(time=0, system=system)
    actual_output = sum_block.output(system_state)
    npt.assert_almost_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "channel_weights, output_size, inputs, expected_output",
    [
        ([1, 1], 1, [1, -1], [0]),
        ([1, 2], 2, [[1, 2], [3, 4]], [7, 10]),
        ([1, 2, 3], 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [30, 36, 42]),
    ],
)
def test_sum_signal(channel_weights, output_size, inputs, expected_output):
    system = System()
    input_signals = [
        InputSignal(system, shape=output_size, value=input_value)
        for input_value in inputs
    ]
    sum_result = sum_signal(input_signals, gains=channel_weights)

    system_state = SystemState(time=0, system=system)
    actual_output = sum_result(system_state)
    npt.assert_almost_equal(actual_output, expected_output)


def test_sum_signal_shape_mismatch():
    """Test the shape mismatch exception of ``sum_signal``."""

    signal_1 = Signal()
    signal_2 = Signal(shape=2)

    with pytest.raises(ValueError):
        sum_signal(input_signals=(signal_1, signal_2))


def test_sum_signal_gain_mismatch():
    """Test the gain mismatch exception of ``sum_signal``."""

    signal_1 = Signal()
    signal_2 = Signal()

    with pytest.raises(ValueError):
        sum_signal(input_signals=(signal_1, signal_2), gains=(1, 1, 1))
