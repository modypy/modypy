import pytest
import numpy as np
import numpy.testing as npt

from modypy.blocks.linear import LTISystem, Gain, gain, sum_signal, Sum
from modypy.blocks.sources import constant
from modypy.model import System, InputSignal, Signal
from modypy.model.evaluation import Evaluator


def test_lti_nonsquare_state_matrix():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0]],
                  input_matrix=[[]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[]])


def test_lti_state_input_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1]],
                  output_matrix=[[1, 0]],
                  feed_through_matrix=[[1]])


def test_lti_state_output_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1]],
                  feed_through_matrix=[[1]])


def test_lti_output_feed_through_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1], [1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1], [0]])


def test_lti_input_feed_through_mismatch():
    system = System()
    with pytest.raises(ValueError):
        LTISystem(parent=system,
                  system_matrix=[[-1, 0],
                                 [0, -1]],
                  input_matrix=[[1, 1],
                                [1, 1]],
                  output_matrix=[[1, 1]],
                  feed_through_matrix=[[1]])


def test_lti_no_states():
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 1)),
                    output_matrix=np.empty((1, 0)),
                    feed_through_matrix=1)
    source = constant(system, value=1)
    source.connect(lti.input)

    evaluator = Evaluator(time=0, system=system)
    state_derivative = evaluator.get_state_derivative(lti.state)
    assert state_derivative.size == 0


def test_lti_empty():
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 0)),
                    output_matrix=np.empty((0, 0)),
                    feed_through_matrix=np.empty((0, 0)))

    evaluator = Evaluator(time=0, system=system)
    output = lti.output(evaluator)
    assert output.size == 0


def test_gain_class():
    system = System()
    gain_block = Gain(system, k=[[1, 2], [3, 4]])
    gain_in = constant(system, value=[3, 4])
    gain_block.input.connect(gain_in)

    evaluator = Evaluator(time=0, system=system)
    npt.assert_almost_equal(gain_block.output(evaluator),
                            [11, 25])


def test_gain_function():
    system = System()
    gain_in = constant(system, value=[3, 4])
    gain_signal = gain(system,
                       gain_matrix=[[1, 2], [3, 4]],
                       input_signal=gain_in)

    evaluator = Evaluator(time=0, system=system)
    npt.assert_almost_equal(gain_signal(evaluator),
                            [11, 25])


@pytest.mark.parametrize(
    "channel_weights, output_size, inputs, expected_output",
    [
        ([1, 1], 1, [1, -1], [0]),
        ([1, 2], 2, [[1, 2], [3, 4]], [7, 10]),
        ([1, 2, 3], 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [30, 36, 42]),
    ]
)
def test_sum_block(channel_weights, output_size, inputs, expected_output):
    system = System()
    sum_block = Sum(system,
                    channel_weights=channel_weights,
                    output_size=output_size)
    for idx in range(len(inputs)):
        input_signal = InputSignal(system, shape=output_size, value=inputs[idx])
        sum_block.inputs[idx].connect(input_signal)

    evaluator = Evaluator(time=0, system=system)
    actual_output = sum_block.output(evaluator)
    npt.assert_almost_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "channel_weights, output_size, inputs, expected_output",
    [
        ([1, 1], 1, [1, -1], [0]),
        ([1, 2], 2, [[1, 2], [3, 4]], [7, 10]),
        ([1, 2, 3], 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [30, 36, 42]),
    ]
)
def test_sum_signal(channel_weights, output_size, inputs, expected_output):
    system = System()
    input_signals = [InputSignal(system,
                                 shape=output_size,
                                 value=input_value)
                     for input_value in inputs]
    sum_result = sum_signal(system,
                            input_signals,
                            gains=channel_weights)

    evaluator = Evaluator(time=0, system=system)
    actual_output = sum_result(evaluator)
    npt.assert_almost_equal(actual_output, expected_output)


def test_sum_signal_shape_mismatch():
    """Test the shape mismatch exception of ``sum_signal``."""

    system = System()
    signal_1 = Signal(system)
    signal_2 = Signal(system, shape=2)

    with pytest.raises(ValueError):
        sum_result = sum_signal(system,
                                input_signals=(signal_1, signal_2))


def test_sum_signal_gain_mismatch():
    """Test the gain mismatch exception of ``sum_signal``."""

    system = System()
    signal_1 = Signal(system)
    signal_2 = Signal(system)

    with pytest.raises(ValueError):
        sum_result = sum_signal(system,
                                input_signals=(signal_1, signal_2),
                                gains=(1, 1, 1))
