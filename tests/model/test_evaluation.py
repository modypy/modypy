"""
Tests for ``modypy.model.evaluator``
"""
import numpy as np
import numpy.testing as npt

import pytest

from modypy.blocks.sources import constant
from modypy.model import System, State, Port, Signal, InputSignal, OutputPort, ZeroCrossEventSource
from modypy.model.evaluation import Evaluator, AlgebraicLoopError, PortNotConnectedError


def test_evaluator():
    """Test the ``Evaluator`` class"""

    system = System()
    state_a = State(system,
                    shape=(3, 3),
                    initial_condition=[[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                    derivative_function=(lambda data: data.signals[input_a]))
    state_a_dep = State(system,
                        shape=(3, 3),
                        initial_condition=[[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]],
                        derivative_function=(lambda data: data.inputs[input_a]))
    state_b = State(system,
                    shape=3,
                    initial_condition=[10, 11, 12],
                    derivative_function=(lambda data: np.r_[13, 14, 15]))
    state_b1 = State(system,
                     shape=3,
                     derivative_function=(lambda data: data.states[state_b]))
    state_b2 = State(system,
                     shape=3,
                     derivative_function=None)
    input_a = InputSignal(system, shape=(3, 3), value=np.eye(3))
    input_c = InputSignal(system, value=1)
    input_d = InputSignal(system, shape=2, value=[2, 3])
    signal_c = constant(system, value=16)
    signal_d = Signal(system, shape=2, value=(lambda data: [17, 19]))
    output_a = OutputPort(system, shape=(3, 3))
    output_c = OutputPort(system)
    event_a = ZeroCrossEventSource(system, event_function=(lambda data: 23))
    event_b = ZeroCrossEventSource(system, event_function=(lambda data: 25))

    evaluator = Evaluator(time=0, system=system)

    output_a.connect(input_a)
    output_c.connect(input_c)

    # Check the initial state
    npt.assert_almost_equal(evaluator.state[state_a.state_slice],
                            state_a.initial_condition.flatten())
    npt.assert_almost_equal(evaluator.state[state_a_dep.state_slice],
                            state_a_dep.initial_condition.flatten())
    npt.assert_almost_equal(evaluator.state[state_b.state_slice],
                            state_b.initial_condition.flatten())
    npt.assert_almost_equal(evaluator.state[state_b1.state_slice],
                            state_b1.initial_condition.flatten())
    npt.assert_almost_equal(evaluator.state[state_b2.state_slice],
                            np.zeros(state_b2.size))

    # Check the derivative property
    npt.assert_almost_equal(evaluator.state_derivative[state_a.state_slice],
                            input_a.value.flatten())
    npt.assert_almost_equal(evaluator.state_derivative[state_a_dep.state_slice],
                            input_a.value.flatten())
    npt.assert_almost_equal(evaluator.state_derivative[state_b.state_slice],
                            state_b.derivative_function(None).flatten())
    npt.assert_almost_equal(evaluator.state_derivative[state_b1.state_slice],
                            state_b.initial_condition.flatten())

    # Check the inputs property
    npt.assert_almost_equal(evaluator.inputs[input_a.input_slice],
                            input_a.value.flatten())
    npt.assert_almost_equal(evaluator.inputs[input_c.input_slice],
                            input_c.value.flatten())
    npt.assert_almost_equal(evaluator.inputs[input_d.input_slice],
                            input_d.value.flatten())

    # Check the signals property
    npt.assert_almost_equal(evaluator.signals[input_a.signal_slice],
                            input_a.value.flatten())
    npt.assert_almost_equal(evaluator.signals[input_c.signal_slice],
                            input_c.value.flatten())
    npt.assert_almost_equal(evaluator.signals[input_d.signal_slice],
                            input_d.value.flatten())
    npt.assert_almost_equal(evaluator.signals[signal_c.signal_slice],
                            signal_c.value.flatten())
    npt.assert_almost_equal(evaluator.signals[signal_d.signal_slice],
                            signal_d.value(None))
    npt.assert_almost_equal(evaluator.signals[output_a.signal_slice],
                            input_a.value.flatten())
    npt.assert_almost_equal(evaluator.signals[output_c.signal_slice],
                            input_c.value.flatten())

    # Check the outputs property
    npt.assert_almost_equal(evaluator.outputs[output_a.output_slice],
                            evaluator.signals[output_a.signal_slice])
    npt.assert_almost_equal(evaluator.outputs[output_c.output_slice],
                            evaluator.signals[output_c.signal_slice])

    # Check the event_values property
    npt.assert_almost_equal(evaluator.event_values[event_a.event_index],
                            event_a.event_function(None))
    npt.assert_almost_equal(evaluator.event_values[event_b.event_index],
                            event_b.event_function(None))

    # Check the get_state_value function
    npt.assert_almost_equal(evaluator.get_state_value(state_a),
                            state_a.initial_condition)
    npt.assert_almost_equal(evaluator.get_state_value(state_a_dep),
                            state_a_dep.initial_condition)
    npt.assert_almost_equal(evaluator.get_state_value(state_b),
                            state_b.initial_condition)


def test_evaluator_with_initial_state():
    """Test the ``Evaluator`` class with an explicitly specified initial
    state"""

    system = System()
    State(system,
          shape=(3, 3),
          derivative_function=None,
          initial_condition=np.eye(3))

    initial_state = np.arange(system.num_states)
    evaluator = Evaluator(time=0, system=system, state=initial_state)

    npt.assert_almost_equal(evaluator.state,
                            initial_state)


def test_evaluator_with_initial_inputs():
    """Test the ``Evaluator`` class with explicitly specified initial inputs"""

    system = System()
    InputSignal(system, shape=(3, 3), value=np.eye(3))
    InputSignal(system, value=123)
    InputSignal(system, shape=2, value=[456, 789])

    initial_inputs = np.arange(system.num_inputs)
    evaluator = Evaluator(time=0, system=system, inputs=initial_inputs)

    npt.assert_almost_equal(evaluator.inputs,
                            initial_inputs)


def test_algebraic_loop_error():
    """Test the detection of algebraic loops"""

    system = System()
    port_a = Port(system)
    port_b = Port(system)
    signal_a = Signal(system,
                      value=(lambda data: data.signals[port_b]))
    signal_b = Signal(system,
                      value=(lambda data: data.signals[port_a]))

    port_a.connect(signal_a)
    port_b.connect(signal_b)

    evaluator = Evaluator(time=0, system=system)
    with pytest.raises(AlgebraicLoopError):
        evaluator.signals


def test_port_not_connected_error():
    """Test the detection of unconnected ports"""

    system = System()
    port = Port(system)

    evaluator = Evaluator(time=0, system=system)
    with pytest.raises(PortNotConnectedError):
        evaluator.get_port_value(port)
