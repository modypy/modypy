"""
Tests for ``modypy.model.system``
"""
import numpy as np
from numpy import testing as npt

from modypy.blocks.sources import constant
from modypy.model import State, InputSignal, ZeroCrossEventSource, SystemState
from modypy.model.system import System, Block


def test_system():
    """Test the ``System`` class"""

    system = System()
    state_a = State(system,
                    shape=(3, 3),
                    initial_condition=[[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                    derivative_function=None)
    state_b = State(system,
                    shape=3,
                    initial_condition=[10, 11, 12],
                    derivative_function=None)
    input_c = InputSignal(system, value=1)
    input_d = InputSignal(system, shape=2, value=[2, 3])
    event_a = ZeroCrossEventSource(system, event_function=None)
    event_b = ZeroCrossEventSource(system, event_function=None)

    # Check the counts
    assert system.num_inputs == input_c.size+input_d.size
    assert system.num_states == state_a.size+state_b.size
    assert system.num_events == 2

    # Check the initial condition
    # This also checks that state indices are disjoint.
    initial_condition = system.initial_condition
    npt.assert_almost_equal(initial_condition[state_a.state_slice],
                            state_a.initial_condition.flatten())
    npt.assert_almost_equal(initial_condition[state_b.state_slice],
                            state_b.initial_condition.flatten())

    # Check the initial inputs
    # This also checks that input indices are disjoint.
    initial_input = system.initial_input
    npt.assert_almost_equal(initial_input[input_c.input_slice],
                            np.atleast_1d(input_c.value).flatten())
    npt.assert_almost_equal(initial_input[input_d.input_slice],
                            np.atleast_1d(input_d.value).flatten())

    # Check that event indices are disjoint
    assert event_a.event_index != event_b.event_index


def test_block():
    """Test the ``Block`` class"""

    system = System()
    block_a = Block(system)
    block_b = Block(block_a)

    # Check the system property
    assert block_a.system == system
    assert block_b.system == system


def test_system_state():
    """Test the ``SystemState`` class"""

    system = System()

    input_a = InputSignal(system, shape=(3, 3), value=np.eye(3))
    input_c = InputSignal(system, value=1)
    input_d = InputSignal(system, shape=2, value=[2, 3])
    state_a = State(system,
                    shape=(3, 3),
                    initial_condition=[[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                    derivative_function=input_a)
    state_a_dep = State(system,
                        shape=(3, 3),
                        initial_condition=[[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]],
                        derivative_function=input_a)
    state_b = State(system,
                    shape=3,
                    initial_condition=[10, 11, 12],
                    derivative_function=(lambda data: np.r_[13, 14, 15]))
    state_b1 = State(system,
                     shape=3,
                     derivative_function=state_b)
    state_b2 = State(system,
                     shape=3,
                     derivative_function=None)
    signal_c = constant(value=16)
    event_a = ZeroCrossEventSource(system, event_function=(lambda data: 23))

    system_state = SystemState(time=0, system=system)

    # Check the initial state
    npt.assert_almost_equal(system_state.state[state_a.state_slice],
                            state_a.initial_condition.flatten())
    npt.assert_almost_equal(system_state.state[state_a_dep.state_slice],
                            state_a_dep.initial_condition.flatten())
    npt.assert_almost_equal(system_state.state[state_b.state_slice],
                            state_b.initial_condition.flatten())
    npt.assert_almost_equal(system_state.state[state_b1.state_slice],
                            state_b1.initial_condition.flatten())
    npt.assert_almost_equal(system_state.state[state_b2.state_slice],
                            np.zeros(state_b2.size))

    # Check the inputs property
    npt.assert_almost_equal(system_state.inputs[input_a.input_slice],
                            np.ravel(input_a.value))
    npt.assert_almost_equal(system_state.inputs[input_c.input_slice],
                            np.ravel(input_c.value))
    npt.assert_almost_equal(system_state.inputs[input_d.input_slice],
                            np.ravel(input_d.value))

    # Check the get_state_value function
    npt.assert_almost_equal(system_state.get_state_value(state_a),
                            state_a.initial_condition)
    npt.assert_almost_equal(system_state.get_state_value(state_a_dep),
                            state_a_dep.initial_condition)
    npt.assert_almost_equal(system_state.get_state_value(state_b),
                            state_b.initial_condition)

    # Check the function access
    npt.assert_equal(event_a(system_state),
                     event_a.event_function(system_state))
    npt.assert_equal(state_a(system_state),
                     state_a.initial_condition)
    npt.assert_equal(input_a(system_state),
                     input_a.value)
    npt.assert_equal(signal_c(system_state),
                     signal_c.value)


def test_evaluator_with_initial_state():
    """Test the ``SystemState`` class with an explicitly specified initial
    state"""

    system = System()
    State(system,
          shape=(3, 3),
          derivative_function=None,
          initial_condition=np.eye(3))

    initial_state = np.arange(system.num_states)
    system_state = SystemState(time=0, system=system, state=initial_state)

    npt.assert_almost_equal(system_state.state,
                            initial_state)


def test_evaluator_with_initial_inputs():
    """Test the ``SystemState`` class with explicitly specified initial inputs"""

    system = System()
    InputSignal(system, shape=(3, 3), value=np.eye(3))
    InputSignal(system, value=123)
    InputSignal(system, shape=2, value=[456, 789])

    initial_inputs = np.arange(system.num_inputs)
    system_state = SystemState(time=0, system=system, inputs=initial_inputs)

    npt.assert_almost_equal(system_state.inputs,
                            initial_inputs)
