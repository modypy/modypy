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
    state_a = State(
        system,
        shape=(3, 3),
        initial_condition=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        derivative_function=None,
    )
    state_b = State(
        system,
        shape=3,
        initial_condition=[10, 11, 12],
        derivative_function=None,
    )
    input_c = InputSignal(system, value=1)
    input_d = InputSignal(system, shape=2, value=[2, 3])
    event_a = ZeroCrossEventSource(system, event_function=None)
    event_b = ZeroCrossEventSource(system, event_function=None)

    # Check the counts
    assert system.num_inputs == input_c.size + input_d.size
    assert system.num_states == state_a.size + state_b.size
    assert system.num_events == 2

    # Check the initial condition
    # This also checks that state indices are disjoint.
    initial_condition = system.initial_condition
    npt.assert_almost_equal(
        initial_condition[state_a.state_slice],
        state_a.initial_condition.ravel(),
    )
    npt.assert_almost_equal(
        initial_condition[state_b.state_slice],
        state_b.initial_condition.ravel(),
    )

    # Check the initial inputs
    # This also checks that input indices are disjoint.
    initial_input = system.initial_input
    npt.assert_almost_equal(
        initial_input[input_c.input_slice], np.atleast_1d(input_c.value).ravel()
    )
    npt.assert_almost_equal(
        initial_input[input_d.input_slice], np.atleast_1d(input_d.value).ravel()
    )

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
    state_a = State(
        system,
        shape=(3, 3),
        initial_condition=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        derivative_function=input_a,
    )
    state_a_dep = State(
        system,
        shape=(3, 3),
        initial_condition=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        derivative_function=input_a,
    )
    state_b = State(
        system,
        shape=3,
        initial_condition=[10, 11, 12],
        derivative_function=(lambda data: np.r_[13, 14, 15]),
    )
    state_b1 = State(system, shape=3, derivative_function=state_b)
    state_b2 = State(system, shape=3, derivative_function=None)
    signal_c = constant(value=16)
    event_a = ZeroCrossEventSource(system, event_function=(lambda data: 23))

    system_state = SystemState(time=0, system=system)

    # Check the initial state
    npt.assert_almost_equal(
        system_state.state[state_a.state_slice],
        state_a.initial_condition.ravel(),
    )
    npt.assert_almost_equal(
        system_state.state[state_a_dep.state_slice],
        state_a_dep.initial_condition.ravel(),
    )
    npt.assert_almost_equal(
        system_state.state[state_b.state_slice],
        state_b.initial_condition.ravel(),
    )
    npt.assert_almost_equal(
        system_state.state[state_b1.state_slice],
        state_b1.initial_condition.ravel(),
    )
    npt.assert_almost_equal(
        system_state.state[state_b2.state_slice], np.zeros(state_b2.size)
    )

    # Check the inputs property
    npt.assert_almost_equal(
        system_state.inputs[input_a.input_slice], np.ravel(input_a.value)
    )
    npt.assert_almost_equal(
        system_state.inputs[input_c.input_slice], np.ravel(input_c.value)
    )
    npt.assert_almost_equal(
        system_state.inputs[input_d.input_slice], np.ravel(input_d.value)
    )

    # Check the get_state_value function
    npt.assert_almost_equal(
        system_state.get_state_value(state_a), state_a.initial_condition
    )
    npt.assert_almost_equal(
        system_state.get_state_value(state_a_dep), state_a_dep.initial_condition
    )
    npt.assert_almost_equal(
        system_state.get_state_value(state_b), state_b.initial_condition
    )

    # Check the function access
    npt.assert_equal(
        event_a(system_state), event_a.event_function(system_state)
    )
    npt.assert_equal(state_a(system_state), state_a.initial_condition)
    npt.assert_equal(input_a(system_state), input_a.value)
    npt.assert_equal(signal_c(system_state), signal_c.value)


def test_multi_time_system_state():
    """Test the system state with multiple time instances"""

    system = System()
    state_1 = State(owner=system)
    state_2 = State(owner=system, shape=2)
    input_1 = InputSignal(owner=system)
    input_2 = InputSignal(owner=system, shape=2)

    times = np.r_[0.0, 1.0, 2.0]
    states = np.empty(shape=(system.num_states, times.shape[0]))
    inputs = np.empty(shape=(system.num_inputs, times.shape[0]))

    states[state_1.state_slice] = (1.0, 2.0, 3.0)
    states[state_2.state_slice] = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    inputs[input_1.input_slice] = (1.0, 2.0, 3.0)
    inputs[input_2.input_slice] = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    system_state = SystemState(
        system=system, time=times, state=states, inputs=inputs
    )

    npt.assert_equal(system_state[state_1], (1.0, 2.0, 3.0))
    npt.assert_equal(system_state[state_2], ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))

    npt.assert_equal(system_state[input_1], (1.0, 2.0, 3.0))
    npt.assert_equal(system_state[input_2], ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))


def test_system_state_dictionary_access():
    """Test the deprecated dictionary access for system states"""

    system = System()
    state_a = State(system, initial_condition=10)
    state_b = State(system, shape=2, initial_condition=[11, 12])
    state_c = State(
        system, shape=(2, 2), initial_condition=[[13, 14], [15, 16]]
    )

    system_state = SystemState(time=0, system=system)

    npt.assert_equal(system_state[state_a], state_a(system_state))
    npt.assert_equal(system_state[state_b], state_b(system_state))
    npt.assert_equal(system_state[state_b, 0], state_b(system_state)[0])
    npt.assert_equal(system_state[state_b, 1], state_b(system_state)[1])
    npt.assert_equal(system_state[state_c, 0, 0], state_c(system_state)[0, 0])
