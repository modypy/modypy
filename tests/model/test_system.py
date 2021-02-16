"""
Tests for ``modypy.model.system``
"""
import numpy as np
import numpy.testing as npt

from modypy.model import State, InputSignal, OutputPort, ZeroCrossEventSource
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
    output_a = OutputPort(system, shape=(3, 3))
    output_c = OutputPort(system)
    event_a = ZeroCrossEventSource(system, event_function=None)
    event_b = ZeroCrossEventSource(system, event_function=None)

    # Check the counts
    assert system.num_inputs == input_c.size+input_d.size
    assert system.num_outputs == output_a.size+output_c.size
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

    # Check that output indices are disjoint
    assert (output_a.output_index+output_a.size <= output_c.output_index or
            output_c.output_index+output_c.size <= output_a.output_index)

    # Check that signal indices are disjoint
    assert (input_c.signal_index+input_c.size <= input_d.signal_index or
            input_d.signal_index+input_d.size <= input_c.signal_index)

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
