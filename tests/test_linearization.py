import pytest

import numpy as np
import numpy.testing as npt

from modypy.linearization import system_jacobian
from modypy.model import System, OutputPort
from fixtures.models import first_order_lag, first_order_lag_no_input, damped_oscillator, lti_gain
from modypy.steady_state import SteadyStateConfiguration, find_steady_state


@pytest.fixture(params=[3, 5, 7, 9, 11])
def interpolation_order(request):
    return request.param


@pytest.mark.parametrize(
    "param",
    [
        # 1 input, 1 output, 1 state
        first_order_lag(time_constant=1, initial_value=10),

        # no input, with state
        first_order_lag_no_input(time_constant=1, initial_value=10),

        # 1 input, 2 states, 1 output
        damped_oscillator(mass=100,
                          spring_coefficient=1.,
                          damping_coefficient=20),   # critically damped
        damped_oscillator(mass=100,
                          spring_coefficient=0.5,
                          damping_coefficient=20),  # overdamped
    ])
def test_steady_state_linearisation(param, interpolation_order):
    system, lti, sim_time = param

    # Find the steady state of the system
    steady_state_config = SteadyStateConfiguration(system)
    # Constrain all outputs to zero
    for output in system.outputs:
        steady_state_config.signal_bounds[output.signal_slice, :] = 0
    # Find the steady state
    sol = find_steady_state(steady_state_config)

    assert sol.success
    assert sol.state.size == system.num_states
    assert sol.inputs.size == system.num_inputs

    npt.assert_allclose(sol.evaluator.state_derivative,
                        np.zeros(system.num_states),
                        rtol=0,
                        atol=1E-5)
    npt.assert_allclose(sol.evaluator.outputs,
                        np.zeros(system.num_outputs))

    # Linearize the system
    A, B, C, D = system_jacobian(system,
                                 time=0,
                                 x_ref=sol.state,
                                 u_ref=sol.inputs,
                                 order=interpolation_order)

    # Check the matrices
    npt.assert_almost_equal(A, lti.system_matrix)
    npt.assert_almost_equal(B, lti.input_matrix)
    npt.assert_almost_equal(C, lti.output_matrix)
    npt.assert_almost_equal(D, lti.feed_through_matrix)

    # Get the full jacobian
    jac = system_jacobian(system,
                          time=0,
                          x_ref=sol.state,
                          u_ref=sol.inputs,
                          order=interpolation_order,
                          single_matrix=True)
    A = jac[:system.num_states, :system.num_states]
    B = jac[:system.num_states, system.num_states:]
    C = jac[system.num_states:, :system.num_states]
    D = jac[system.num_states:, system.num_states:]
    npt.assert_almost_equal(A, lti.system_matrix)
    npt.assert_almost_equal(B, lti.input_matrix)
    npt.assert_almost_equal(C, lti.output_matrix)
    npt.assert_almost_equal(D, lti.feed_through_matrix)


def test_output_only():
    system = System()
    OutputPort(system)

    # Try to linearize the system
    with pytest.raises(ValueError):
        system_jacobian(system, time=0, x_ref=[], u_ref=[])
