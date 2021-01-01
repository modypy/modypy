import math
import pytest
import numpy as np
import numpy.testing as npt
import scipy.signal as signal

from fixtures.models import \
    first_order_lag, first_order_lag_no_input, \
    damped_oscillator, damped_oscillator_with_events, \
    propeller_model, \
    bouncing_ball_model, \
    oscillator_with_sine_input, sine_input_with_gain, \
    sine_source

from simtree.blocks.linear import LTISystem
from simtree.compiler import compile
from simtree.simulator import Simulator


@pytest.fixture(params=[
    first_order_lag(time_constant=1, initial_value=10),

    first_order_lag_no_input(time_constant=1, initial_value=10),

    damped_oscillator(mass=100, spring_coefficient=1., damping_coefficient=20),  # critically damped
    damped_oscillator(mass=100, spring_coefficient=0.5, damping_coefficient=20),  # overdamped
    damped_oscillator(mass=100, spring_coefficient=2., damping_coefficient=20),  # underdamped
])
def lti_system_with_reference(request):
    lti_system, ref_time = request.param
    assert isinstance(lti_system, LTISystem)

    ref_system = signal.StateSpace(lti_system.system_matrix,
                                   lti_system.input_matrix,
                                   lti_system.output_matrix,
                                   lti_system.feed_through_matrix)

    return lti_system, ref_system, ref_time


def test_lti_simulation(lti_system_with_reference):
    sys, ref_system, ref_time = lti_system_with_reference

    # Compile and run the block
    sys_compiled = compile(sys)

    simulator = Simulator(sys_compiled, t0=0, t_bound=ref_time,
                          initial_condition=sys_compiled.initial_condition)
    message = simulator.run()

    # Simulation must be successful
    assert message is None
    assert simulator.status == "finished"

    # Determine the system response and state values of the reference system
    ref_time, ref_output, ref_state = signal.lsim2(ref_system,
                                                   X0=sys_compiled.initial_condition,
                                                   T=simulator.result.time,
                                                   U=None,
                                                   rtol=simulator.integrator_options['rtol'],
                                                   atol=simulator.integrator_options['atol'])

    # Determine the output values at the simulated times as per the reference function
    npt.assert_allclose(ref_time,
                        simulator.result.time)
    npt.assert_allclose(simulator.result.state,
                        ref_state)
    npt.assert_allclose(simulator.result.output.flatten(),
                        ref_output)


class MockupIntegrator:
    """
    Mockup integrator class to force integration error.
    """

    def __init__(self, fun, t0, y0, t_bound):
        del fun  # unused
        del t_bound  # unused
        self.status = "running"
        self.t = t0
        self.y = y0

    def step(self):
        self.status = "failed"
        return "failed"


def test_lti_simulation_failure(lti_system_with_reference):
    sys, ref_function, sim_time = lti_system_with_reference
    del ref_function  # unused

    # Compile and run the block
    sys_compiled = compile(sys)

    simulator = Simulator(sys_compiled,
                          t0=0,
                          t_bound=sim_time,
                          integrator_constructor=MockupIntegrator,
                          integrator_options={})
    message = simulator.run()

    # Integration must fail
    assert message == "failed"


@pytest.mark.parametrize(
    "params",
    [
        (bouncing_ball_model(initial_condition=[0, 10, 1, 0]), 10),
        oscillator_with_sine_input(mass=1,
                                   spring_coefficient=40,
                                   damping_coefficient=20,
                                   initial_value=10,
                                   omega=20),
        sine_input_with_gain(omega=10),
        sine_source(omega=10),
        damped_oscillator_with_events()
    ])
def test_events(params):
    sys, sim_time = params
    del sim_time  # unused

    # Compile and run the block
    sys_compiled = compile(sys)

    simulator = Simulator(sys_compiled, t0=0, t_bound=10.0)
    message = simulator.run()

    # Check for successful run
    assert message is None
    assert simulator.status == "finished"

    # Check that event value is close to zero at the event
    event_idx = np.flatnonzero(simulator.result.events[:, 0])

    assert event_idx.size > 0

    evfun_at_event = np.empty(event_idx.size)

    for idx in range(event_idx.size):
        evidx = event_idx[idx]
        evfun_at_event[idx] = sys_compiled.event_function(simulator.result.time[evidx],
                                                          simulator.result.state[evidx, :],
                                                          simulator.result.output[evidx, :])[0]

    npt.assert_allclose(evfun_at_event, np.zeros_like(
        evfun_at_event), atol=1E-7)


@pytest.mark.parametrize(
    "system, input_function",
    [
        (damped_oscillator(mass=1, spring_coefficient=40, damping_coefficient=20, initial_value=10),
         (lambda t: np.sin(2 * math.pi * t))
         ),

        (
                (
                        propeller_model(),
                        10
                ),
                (lambda t: [np.sin(2 * math.pi * t), 1.29])
        )
    ])
def test_input_function(system, input_function):
    sysroot, sim_time = system

    # Compile and run the block
    sys_compiled = compile(sysroot)

    simulator = Simulator(sys_compiled, t0=0, t_bound=sim_time, input_callable=input_function)
    message = simulator.run()

    # Check for successful run
    assert message is None
    assert simulator.status == "finished"

    # TODO: Check for correspondence between inputs in result and values from input function
