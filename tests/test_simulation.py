import bisect
import math

import numpy as np
import numpy.testing as npt
import pytest
import scipy.signal

from fixtures.models import \
    first_order_lag, first_order_lag_no_input, \
    damped_oscillator, damped_oscillator_with_events, \
    BouncingBall
from modypy.blocks.discrete import zero_order_hold
from modypy.blocks.linear import LTISystem, integrator
from modypy.blocks.sources import constant
from modypy.model import SystemState, System, Clock, State, ZeroCrossEventSource
from modypy.simulation import Simulator, ExcessiveEventError, _find_event_time


@pytest.fixture(params=[
    first_order_lag(time_constant=1, initial_value=10),

    first_order_lag_no_input(time_constant=1, initial_value=10),

    damped_oscillator(mass=100,
                      spring_coefficient=1.,
                      damping_coefficient=20),  # critically damped
    damped_oscillator(mass=100,
                      spring_coefficient=0.5,
                      damping_coefficient=20),  # overdamped
    damped_oscillator(mass=100,
                      spring_coefficient=2.,
                      damping_coefficient=20),  # underdamped

    damped_oscillator_with_events(mass=100,
                                  spring_coefficient=2.,
                                  damping_coefficient=20),  # underdamped

])
def lti_system_with_reference(request):
    system, lti_system, ref_time, outputs = request.param
    assert isinstance(lti_system, LTISystem)

    ref_system = scipy.signal.StateSpace(lti_system.system_matrix,
                                         lti_system.input_matrix,
                                         lti_system.output_matrix,
                                         lti_system.feed_through_matrix)
    return system, ref_system, ref_time, outputs


def test_lti_simulation(lti_system_with_reference):
    sys, ref_system, ref_time, _ = lti_system_with_reference

    simulator = Simulator(sys, start_time=0)
    message = simulator.run_until(ref_time)

    # Simulation must be successful
    assert message is None

    # Check correspondence of the result with intermediate values
    for idx in range(simulator.result.time.shape[0]):
        time = simulator.result.time[idx]
        inputs = simulator.result.inputs[:, idx]
        state = simulator.result.state[:, idx]

        system_state = SystemState(time=time, system=sys, state=state, inputs=inputs)
        npt.assert_almost_equal(system_state.inputs, inputs)
        npt.assert_almost_equal(system_state.state, state)

    # Check that states are properly mapped in the result
    for state in sys.states:
        from_result = simulator.result.state[state.state_slice]
        from_result = from_result.reshape(state.shape+(-1,))
        npt.assert_equal(state(simulator.result), from_result)

    # Check that inputs are properly mapped in the result
    for input_signal in sys.inputs:
        from_result = simulator.result.inputs[input_signal.input_slice]
        from_result = from_result.reshape(input_signal.shape+(-1,))
        npt.assert_equal(input_signal(simulator.result), from_result)

    # Determine the system response and state values of the reference system
    ref_time, ref_output, ref_state = scipy.signal.lsim2(
        ref_system,
        X0=sys.initial_condition,
        T=simulator.result.time,
        U=None,
        rtol=simulator.integrator_options["rtol"],
        atol=simulator.integrator_options["atol"])

    # Determine the state values at the simulated times
    # as per the reference value
    npt.assert_allclose(ref_time,
                        simulator.result.time,
                        rtol=simulator.integrator_options["rtol"],
                        atol=simulator.integrator_options["atol"])
    npt.assert_allclose(simulator.result.state,
                        ref_state.T,
                        rtol=simulator.integrator_options["rtol"] * 1E2,
                        atol=simulator.integrator_options["atol"] * 1E2)


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
    sys, ref_function, sim_time, outputs = lti_system_with_reference
    del ref_function  # unused

    simulator = Simulator(sys,
                          start_time=0,
                          integrator_constructor=MockupIntegrator,
                          integrator_options={})
    message = simulator.run_until(sim_time)

    # Integration must fail
    assert message is not None


def test_zero_crossing_event_detection():
    """Test the detection of zero-crossing events."""

    x0 = 0
    y0 = 10
    vx0 = 1
    vy0 = 0
    g = 9.81
    gamma = 0.7
    system = System()
    bouncing_ball = BouncingBall(parent=system,
                                 gravity=-g,
                                 gamma=gamma)

    initial_condition = np.empty(system.num_states)
    initial_condition[bouncing_ball.velocity.state_slice] = [vx0, vy0]
    initial_condition[bouncing_ball.position.state_slice] = [x0, y0]
    rootfinder_options = {
        "xtol": 1.E-9,
        "maxiter": 1000
    }

    simulator = Simulator(system,
                          start_time=0,
                          initial_condition=initial_condition,
                          rootfinder_options=rootfinder_options)
    message = simulator.run_until(time_boundary=8.0)

    # Check for successful run
    assert message is None

    # Determine the time of the first impact
    t_impact = (2 * vy0 + math.sqrt(4 * vy0 ** 2 + 8 * g * y0)) / (2 * g)

    # Check that the y-component of velocity changes sign around time of impact
    idx = bisect.bisect_left(simulator.result.time, t_impact)
    assert simulator.result.time[idx - 1] < t_impact
    assert simulator.result.time[idx] >= t_impact
    vy = bouncing_ball.velocity(simulator.result)[1]

    assert np.sign(vy[idx - 1]) != np.sign(vy[idx + 1])

    # Check detection of excessive event error
    with pytest.raises(ExcessiveEventError):
        simulator.run_until(time_boundary=10.0)


def test_excessive_events_second_level():
    """Test the detection of excessive events when it is introduced by
    toggling the same event over and over."""

    system = System()
    state = State(system,
                  derivative_function=(lambda data: -1),
                  initial_condition=5)
    event = ZeroCrossEventSource(system,
                                 event_function=state)

    def event_handler(data):
        """Event handler for the zero-crossing event"""
        state.set_value(data, -state(data))

    event.register_listener(event_handler)

    simulator = Simulator(system, start_time=0)

    with pytest.raises(ExcessiveEventError):
        simulator.run_until(6)


def test_clock_handling():
    """Test the handling of clocks in the simulator."""

    time_constant = 1.0
    initial_value = 2.0
    system = System()

    input_signal = constant(system, value=0.0)
    lag = LTISystem(parent=system,
                    system_matrix=-1 / time_constant,
                    input_matrix=1,
                    output_matrix=1,
                    feed_through_matrix=0,
                    initial_condition=[initial_value])
    lag.input.connect(input_signal)

    clock1 = Clock(system, period=0.2)
    hold1 = zero_order_hold(system,
                            input_port=lag.output,
                            event_port=clock1,
                            initial_condition=initial_value)

    clock2 = Clock(system, period=0.25, end_time=2.0)
    hold2 = zero_order_hold(system,
                            input_port=lag.output,
                            event_port=clock2,
                            initial_condition=initial_value)

    # Clock 3 will not fire within the simulation time frame
    clock3 = Clock(system, period=0.25, start_time=-6.0, end_time=-1.0)
    hold3 = zero_order_hold(system,
                            input_port=lag.output,
                            event_port=clock3,
                            initial_condition=initial_value)

    simulator = Simulator(system,
                          start_time=0.0)
    simulator.run_until(time_boundary=5.0)

    time_floor1 = np.floor(simulator.result.time / clock1.period) * clock1.period
    time_floor2 = np.minimum(clock2.end_time,
                             (np.floor(simulator.result.time / clock2.period) *
                              clock2.period))
    reference1 = initial_value * np.exp(-time_floor1 / time_constant)
    reference2 = initial_value * np.exp(-time_floor2 / time_constant)

    npt.assert_almost_equal(hold1(simulator.result)[0], reference1)
    npt.assert_almost_equal(hold2(simulator.result)[0], reference2)
    npt.assert_almost_equal(hold3(simulator.result)[0], initial_value)


def test_discrete_only():
    """Test a system with only discrete-time states."""

    system = System()
    clock = Clock(system,
                  period=1.0)
    counter = State(system)

    def increase_counter(data):
        """Increase the counter whenever a clock event occurs"""
        counter.set_value(data, counter(data) + 1)

    clock.register_listener(increase_counter)

    simulator = Simulator(system, start_time=0.0)
    assert simulator.run_until(time_boundary=10.0) is None

    npt.assert_almost_equal(simulator.result.time,
                            np.arange(start=0.0, stop=11.0))
    npt.assert_almost_equal(counter(simulator.result),
                            np.arange(start=1.0, stop=12.0).reshape(1, -1))


def test_integrator():
    """Test an integrator"""

    system = System()
    int_input = constant(system, 1.0)
    int_output = integrator(system, int_input)

    simulator = Simulator(system, start_time=0.0)
    assert simulator.run_until(time_boundary=10.0) is None

    npt.assert_almost_equal(
        int_output(simulator.result).flatten(),
        simulator.result.time)


def test_find_event_empty():
    """Test the _find_event_time function with an empty interval"""

    with pytest.raises(ValueError):
        _find_event_time(None, a=-1, b=-2, tolerance=1E-12)


def test_find_event_no_change():
    """Test the _find_event_time function with a function that does not
    change sign in the given interval"""

    with pytest.raises(ValueError):
        _find_event_time(np.sin, a=np.pi / 8, b=np.pi / 4, tolerance=1E-12)
