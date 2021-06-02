# pylint: disable=redefined-outer-name,missing-module-docstring
import math

import bisect
import numpy as np
import pytest
import scipy.signal
from fixtures.models import (
    BouncingBall,
    damped_oscillator,
    damped_oscillator_with_events,
    first_order_lag,
    first_order_lag_no_input,
)
from modypy.blocks.discrete import zero_order_hold
from modypy.blocks.linear import LTISystem, integrator
from modypy.blocks.sources import constant
from modypy.model import (
    Clock,
    State,
    System,
    SystemState,
    ZeroCrossEventSource,
)
from modypy.simulation import (
    ExcessiveEventError,
    SimulationError,
    SimulationResult,
    Simulator,
)
from numpy import testing as npt


@pytest.fixture(
    params=[
        first_order_lag(time_constant=1, initial_value=10),
        first_order_lag_no_input(time_constant=1, initial_value=10),
        damped_oscillator(
            mass=100, spring_coefficient=1.0, damping_coefficient=20
        ),  # critically damped
        damped_oscillator(
            mass=100, spring_coefficient=0.5, damping_coefficient=20
        ),  # overdamped
        damped_oscillator(
            mass=100, spring_coefficient=2.0, damping_coefficient=20
        ),  # underdamped
        damped_oscillator_with_events(
            mass=100, spring_coefficient=2.0, damping_coefficient=20
        ),  # underdamped
    ]
)
def lti_system_with_reference(request):
    system, lti_system, ref_time, outputs = request.param
    assert isinstance(lti_system, LTISystem)

    ref_system = scipy.signal.StateSpace(
        lti_system.system_matrix,
        lti_system.input_matrix,
        lti_system.output_matrix,
        lti_system.feed_through_matrix,
    )
    return system, ref_system, ref_time, outputs


def test_lti_simulation(lti_system_with_reference):
    sys, ref_system, ref_time, _ = lti_system_with_reference

    rtol = 1e-12
    atol = 1e-12
    simulator = Simulator(sys, start_time=0, rtol=rtol, atol=atol)
    result = SimulationResult(sys)
    result.collect_from(
        simulator.run_until(time_boundary=ref_time / 2, include_last=False)
    )
    result.collect_from(
        simulator.run_until(time_boundary=ref_time, include_last=False)
    )

    # Check that states are properly mapped in the result
    for state in sys.states:
        from_result = result.state[state.state_slice]
        from_result = from_result.reshape(state.shape + (-1,))
        npt.assert_equal(state(result), from_result)

    # Check that inputs are properly mapped in the result
    for input_signal in sys.inputs:
        from_result = result.inputs[input_signal.input_slice]
        from_result = from_result.reshape(input_signal.shape + (-1,))
        npt.assert_equal(input_signal(result), from_result)

    # Determine the system response and state values of the reference system
    ref_time, ref_output, ref_state = scipy.signal.lsim2(
        ref_system,
        X0=sys.initial_condition,
        T=result.time,
        U=None,
        rtol=rtol,
        atol=atol,
    )
    del ref_output

    # Determine the state values at the simulated times
    # as per the reference value
    npt.assert_allclose(ref_time, result.time, rtol=rtol, atol=atol)
    # FIXME: These factors are somewhat arbitrary
    npt.assert_allclose(
        result.state, ref_state.T, rtol=rtol * 1e2, atol=atol * 1e2
    )

    # Check that SimulationResult properly implements the sequence interface
    assert len(result) == len(result.time)
    for idx, item in enumerate(result):
        npt.assert_equal(item.time, result.time[idx])
        npt.assert_equal(item.state, result.state[:, idx])
        npt.assert_equal(item.inputs, result.inputs[:, idx])


class MockupIntegrator:
    """
    Mockup integrator class to force integration error.
    """

    def __init__(self, fun, t0, y0, t_bound, vectorized=False):
        del fun  # unused
        del t_bound  # unused
        del vectorized  # unused
        self.status = "running"
        self.t = t0
        self.y = y0

    def step(self):
        self.status = "failed"
        return "failed"


def test_lti_simulation_failure(lti_system_with_reference):
    sys, ref_function, sim_time, outputs = lti_system_with_reference
    del ref_function, outputs  # unused

    simulator = Simulator(sys, start_time=0, solver_method=MockupIntegrator)
    with pytest.raises(SimulationError):
        for _ in simulator.run_until(sim_time):
            pass


def test_zero_crossing_event_detection():
    """Test the detection of zero-crossing events."""

    x0 = 0
    y0 = 10
    vx0 = 1
    vy0 = 0
    g = 9.81
    gamma = 0.7
    system = System()
    bouncing_ball = BouncingBall(parent=system, gravity=-g, gamma=gamma)

    initial_condition = np.empty(system.num_states)
    initial_condition[bouncing_ball.velocity.state_slice] = [vx0, vy0]
    initial_condition[bouncing_ball.position.state_slice] = [x0, y0]

    simulator = Simulator(
        system,
        start_time=0,
        initial_condition=initial_condition,
        max_step=0.05,
        event_xtol=1.0e-9,
        event_maxiter=1000,
    )
    result = SimulationResult(system, simulator.run_until(time_boundary=8.0))

    # Determine the time of the first impact
    t_impact = (2 * vy0 + math.sqrt(4 * vy0 ** 2 + 8 * g * y0)) / (2 * g)

    # Check that the y-component of velocity changes sign around time of impact
    t_tolerance = 0.1
    idx_start = bisect.bisect_left(result.time, t_impact - t_tolerance)
    idx_end = bisect.bisect_left(result.time, t_impact + t_tolerance)
    assert result.time[idx_start] < t_impact
    assert result.time[idx_end] >= t_impact
    vy = bouncing_ball.velocity(result)[1][idx_start:idx_end]
    sign_change = np.sign(vy[:-1]) != np.sign(vy[1:])
    assert any(sign_change)

    # Check detection of excessive event error
    with pytest.raises(ExcessiveEventError):
        for _ in simulator.run_until(time_boundary=10.0):
            pass


def test_excessive_events_second_level():
    """Test the detection of excessive events when it is introduced by
    toggling the same event over and over."""

    system = System()
    state = State(
        system, derivative_function=(lambda data: -1), initial_condition=5
    )
    event = ZeroCrossEventSource(system, event_function=state)

    def event_handler(data):
        """Event handler for the zero-crossing event"""
        state.set_value(data, -state(data))

    event.register_listener(event_handler)

    simulator = Simulator(system, start_time=0)

    with pytest.raises(ExcessiveEventError):
        for _ in simulator.run_until(6):
            pass


def test_clock_handling():
    """Test the handling of clocks in the simulator."""

    time_constant = 1.0
    initial_value = 2.0
    system = System()

    input_signal = constant(value=0.0)
    lag = LTISystem(
        parent=system,
        system_matrix=-1 / time_constant,
        input_matrix=1,
        output_matrix=1,
        feed_through_matrix=0,
        initial_condition=[initial_value],
    )
    lag.input.connect(input_signal)

    clock1 = Clock(system, period=0.2)
    hold1 = zero_order_hold(
        system,
        input_port=lag.output,
        event_port=clock1,
        initial_condition=initial_value,
    )

    clock2 = Clock(system, period=0.25, end_time=2.0)
    hold2 = zero_order_hold(
        system,
        input_port=lag.output,
        event_port=clock2,
        initial_condition=initial_value,
    )

    # Clock 3 will not fire within the simulation time frame
    clock3 = Clock(system, period=0.25, start_time=-6.0, end_time=-1.0)
    hold3 = zero_order_hold(
        system,
        input_port=lag.output,
        event_port=clock3,
        initial_condition=initial_value,
    )

    simulator = Simulator(system, start_time=0.0)
    result = SimulationResult(system, simulator.run_until(time_boundary=5.0))

    time_floor1 = np.floor(result.time / clock1.period) * clock1.period
    time_floor2 = np.minimum(
        clock2.end_time, (np.floor(result.time / clock2.period) * clock2.period)
    )
    reference1 = initial_value * np.exp(-time_floor1 / time_constant)
    reference2 = initial_value * np.exp(-time_floor2 / time_constant)

    npt.assert_almost_equal(hold1(result), reference1)
    npt.assert_almost_equal(hold2(result), reference2)
    npt.assert_almost_equal(hold3(result), initial_value)


def test_discrete_only():
    """Test a system with only discrete-time states."""

    system = System()
    clock = Clock(system, period=1.0)
    counter = State(system)

    def increase_counter(data):
        """Increase the counter whenever a clock event occurs"""
        counter.set_value(data, counter(data) + 1)

    clock.register_listener(increase_counter)

    simulator = Simulator(system, start_time=0.0)
    result = SimulationResult(system)
    result.collect_from(simulator.run_until(5.5, include_last=False))
    result.collect_from(simulator.run_until(10))

    npt.assert_almost_equal(
        result.time, [0, 1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10]
    )
    npt.assert_almost_equal(
        counter(result), [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11]
    )


def test_integrator():
    """Test an integrator"""

    system = System()
    int_input = constant(1.0)
    int_output = integrator(system, int_input)

    simulator = Simulator(system, start_time=0.0)
    result = SimulationResult(system, simulator.run_until(time_boundary=10.0))

    npt.assert_almost_equal(int_output(result).ravel(), result.time)


def test_simulation_result_dictionary_access():
    """Test the deprecated dictionary access for simulation results"""

    system = System()
    state_a = State(system, initial_condition=10)
    state_b = State(system, shape=2, initial_condition=[11, 12])
    state_c = State(
        system, shape=(2, 2), initial_condition=[[13, 14], [15, 16]]
    )

    system_state = SystemState(time=0, system=system)

    result = SimulationResult(system)
    result.append(system_state)

    npt.assert_equal(result[state_a], state_a(result))
    npt.assert_equal(result[state_b], state_b(result))
    npt.assert_equal(result[state_b, 0], state_b(result)[0])
    npt.assert_equal(result[state_b, 1], state_b(result)[1])
    npt.assert_equal(result[state_c, 0, 0], state_c(result)[0, 0])


def test_system_state_updater_dictionary_access():
    """Test the deprecated dictionary access for simulation results"""

    system = System()
    counter = State(system, shape=(2, 2))

    clock = Clock(system, 1.0)

    def _update_counter(system_state):
        old_val = counter(system_state).copy()
        system_state[counter] += 1
        system_state[counter, 0] += 1
        system_state[counter, 0, 0] += 1
        new_val = counter(system_state)
        npt.assert_almost_equal(old_val + [[3, 2], [1, 1]], new_val)

    clock.register_listener(_update_counter)

    simulator = Simulator(system, start_time=0)
    for _ in simulator.run_until(time_boundary=10):
        pass
