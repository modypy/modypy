import pytest
import numpy as np
import numpy.testing as npt
from simtree.blocks import LeafBlock, NonLeafBlock
from simtree.blocks.linear import LTISystem, Gain
from simtree.blocks.sources import SourceFromCallable
from simtree.compiler import Compiler
from simtree.simulator import Simulator


def get_first_order_lag(T=1, x0=10):
    sys = LTISystem(A=[[-1 / T]],
                    B=[[]],
                    C=[[1]],
                    D=[[]],
                    initial_condition=[x0])

    def ref_function(t):
        return x0 * np.exp(-t / T).reshape(-1, 1)

    return sys, ref_function, 3 * T


def get_damped_oscillator(m=100., d=50., k=1., x0=10.):
    sys = LTISystem(A=[[0, 1], [-k / m, -d / m]],
                    B=np.empty((2, 0)),
                    C=[[1, 0]],
                    D=np.empty((1, 0)),
                    initial_condition=[x0, 0])
    T = 2 * m / d
    if d ** 2 - 4 * m * k < 0:
        # underdamped case
        omega = np.sqrt(4 * m * k - d ** 2) / (2 * m)
        phi = np.arctan2(-1.0, T * omega)
        c = x0 / np.cos(phi)

        def ref_function(t):
            return (c * np.exp(-t / T) * np.cos(omega * t + phi)).reshape(-1, 1)
    elif d ** 2 - 4 * m * k > 0:
        # overdamped case
        omega = np.sqrt(d ** 2 - 4 * m * k) / (2 * m)
        c1 = (T * omega + 1) * x0 / (2 * T * omega)
        c2 = (T * omega - 1) * x0 / (2 * T * omega)

        def ref_function(t):
            return (np.exp(-t / T) *
                    (c1 * np.exp(omega * t) + c2 * np.exp(-omega * t))
                    ).reshape(-1, 1)
    else:
        # critically damped case
        c1 = x0
        c2 = d * x0 / (2 * m)

        def ref_function(t):
            return (np.exp(-t / T) * (c1 + c2 * t)).reshape(-1, 1)
    return sys, ref_function, 3 * T


def get_controlled_integrator(T=1, gamma=0.5, x0=10):
    integ = LTISystem(A=[[1 / T]],
                      B=[[1]],
                      C=[[1]],
                      D=np.zeros((1, 1)),
                      name="integrator",
                      initial_condition=[x0],
                      feedthrough_inputs=[])
    # Determine control feedback for position
    ctrl = Gain(k=[[-gamma]], name="gain")
    sys = NonLeafBlock(children=[integ, ctrl],num_outputs=1)
    sys.connect(integ, 0, ctrl, 0)
    sys.connect(ctrl, 0, integ, 0)
    sys.connect_output(integ,0,0)

    def ref_function(t):
        return (x0 * np.exp((1 / T - gamma) * t)).reshape(-1,1)

    return sys, ref_function, 3 * T


@pytest.fixture(params=[
    get_first_order_lag(),

    get_damped_oscillator(m=100, k=1., d=20),  # critically damped
    get_damped_oscillator(m=100, k=0.5, d=20),  # overdamped
    get_damped_oscillator(m=100, k=2., d=20),  # underdamped

    get_controlled_integrator(T=2, gamma=0.5, x0=10),  # astable
    get_controlled_integrator(T=2, gamma=0.8, x0=10),  # asymptotically stable
    get_controlled_integrator(T=2, gamma=0.2, x0=10),  # unstable
])
def system_reference(request):
    sys, ref_function, Tsim = request.param

    return sys, ref_function, Tsim


def test_simulation(system_reference):
    sys, ref_function, Tsim = system_reference

    # Compile and run the system
    compiler = Compiler(sys)
    sys_compiled = compiler.compile()

    simulator = Simulator(sys_compiled, t0=0, tbound=Tsim,
                          initial_condition=sys_compiled.initial_condition)
    message = simulator.run()

    # Simulation must be successful
    assert message is None
    assert simulator.status == "finished"

    # Determine the output values at the simulated times as per the reference function
    ref_output = ref_function(simulator.result.t)

    npt.assert_allclose(simulator.result.output, ref_output)


class MockupIntegrator:
    """
    Mockup integrator class to force integration error.
    """

    def __init__(self, fun, t0, y, tbound):
        self.status = "running"
        self.t = t0
        self.y = y

    def step(self):
        self.status = "failed"
        return "failed"


def test_simulation_failure(system_reference):
    sys, ref_function, Tsim = system_reference

    # Compile and run the system
    compiler = Compiler(sys)
    sys_compiled = compiler.compile()

    simulator = Simulator(sys_compiled, t0=0, tbound=Tsim, integrator_constructor=MockupIntegrator,
                          integrator_options={})
    message = simulator.run()

    # Integration must fail
    assert message == "failed"


class BouncingBall(LeafBlock):
    def __init__(self, g=-9.81, gamma=0.3, **kwargs):
        LeafBlock.__init__(self, num_states=4, num_events=1,
                           num_outputs=2, **kwargs)
        self.g = g
        self.gamma = gamma

    def output_function(self, t, state):
        return [state[1]]

    def state_update_function(self, t, state):
        return [state[2], state[3],
                -self.gamma * state[2], self.g - self.gamma * state[3]]

    def event_function(self, t, state):
        return [state[1]]

    def update_state_function(self, t, state):
        return [state[0], abs(state[1]), state[2], -state[3]]


def get_oscillator_with_sine_input(m, d, k, x0, omega):
    osc = LTISystem(A=[[0, 1], [-k / m, -d / m]],
                    B=[[-1 / m], [0]],
                    C=[[1, 0]],
                    D=np.zeros((1, 1)),
                    num_events=1,
                    initial_condition=[x0, 0],
                    feedthrough_inputs=[])
    osc.event_function = (lambda t, states, inputs: [states[0]])
    osc.update_state_function = (lambda t, states, inputs: states)
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1)
    sys = NonLeafBlock(children=[osc, sin_input])
    sys.connect(sin_input, 0, osc, 0)

    return sys


def get_oscillator_with_sine_input_and_gain(m, d, k, x0, omega):
    osc = LTISystem(A=[[0, 1], [-k / m, -d / m]],
                    B=[[-1 / m], [0]],
                    C=[[1, 0]],
                    D=np.zeros((1, 1)),
                    initial_condition=[x0, 0],
                    feedthrough_inputs=[])
    gain = Gain([[1.0]], num_events=1)
    gain.event_function = (lambda t, inputs: [inputs[0]])
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1)
    sys = NonLeafBlock(children=[gain, osc, sin_input])
    sys.connect(osc, 0, gain, 0)
    sys.connect(sin_input, 0, osc, 0)

    return sys


def get_sine_source(omega):
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1, num_events=1)
    sin_input.event_function = (lambda t: [sin_scaled(t)])

    sys = NonLeafBlock(children=[sin_input])

    return sys


@pytest.fixture(
    params=[
        BouncingBall(initial_condition=[0, 10, 1, 0]),
        get_oscillator_with_sine_input(m=1, k=40, d=20, x0=10, omega=20),
        get_oscillator_with_sine_input_and_gain(
            m=100, k=0.5, d=20, x0=10, omega=10),
        get_sine_source(omega=10)
    ])
def event_systems(request):
    return request.param


def test_events(event_systems):
    sys = event_systems

    # Compile and run the system
    compiler = Compiler(sys)
    sys_compiled = compiler.compile()

    simulator = Simulator(sys_compiled, t0=0, tbound=10.0)
    message = simulator.run()

    # Check for successful run
    assert message is None
    assert simulator.status == "finished"

    # Check that event value is close to zero at the event
    event_idx = np.flatnonzero(simulator.result.events[:, 0])

    evfun_at_event = np.empty(event_idx.size)

    for idx in range(event_idx.size):
        evidx = event_idx[idx]
        evfun_at_event[idx] = sys_compiled.event_function(simulator.result.t[evidx],
                                                          simulator.result.state[evidx, :],
                                                          simulator.result.output[evidx, :])[0]

    npt.assert_allclose(evfun_at_event, np.zeros_like(
        evfun_at_event), atol=1E-7)
