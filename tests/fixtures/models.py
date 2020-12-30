import math

import numpy as np

from simtree.blocks import LeafBlock, NonLeafBlock
from simtree.blocks.linear import LTISystem, Gain
from simtree.blocks.sources import Constant, SourceFromCallable


def first_order_lag(T=1, x0=10):
    sys = LTISystem(A=-1 / T,
                    B=1,
                    C=1,
                    D=0,
                    initial_condition=[x0])

    return sys, 3 * T


def first_order_lag_no_input(T=1, x0=10):
    sys = LTISystem(A=-1 / T,
                    B=[],
                    C=1,
                    D=[],
                    initial_condition=[x0])

    return sys, 3 * T


def damped_oscillator(m=100., d=50., k=1., x0=10.):
    osc = LTISystem(A=[[0, 1], [-k / m, -d / m]],
                    B=[[-1 / m], [0]],
                    C=[[1, 0]],
                    D=np.zeros((1, 1)),
                    initial_condition=[x0, 0],
                    feedthrough_inputs=[])
    T = 2 * m / d
    return osc, 3 * T


def damped_oscillator_with_events(*args, **kwargs):
    osc, sim_time = damped_oscillator(*args, **kwargs)
    osc.num_events = 1
    osc.event_function = (lambda t, states, inputs: [states[1]])
    osc.update_state_function = (lambda t, states, inputs: states)

    return osc, sim_time


def oscillator_with_sine_input(omega, *args, **kwargs):
    osc, sim_time = damped_oscillator(*args, **kwargs)
    osc.num_events = 1
    osc.event_function = (lambda t, states, inputs: [states[0]])
    osc.update_state_function = (lambda t, states, inputs: states)
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1)
    sys = NonLeafBlock(children=[osc, sin_input])
    sys.connect(sin_input, 0, osc, 0)

    return sys, sim_time


def sine_input_with_gain(omega, *args, **kwargs):
    gain = Gain([[1.0]], num_events=1)
    gain.event_function = (lambda t, inputs: [inputs[0]])
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1)
    sys = NonLeafBlock(children=[gain, sin_input])
    sys.connect(sin_input, 0, gain, 0)

    return sys, 10.0


def lti_gain(gain):
    sys = LTISystem(A=np.empty((0, 0)),
                    B=np.empty((0, 1)),
                    C=np.empty((1, 0)),
                    D=gain)
    return sys, 10.0


def sine_source(omega):
    sin_scaled = (lambda t: np.sin(omega * t))
    sin_input = SourceFromCallable(fun=sin_scaled, num_outputs=1, num_events=1)
    sin_input.event_function = (lambda t: [sin_scaled(t)])

    return sin_input, 3*2*math.pi/omega


class StaticPropeller(LeafBlock):
    def __init__(self, ct, cp, D, **kwargs):
        LeafBlock.__init__(self, num_inputs=2, num_outputs=3, **kwargs)
        self.cp = cp
        self.ct = ct
        self.D = D

    def output_function(self, t, inputs):
        return [self.ct * inputs[1] * self.D ** 4 * inputs[0] ** 2,
                self.cp * inputs[1] * self.D ** 5 * inputs[0] ** 3,
                self.cp / (2 * math.pi) * inputs[1] * self.D ** 5 * inputs[0] ** 2]


class DCMotor(LTISystem):
    def __init__(self, Kv, R, L, J, **kwargs):
        LTISystem.__init__(self,
                           A=[[0, Kv / J], [-Kv / L, -R / L]],
                           B=[[0, -1 / J], [1 / L, 0]],
                           C=[[1 / (2 * math.pi), 0], [0, 1]],
                           D=[[0, 0], [0, 0]],
                           feedthrough_inputs=[],
                           **kwargs)


def propeller_model(ct=0.09,
                    cp=0.04,
                    D=8 * 25.4E-3):
    return StaticPropeller(ct, cp, D, name="static_propeller")


def engine_model(Kv=789.E-6,
                 R=43.3E-3,
                 L=1.9E-3,
                 J=5.284E-6,
                 *args,
                 **kwargs):
    static_propeller = propeller_model(*args, **kwargs)

    dcmotor = DCMotor(Kv, R, L, J, name="dcmotor")

    engine = NonLeafBlock(name="engine",
                          children=[dcmotor, static_propeller],
                          num_inputs=2,
                          num_outputs=2)

    engine.connect(dcmotor, 0, static_propeller, 0)
    engine.connect(static_propeller, 1, dcmotor, 1)

    engine.connect_input(0, dcmotor, 0)
    engine.connect_input(1, static_propeller, 1)

    engine.connect_output(static_propeller, 0, 0)
    engine.connect_output(dcmotor, 1, 1)

    return engine


def dcmotor_model(voltage=0.4 * 11.1,
                  rho=1.29,
                  *args,
                  **kwargs):
    engine = engine_model(*args, **kwargs)

    voltage = Constant(value=voltage, name="voltage")
    density = Constant(value=rho, name="rho")

    system = NonLeafBlock(name="system",
                          children=[voltage, density, engine])
    system.connect(voltage, 0, engine, 0)
    system.connect(density, 0, engine, 1)

    return system


def dcmotor_model_cyclic(*args, **kwargs):
    system = dcmotor_model(*args, **kwargs)
    # FIXME: This is quite hacky to access the subsystem like that
    engine = system.children[2]
    dcmotor = engine.children[0]

    # Make all inputs feed-through, which leads to a dependency cycle
    dcmotor.feedthrough_inputs = [0, 1]

    return system


class BouncingBall(LeafBlock):
    def __init__(self, g=-9.81, gamma=0.3, **kwargs):
        LeafBlock.__init__(self, num_states=4, num_events=1, num_outputs=1, **kwargs)
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


def bouncing_ball_model(*args, **kwargs):
    return BouncingBall(*args, **kwargs)
