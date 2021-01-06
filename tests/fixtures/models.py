import math

import numpy as np

from modypy.blocks.linear import LTISystem, Gain
from modypy.model import \
    Block,\
    System, \
    Event, \
    InputSignal, \
    Signal, \
    State, \
    OutputPort


def first_order_lag(time_constant=1, initial_value=10):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=-1 / time_constant,
                    input_matrix=1,
                    output_matrix=1,
                    feed_through_matrix=0,
                    initial_condition=[initial_value])

    src = InputSignal(system)
    dest = OutputPort(system)

    lti.input.connect(src)
    dest.connect(lti.output)

    return system, lti, 3 * time_constant


def first_order_lag_no_input(time_constant=1, initial_value=10):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=-1 / time_constant,
                    input_matrix=[],
                    output_matrix=1,
                    feed_through_matrix=[],
                    initial_condition=[initial_value])
    dest = OutputPort(system)
    dest.connect(lti.output)

    return system, lti, 3 * time_constant


def damped_oscillator(mass=100.,
                      damping_coefficient=50.,
                      spring_coefficient=1.,
                      initial_value=10.):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=[[0, 1],
                                   [-spring_coefficient / mass,
                                    -damping_coefficient / mass]],
                    input_matrix=[[-1 / mass], [0]],
                    output_matrix=[[1, 0]],
                    feed_through_matrix=np.zeros((1, 1)),
                    initial_condition=[initial_value, 0])
    time_constant = 2 * mass / damping_coefficient

    src = InputSignal(system)
    dest = OutputPort(system)
    lti.input.connect(src)
    dest.connect(lti.output)

    return system, lti, 3 * time_constant


def damped_oscillator_with_events(mass=100.,
                                  damping_coefficient=50.,
                                  spring_coefficient=1.,
                                  initial_value=10.):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=[[0, 1],
                                   [-spring_coefficient / mass,
                                    -damping_coefficient / mass]],
                    input_matrix=[[-1 / mass], [0]],
                    output_matrix=[[1, 0]],
                    feed_through_matrix=np.zeros((1, 1)),
                    initial_condition=[initial_value, 0])
    Event(owner=system,
          event_function=(lambda data: data.states[lti.state][1]))
    time_constant = 2 * mass / damping_coefficient

    src = InputSignal(system)
    dest = OutputPort(system)
    lti.input.connect(src)
    dest.connect(lti.output)

    return system, lti, 3 * time_constant


def oscillator_with_sine_input(omega,
                               mass=100.,
                               damping_coefficient=50.,
                               spring_coefficient=1.,
                               initial_value=10.):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=[[0, 1],
                                   [-spring_coefficient / mass,
                                    -damping_coefficient / mass]],
                    input_matrix=[[-1 / mass], [0]],
                    output_matrix=[[1, 0]],
                    feed_through_matrix=np.zeros((1, 1)),
                    initial_condition=[initial_value, 0])
    Event(owner=system,
          event_function=(lambda data: data.states[lti.state][1]))
    sine_input = Signal(owner=system,
                        value=(lambda data: np.sin(omega * data.time)))
    sine_input.connect(lti.input)

    time_constant = 2 * mass / damping_coefficient
    return system, lti, 3 * time_constant


def sine_input_with_gain(omega):
    system = System()
    gain = Gain(parent=system, k=1)
    gain_event = Event(owner=system,
                       event_function=(lambda data: data.inputs[gain.input]))
    sine_input = Signal(owner=system,
                        value=(lambda data: np.sin(omega*data.time)))
    sine_input.connect(gain.input)

    return system, 10.0


def lti_gain(gain):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 1)),
                    output_matrix=np.empty((1, 0)),
                    feed_through_matrix=gain)

    src = InputSignal(system)
    dest = OutputPort(system)

    lti.input.connect(src)
    dest.connect(lti.output)

    return system, lti, 10.0


def sine_source(omega):
    system = System()
    sine_input = Signal(owner=system,
                        value=(lambda data: np.sin(omega*data.time)))
    event = Event(owner=system,
                  event_function=(lambda data: data.inputs[sine_input]))

    return system, 3*2*math.pi/omega




class BouncingBall(Block):
    def __init__(self, parent, gravity=-9.81, gamma=0.7, initial_velocity=None, initial_position=None):
        Block.__init__(self, parent)
        self.position = State(self, shape=2, derivative_function=self.position_derivative, initial_condition=initial_position)
        self.velocity = State(self, shape=2, derivative_function=self.velocity_derivative, initial_condition=initial_velocity)
        self.posy = Signal(self, shape=1, value=self.posy_output)
        self.ground = Event(self, event_function=self.ground_event, update_function=self.on_ground_event)
        self.gravity = gravity
        self.gamma = gamma

    def position_derivative(self, data):
        return data.states[self.velocity]

    def velocity_derivative(self, data):
        return np.r_[0, self.gravity]

    def posy_output(self, data):
        return data.states[self.position][1]

    def ground_event(self, data):
        return data.states[self.position][1]

    def on_ground_event(self, data):
        data.states[self.position] = [data.states[self.position][0], abs(data.states[self.position][1])]
        data.states[self.velocity][1] = - self.gamma * data.states[self.velocity][1]


def bouncing_ball_model(g=-9.81,
                        gamma=0.3,
                        initial_velocity=None,
                        initial_position=None):
    system = System()

    BouncingBall(system,
                 gravity=g,
                 gamma=gamma,
                 initial_position=initial_position,
                 initial_velocity=initial_velocity)

    return system
