import numpy as np

from modypy.blocks.linear import LTISystem
from modypy.model import \
    Block, \
    System, \
    ZeroCrossEventSource, \
    InputSignal, \
    State, signal_method


def first_order_lag(time_constant=1, initial_value=10):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=-1 / time_constant,
                    input_matrix=1,
                    output_matrix=1,
                    feed_through_matrix=0,
                    initial_condition=[initial_value])

    src = InputSignal(system)

    lti.input.connect(src)

    return system, lti, 3 * time_constant, [lti.output]


def first_order_lag_no_input(time_constant=1, initial_value=10):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=-1 / time_constant,
                    input_matrix=[],
                    output_matrix=1,
                    feed_through_matrix=[],
                    initial_condition=[initial_value])

    return system, lti, 3 * time_constant, [lti.output]


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
    lti.input.connect(src)

    return system, lti, 3 * time_constant, [lti.output]


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
    ZeroCrossEventSource(owner=system,
                         event_function=(lambda data: lti.state(data)[1]))
    time_constant = 2 * mass / damping_coefficient

    src = InputSignal(system)
    lti.input.connect(src)

    return system, lti, 3 * time_constant, [lti.output]


def lti_gain(gain):
    system = System()
    lti = LTISystem(parent=system,
                    system_matrix=np.empty((0, 0)),
                    input_matrix=np.empty((0, 1)),
                    output_matrix=np.empty((1, 0)),
                    feed_through_matrix=gain)

    src = InputSignal(system)

    lti.input.connect(src)

    return system, lti, 10.0, [lti.output]


class BouncingBall(Block):
    def __init__(self,
                 parent,
                 gravity=-9.81,
                 gamma=0.7,
                 initial_velocity=None,
                 initial_position=None):
        Block.__init__(self, parent)
        self.position = State(self,
                              shape=2,
                              derivative_function=self.position_derivative,
                              initial_condition=initial_position)
        self.velocity = State(self,
                              shape=2,
                              derivative_function=self.velocity_derivative,
                              initial_condition=initial_velocity)
        self.ground = ZeroCrossEventSource(self,
                                           event_function=self.ground_event,
                                           direction=-1)
        self.ground.register_listener(self.on_ground_event)
        self.gravity = gravity
        self.gamma = gamma

    def position_derivative(self, system_state):
        return self.velocity(system_state)

    def velocity_derivative(self, _system_state):
        return np.r_[0, self.gravity]

    @signal_method
    def posy(self, data):
        return self.position(data)[1]

    def ground_event(self, data):
        return self.position(data)[1]

    def on_ground_event(self, data):
        self.position(data)[1] = abs(self.position(data)[1])
        self.velocity(data)[1] = -self.gamma * self.velocity(data)[1]
