from simtree.model import System, Block, State, Signal, Event
from simtree.simulation import Simulator, DEFAULT_INTEGRATOR_OPTIONS
import numpy as np
import matplotlib.pyplot as plt


class BouncingBall(Block):
    def __init__(self, parent, gravity=-9.81, gamma=0.3, initial_velocity=None, initial_position=None):
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
        velocity = data.states[self.velocity]
        return np.r_[-self.gamma * velocity[0], self.gravity - self.gamma * velocity[1]]

    def posy_output(self, data):
        return data.states[self.position][1]

    def ground_event(self, data):
        return data.states[self.position][1]

    def on_ground_event(self, data):
        data.states[self.position][1] = abs(data.states[self.position][1])
        data.states[self.velocity][1] = - data.states[self.velocity][1]


DEFAULT_INTEGRATOR_OPTIONS['max_step'] = 0.05

system = System()
block = BouncingBall(system, initial_velocity=[1,0], initial_position=[0, 10])

simulator = Simulator(system, start_time=0.0)
simulator.run_until(10.0)

position = simulator.result.state[:, block.position.state_slice]
plt.plot(position[:, 0], position[:, 1])
plt.show()
