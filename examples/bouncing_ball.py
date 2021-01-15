"""
An example modelling a bouncing ball, demonstrating the use of zero-crossing
events and event-handler functions.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, Block, State, Signal, ZeroCrossEventSource
from modypy.simulation import Simulator


class BouncingBall(Block):
    """
    Block for modelling a bouncing ball.

    This block has states ``position`` and ``velocity`` with the obvious meaning.

    The coefficient of restitution ``gamma`` gives the ratio of the final to the
    initial velocity on impact.

    The block has a single event which represents the impact on ground by
    monitoring the y-coordinate of the position for sign changes. Upon impact,
    the y-coordinate is set to be positive and the sign of y-velocity is reversed.
    """
    def __init__(self, parent, gravity=-9.81, gamma=0.7, initial_velocity=None, initial_position=None):
        Block.__init__(self, parent)
        self.position = State(self,
                              shape=2,
                              derivative_function=self.position_derivative,
                              initial_condition=initial_position)
        self.velocity = State(self,
                              shape=2,
                              derivative_function=self.velocity_derivative,
                              initial_condition=initial_velocity)
        self.posy = Signal(self, shape=1, value=self.posy_output)
        self.ground = ZeroCrossEventSource(self, event_function=self.ground_event)
        self.ground.register_listener(self.on_ground_event)
        self.gravity = gravity
        self.gamma = gamma

    def position_derivative(self, data):
        """The time-derivative of the position (i.e. the velocity)"""
        return data.states[self.velocity]

    def velocity_derivative(self, data):
        """The time-derivative of the position (i.e. the acceleration)"""
        return np.r_[0, self.gravity]

    def posy_output(self, data):
        """The output for the y-position"""
        return data.states[self.position][1]

    def ground_event(self, data):
        """The event function for impact-detection"""
        return data.states[self.position][1]

    def on_ground_event(self, data):
        """The handler for the ground event"""
        data.states[self.position][1] = abs(data.states[self.position][1])
        data.states[self.velocity][1] = - self.gamma * data.states[self.velocity][1]


# Create a system containing the bouncing ball model
system = System()
block = BouncingBall(system, initial_velocity=[1, 0], initial_position=[0, 10])

# Run a simulation for 10s
simulator = Simulator(system, start_time=0.0)
simulator.run_until(10.0)

# Plot the x- and y-position of the ball against each other
position = simulator.result.state[:, block.position.state_slice]
plt.plot(position[:, 0], position[:, 1])
plt.show()
