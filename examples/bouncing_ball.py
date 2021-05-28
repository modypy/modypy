"""
An example modelling a bouncing ball, demonstrating the use of zero-crossing
events and event-handler functions.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, Block, State, ZeroCrossEventSource, signal_method
from modypy.simulation import Simulator, SimulationResult


class BouncingBall(Block):
    """
    Block for modelling a bouncing ball.

    This block has states ``position`` and ``velocity`` with the obvious
    meaning.

    The coefficient of restitution ``gamma`` gives the ratio of the final to the
    initial velocity on impact.

    The block has a single event which represents the impact on ground by
    monitoring the y-coordinate of the position for sign changes. Upon impact,
    the y-coordinate is set to be positive and the sign of y-velocity is
    reversed.
    """
    def __init__(self,
                 parent,
                 gravity=-9.81,
                 gamma=0.7,
                 initial_velocity=None,
                 initial_position=None):
        Block.__init__(self, parent)
        self.velocity = State(self,
                              shape=2,
                              derivative_function=self.velocity_derivative,
                              initial_condition=initial_velocity)
        self.position = State(self,
                              shape=2,
                              derivative_function=self.velocity,
                              initial_condition=initial_position)
        self.ground = ZeroCrossEventSource(self,
                                           event_function=self.ground_event,
                                           direction=-1,
                                           tolerance=1E-3)
        self.top = ZeroCrossEventSource(self,
                                        event_function=self.top_event,
                                        direction=-1,
                                        tolerance=1E-3)
        self.ground.register_listener(self.on_ground_event)
        self.gravity = gravity
        self.gamma = gamma

    def velocity_derivative(self, _system_state):
        """The time-derivative of the position (i.e. the acceleration)"""
        return np.r_[0, self.gravity]

    @signal_method
    def posy(self, system_state):
        """The output for the y-position"""
        return self.position(system_state)[1]

    def ground_event(self, system_state):
        """The event function for impact-detection"""
        return self.position(system_state)[1]

    def top_event(self, system_state):
        """The event function for top-point-detection"""
        return self.velocity(system_state)[1]

    def on_ground_event(self, system_state):
        """The handler for the ground event"""
        self.position(system_state)[1] = abs(self.position(system_state)[1])
        self.velocity(system_state)[1] = \
            - self.gamma * self.velocity(system_state)[1]


# Create a system containing the bouncing ball model
system = System()
block = BouncingBall(system, initial_velocity=[1, 0], initial_position=[0, 10])

# Run a simulation for 10s
simulator = Simulator(system, start_time=0.0, max_step=0.1)
result = SimulationResult(system, simulator.run_until(10.0))

# Plot the x- and y-position of the ball against each other
position = block.position(result)
plt.plot(position[0], position[1])
plt.show()
