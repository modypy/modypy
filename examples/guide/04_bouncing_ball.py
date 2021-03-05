"""
A bouncing ball
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, State, ZeroCrossEventSource
from modypy.simulation import Simulator

# The system parameters
DELTA = 0.7
G = 9.81

# The initial conditions
INITIAL_HEIGHT = 10.0
INITIAL_VELOCITY = 0.0

# The system
system = System()


# The system states
def velocity_dt(data):
    """Calculate the derivative of the vertical speed"""
    return -G


velocity = State(system,
                 derivative_function=velocity_dt,
                 initial_condition=INITIAL_VELOCITY)
height = State(system,
               derivative_function=velocity,
               initial_condition=INITIAL_HEIGHT)


# Define the zero-crossing-event
def bounce_event_function(data):
    """Define the value of the event function for detecting bounces"""
    return height(data)


bounce_event = ZeroCrossEventSource(system,
                                    event_function=bounce_event_function,
                                    direction=-1)


# Define the event-handler
def bounce_event_handler(data):
    """Reverse the direction of motion after a bounce"""
    data.states[height] = np.abs(height(data))
    data.states[velocity] = -DELTA*velocity(data)


bounce_event.register_listener(bounce_event_handler)

# Run a simulation
simulator = Simulator(system,
                      start_time=0.0)
msg = simulator.run_until(time_boundary=8.0)

if msg is not None:
    print("Simulation failed with message '%s'" % msg)
else:
    # Plot the result
    plt.plot(simulator.result.time,
             simulator.result.state[:, height.state_slice])
    plt.title("Bouncing Ball")
    plt.xlabel("Time")
    plt.savefig("04_bouncing_ball_simulation.png")
    plt.show()
