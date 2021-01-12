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
def height_dt(data):
    return data.states[velocity]


def velocity_dt(data):
    return -G


height = State(system,
               derivative_function=height_dt,
               initial_condition=INITIAL_HEIGHT)
velocity = State(system,
                 derivative_function=velocity_dt,
                 initial_condition=INITIAL_VELOCITY)


# Define the zero-crossing-event
def bounce_event_function(data):
    return data.states[height]


bounce_event = ZeroCrossEventSource(system,
                                    event_function=bounce_event_function)


# Define the event-handler
def bounce_event_handler(data):
    data.states[height] = np.abs(data.states[height])
    data.states[velocity] = -DELTA*data.states[velocity]


bounce_event.register_listener(bounce_event_handler)

# Run a simulation
simulator = Simulator(system,
                      start_time=0.0)
msg = simulator.run_until(time_boundary=10.0)

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
