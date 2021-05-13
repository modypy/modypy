"""
Simple integrator element with cosine wave input.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, State, signal_function
from modypy.simulation import Simulator

# Create a new system
system = System()


# Define the cosine signal
@signal_function(shape=1)
def cosine_input(system_state):
    """Calculate the value of the input signal"""
    return np.cos(system_state.time)


integrator_state = State(system,
                         shape=1,
                         derivative_function=cosine_input)

# Set up a simulation
simulator = Simulator(system,
                      start_time=0.0)

# Run the simulation for 10s
simulator.run_until(time_boundary=10.0)

# Plot the result
input_line, integrator_line = \
    plt.plot(simulator.result.time,
             cosine_input(simulator.result),
             "r",
             simulator.result.time,
             integrator_state(simulator.result)[0],
             "g")
plt.legend((input_line, integrator_line), ("Input", "Integrator State"))
plt.title("Integrator")
plt.xlabel("Time")
plt.savefig("01_integrator_simulation.png")
plt.show()
