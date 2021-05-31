"""
Simple integrator element with cosine wave input.
"""
import numpy as np
from matplotlib import pyplot as plt
from modypy.model import State, System, signal_function
from modypy.simulation import SimulationResult, Simulator


# Create a new system
system = System()


# Define the cosine signal
@signal_function
def cosine_input(system_state):
    """Calculate the value of the input signal"""
    return np.cos(system_state.time)


integrator_state = State(system, derivative_function=cosine_input)

# Set up a simulation
simulator = Simulator(system, start_time=0.0, max_step=0.1)

# Run the simulation for 10s and capture the result
result = SimulationResult(system, simulator.run_until(time_boundary=10.0))

# Plot the result
input_line, integrator_line = plt.plot(
    result.time,
    cosine_input(result),
    "r",
    result.time,
    integrator_state(result),
    "g",
)
plt.legend((input_line, integrator_line), ("Input", "Integrator State"))
plt.title("Integrator")
plt.xlabel("Time")
plt.savefig("01_integrator_simulation.png")
plt.show()
