"""
Simple integrator element with cosine wave input.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.model import System, State, Signal
from modypy.simulation import Simulator

# Create a new system
system = System()


# Define the function for generating the cosine signal
def cosine_input(system_state):
    """Calculate the value of the input signal"""
    return np.cos(system_state.time)


# Define the input signal
input_signal = Signal(system,
                      shape=1,
                      value=cosine_input)

integrator_state = State(system,
                         shape=1,
                         derivative_function=input_signal)

# Set up a simulation
simulator = Simulator(system,
                      start_time=0.0)

# Run the simulation for 10s
msg = simulator.run_until(time_boundary=10.0)

if msg is not None:
    print("Simulation failed with message '%s'" % msg)
else:
    # Plot the result
    input_line, integrator_line = \
        plt.plot(simulator.result.time,
                 input_signal(simulator.result),
                 "r",
                 simulator.result.time,
                 integrator_state(simulator.result)[0],
                 "g")
    plt.legend((input_line, integrator_line), ("Input", "Integrator State"))
    plt.title("Integrator")
    plt.xlabel("Time")
    plt.savefig("01_integrator_simulation.png")
    plt.show()
