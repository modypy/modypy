"""
A pendulum.
"""
import numpy as np
import matplotlib.pyplot as plt

from modypy.blocks.linear import integrator
from modypy.model import System, State
from modypy.simulation import Simulator, SimulationResult

# Define the system parameters
LENGTH = 1.0
GRAVITY = 9.81

# Define the initial conditions
ALPHA_0 = np.deg2rad(10)
OMEGA_0 = 0

# Create the system
system = System()


# Define the derivatives of the states
def omega_dt(system_state):
    """Calculate the derivative of the angular velocity"""
    return -GRAVITY/LENGTH * np.sin(alpha(system_state))


# Create the omega state
omega = State(system,
              derivative_function=omega_dt,
              initial_condition=OMEGA_0)

# Create the alpha state
alpha = integrator(system, input_signal=omega, initial_condition=ALPHA_0)

# Run a simulation and capture the result
simulator = Simulator(system, start_time=0.0, max_step=0.1)
result = SimulationResult(system, simulator.run_until(time_boundary=10.0))

# Plot the result
alpha_line, omega_line = \
    plt.plot(result.time, alpha(result), "r",
             result.time, omega(result), "g")
plt.legend((alpha_line, omega_line), ("Alpha", "Omega"))
plt.title("Pendulum")
plt.xlabel("Time")
plt.savefig("02_pendulum_simulation.png")
plt.show()
