"""
A planet orbiting a sun.
"""
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from modypy.blocks.linear import integrator
from modypy.model import System, State
from modypy.simulation import Simulator, SimulationResult

# Define the system parameters
G = 6.67E-11*(24*60*60)**2
SUN_MASS = 1.989E30
PLANET_ORBIT = 149.6E09
PLANET_ORBIT_TIME = 365.256

# Define the initial state
PLANET_VELOCITY = 2 * np.pi * PLANET_ORBIT / PLANET_ORBIT_TIME
X_0 = PLANET_ORBIT * np.c_[np.cos(np.deg2rad(20)),
                           np.sin(np.deg2rad(20))]
V_0 = 0.5*PLANET_VELOCITY * np.c_[-np.sin(np.deg2rad(20)),
                                  +np.cos(np.deg2rad(20))]

# Create the system
system = System()


# Define the derivatives
def velocity_dt(system_state):
    """Calculate the derivative of the velocity"""
    pos = position(system_state)
    distance = linalg.norm(pos)
    return -G * SUN_MASS/(distance**3) * pos


# Create the states
velocity = State(system,
                 shape=2,
                 derivative_function=velocity_dt,
                 initial_condition=V_0)
position = integrator(system, input_signal=velocity, initial_condition=X_0)

# Run a simulation
simulator = Simulator(system,
                      start_time=0.0,
                      integrator_options={
                          "rtol": 1E-6
                      })
result = SimulationResult(system,
                          simulator.run_until(time_boundary=PLANET_ORBIT_TIME))

# Plot the result
trajectory = position(result)
plt.plot(trajectory[0], trajectory[1])
plt.title("Planet Orbit")
plt.savefig("03_planet_orbit_simulation.png")
plt.show()
