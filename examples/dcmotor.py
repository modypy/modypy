"""
A simple example using a DC-motor driving a propeller.
"""

import matplotlib.pyplot as plt

from modypy.blocks.aerodyn import Propeller
from modypy.blocks.elmech import DCMotor
from modypy.blocks.sources import constant
from modypy.model import System
from modypy.simulation import Simulator
from modypy.utils.uiuc_db import load_static_propeller

# Import thrust and torque coefficients from the UIUC propeller database
thrust_coeff, torque_coeff = \
    load_static_propeller(
        'volume-1/data/apcsf_8x3.8_static_2777rd.txt',
        interp_options={
            "bounds_error": False,
            "fill_value": "extrapolate"
        }
    )

system = System()

# Create the engine, consisting of the motor and a propeller
dcmotor = DCMotor(system,
                  Kv=789.E-6,
                  R=43.3E-3,
                  L=1.9E-3,
                  J=5.284E-6)
propeller = Propeller(system,
                      thrust_coeff=thrust_coeff,
                      power_coeff=torque_coeff,
                      diameter=8*25.4E-3)

# Connect the motor and propeller to each other
propeller.torque.connect(dcmotor.external_torque)
dcmotor.speed_rps.connect(propeller.speed_rps)

# Create the sources for voltage and air density
voltage = constant(system, value=3.5)
density = constant(system, value=1.29)

# Connect the voltage to the motor
voltage.connect(dcmotor.voltage)
# Connect the density to the propeller
density.connect(propeller.density)

# Run a simulation for 1/2s
simulator = Simulator(system=system, start_time=0)
simulator.run_until(t_bound=0.5)

# Plot the thrust output over time
plt.plot(simulator.result.time, simulator.result.signals[:, propeller.thrust.signal_slice])
plt.title("Propeller Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.show()
