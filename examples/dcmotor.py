import matplotlib.pyplot as plt

from simtree.blocks.aerodyn import Propeller
from simtree.blocks.elmech import DCMotor
from simtree.blocks.sources import constant
from simtree.model import System
from simtree.simulation import Simulator
from simtree.utils.uiuc_db import load_static_propeller

# Import propeller data from UIUC database
thrust_coeff, torque_coeff = \
    load_static_propeller(
        'volume-1/data/apcsf_8x3.8_static_2777rd.txt',
        interp_options={
            "bounds_error": False,
            "fill_value": "extrapolate"
        }
    )

# Create the Engine
system = System()
dcmotor = DCMotor(system,
                  Kv=789.E-6,
                  R=43.3E-3,
                  L=1.9E-3,
                  J=5.284E-6)
propeller = Propeller(system,
                      thrust_coeff=thrust_coeff,
                      power_coeff=torque_coeff,
                      diameter=8*25.4E-3)

# Connect the signals
propeller.torque.connect(dcmotor.external_torque)
dcmotor.speed_rps.connect(propeller.speed_rps)

# Create the sources
voltage = constant(system, value=3.5)
density = constant(system, value=1.29)

# Connect the sources
voltage.connect(dcmotor.voltage)
density.connect(propeller.density)

# Run the simulator
simulator = Simulator(system=system, start_time=0)
simulator.run_until(t_bound=0.5)

# Plot the results
plt.plot(simulator.result.time, simulator.result.signals[:, propeller.thrust.signal_slice])
plt.title("Propeller Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.show()
