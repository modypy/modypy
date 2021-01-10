"""
A simple example using a DC-motor driving a propeller and sampling the
thrust using a zero-order hold.
"""

import matplotlib.pyplot as plt

from modypy.blocks.aerodyn import Propeller
from modypy.blocks.discrete import zero_order_hold
from modypy.blocks.elmech import DCMotor
from modypy.blocks.sources import constant
from modypy.model import System, Clock
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

# We want to monitor the sampled thrust
sample_clock = Clock(system, period=1/50.0)
thrust_sampler = zero_order_hold(system,
                                 input_port=propeller.thrust,
                                 event_port=sample_clock)

# Run a simulation for 1/2s
simulator = Simulator(system=system, start_time=0)
simulator.run_until(t_bound=0.5)

# Plot the thrust output over time
fig, ax = plt.subplots()
ax.plot(simulator.result.time,
        simulator.result.signals[:, propeller.thrust.signal_slice],
        label="continuous-time")
ax.step(simulator.result.time,
        simulator.result.signals[:, thrust_sampler.signal_slice],
        label="sampled",
        where="post")

ax.set_title("Propeller Simulation")
ax.legend()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Thrust (N)")
fig.savefig("propeller.png")
plt.show()
