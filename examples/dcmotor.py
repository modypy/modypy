import matplotlib.pyplot as plt

from simtree.blocks import NonLeafBlock
from simtree.blocks.aerodyn import Propeller, Thruster
from simtree.blocks.elmech import DCMotor
from simtree.blocks.sources import Constant
from simtree.compiler import compile
from simtree.simulator import Simulator
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
dcmotor = DCMotor(Kv=789.E-6,
                  R=43.3E-3,
                  L=1.9E-3,
                  J=5.284E-6,
                  name="dcmotor")
propeller = Propeller(thrust_coeff=thrust_coeff,
                      power_coeff=torque_coeff,
                      diameter=8*25.4E-3,
                      name="apc_slowflyer")
engine = NonLeafBlock(children=[dcmotor, propeller],
                      num_inputs=2,
                      num_outputs=2)
engine.connect_input(0, dcmotor, 0)       # Connect input voltage
engine.connect_input(1, propeller, 1)     # Connect air density
engine.connect(dcmotor, 0, propeller, 0)  # Connect propeller speed
engine.connect(propeller, 1, dcmotor, 1)  # Connect propeller braking torque
engine.connect_output(propeller, 0, 0)    # Connect generated thrust
engine.connect_output(dcmotor, 1, 1)      # Connect generated torque

# Create the main system
voltage = Constant(value=3.5, name="voltage")
density = Constant(value=1.29, name="density")
system = NonLeafBlock(children=[density, voltage, engine],
                      num_outputs=2)
system.connect(voltage, 0, engine, 0)
system.connect(density, 0, engine, 1)
system.connect_output(engine, range(2), range(2))

# Compile the system
compiled_sys = compile(system)

# Run the simulator
simulator = Simulator(system=compiled_sys, t0=0, t_bound=0.3)
simulator.run()

# Plot the results
plt.plot(simulator.result.time, simulator.result.output[:, 0])
plt.title("Propeller Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.show()
