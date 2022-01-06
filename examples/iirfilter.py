"""An example of using the `IIRFilter` block."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from modypy.blocks import linear, sources, filters, discrete
from modypy import model, simulation

# Create a signal that is a sum of two sines with different frequencies
sin1_sig = sources.FunctionSignal(
    np.sin, linear.gain(2 * 10 * np.pi, sources.time)
)
sin2_sig = sources.FunctionSignal(
    np.sin, linear.gain(2 * 20 * np.pi, sources.time)
)
sum_sig = linear.sum_signal((sin1_sig, sin2_sig))

# Create a system with a sampling clock at 1kHz
system = model.System()
clock = model.Clock(owner=system, period=1 / 1000.0)

zoh = discrete.zero_order_hold(system, sum_sig, clock)

# Create an order-10 Butterworth High-Pass filter with a limit frequency of
# 15Hz and trigger it with the sampling clock.
filter_spec = signal.butter(10, 15, "hp", fs=1000, output="sos")
filter_block = filters.IIRFilter(
    parent=system, source=zoh, filter_spec=filter_spec, filter_format="sos"
)
filter_block.trigger.connect(clock)

# Run the simulation
sim = simulation.Simulator(system=system, start_time=0)
result = simulation.SimulationResult(system, sim.run_until(time_boundary=1.0))

orig_output = signal.sosfilt(filter_spec, zoh(result))

# Plot the result
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(result.time, sum_sig(result))
ax1.set_title("10 Hz and 20 Hz sinusoids")
ax1.axis([0, 1, -2, 2])
ax2.plot(result.time, filter_block(result), "g")
ax2.plot(result.time, orig_output, "r")
ax2.set_title("After 15 Hz high-pass filter")
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel("Time [seconds]")
plt.tight_layout()
plt.show()
