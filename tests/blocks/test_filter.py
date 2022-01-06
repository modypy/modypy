"""Tests for the filter package"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy import signal

from modypy import model, simulation

from modypy.blocks import sources, linear, filters, discrete


@pytest.mark.parametrize(
    "filter_type,converter",
    [("sos", lambda *x: x), ("ba", signal.tf2sos), ("zpk", signal.zpk2sos)],
)
def test_iir_filter(filter_type, converter):
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
    # 15 Hz and trigger it with the sampling clock.
    filter_spec = signal.butter(10, 15, "hp", fs=1000, output=filter_type)
    filter_spec_sos = converter(*filter_spec)
    filter_block = filters.IIRFilter(
        parent=system,
        source=zoh,
        filter_spec=filter_spec,
        filter_format=filter_type,
    )
    filter_block.trigger.connect(clock)

    # Run the simulation
    sim = simulation.Simulator(system=system, start_time=0)
    result = simulation.SimulationResult(
        system, sim.run_until(time_boundary=1.0)
    )

    # Run the original filter
    orig_output = signal.sosfilt(filter_spec_sos, zoh(result))

    npt.assert_almost_equal(filter_block(result), orig_output)


def test_iir_filter_invalid_format():
    system = model.System()
    filter_spec = signal.butter(10, 15, "hp", fs=1000, output="sos")
    with pytest.raises(ValueError):
        filters.IIRFilter(
            parent=system,
            source=sources.constant(0),
            filter_spec=filter_spec,
            filter_format="invalid",
        )
