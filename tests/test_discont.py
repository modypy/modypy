import numpy as np
import numpy.testing as npt

from modypy.blocks.discont import saturation
from modypy.model import System, Signal, Clock
from modypy.simulation import Simulator


def test_saturation():
    system = System()
    Clock(system, period=0.01)
    sine_source = Signal(owner=system,
                         value=lambda data: np.sin(2 * np.pi * data.time))
    saturated_out = saturation(system,
                               sine_source,
                               lower_limit=-0.5,
                               upper_limit=0.6)

    simulator = Simulator(system, start_time=0.0)
    message = simulator.run_until(time_boundary=1.0)

    assert message is None

    sine_data = sine_source(simulator.result)
    saturated_data = saturated_out(simulator.result)
    saturated_exp = np.minimum(np.maximum(sine_data, -0.5), 0.6)
    npt.assert_almost_equal(saturated_data, saturated_exp)
