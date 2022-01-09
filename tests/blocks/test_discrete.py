"""Tests for the discrete package"""
import numpy as np
import numpy.testing as npt
import pytest

from scipy import stats
from modypy.blocks import discrete
from modypy import model, simulation


class RandomSourceMock:
    """Mock class for checking and collecting proper usage of a random data
    source by the noise source"""

    def __init__(self, random_source, expected_size):
        self.random_source = random_source
        try:
            len(expected_size)
        except TypeError:
            expected_size = (expected_size,)
        self.expected_size = expected_size
        self.logged_data = list()

    def __call__(self, *args, size=(), **kwargs):
        try:
            len(size)
        except TypeError:
            size = (size,)
        assert size == self.expected_size
        ret = self.random_source(size=size, *args, **kwargs)
        self.logged_data.append(ret)
        return ret


@pytest.mark.parametrize("shape", [(), 1, (2, 2)])
def test_noise_source(shape):
    random_source = RandomSourceMock(
        random_source=stats.uniform.rvs, expected_size=shape
    )
    system = model.System()
    trigger = model.Clock(owner=system, period=1.0)
    noise_source = discrete.NoiseSource(
        owner=system, trigger=trigger, shape=shape, random_source=random_source
    )

    simulator = simulation.Simulator(system=system, start_time=0)
    result = simulation.SimulationResult(
        system=system, source=simulator.run_until(time_boundary=10.0)
    )

    npt.assert_equal(
        np.moveaxis(noise_source(result), -1, 0), random_source.logged_data
    )
