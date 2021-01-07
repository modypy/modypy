"""
Tests for the ``modypy.model.clocks`` module
"""
import pytest

import numpy.testing as npt

from modypy.model import System
from modypy.model.clocks import ClockPort, Clock, MultipleClocksError

def test_clock_port():
    """Test the ``ClockPort`` class"""

    system = System()

    port_a = ClockPort(system)
    port_b = ClockPort(system)
    port_c = ClockPort(system)
    port_d = ClockPort(system)

    port_a.register_listener("1")
    port_c.register_listener("2")

    port_a.connect(port_b)
    port_c.connect(port_d)
    port_c.connect(port_b)

    port_d.register_listener("3")

    assert port_a.listeners is port_b.listeners
    assert port_b.listeners is port_c.listeners
    assert port_c.listeners is port_d.listeners

    assert port_a.listeners == {"1", "2", "3"}


def test_clock():
    """
    Test the ``Clock`` class
    """

    system = System()

    clock_a = Clock(system, period=1.0)

    port_a = ClockPort(system)
    port_b = ClockPort(system)
    port_c = ClockPort(system)
    port_d = ClockPort(system)

    port_b.connect(clock_a)
    port_d.connect(clock_a)

    port_a.connect(port_b)
    port_c.connect(port_d)
    port_c.connect(port_b)

    assert port_a.clock == clock_a
    assert port_b.clock == clock_a
    assert port_c.clock == clock_a
    assert port_d.clock == clock_a


def test_multiple_clocks_error():
    system = System()

    clock_a = Clock(system, period=1.0)
    clock_b = Clock(system, period=1.0)

    port_a = ClockPort(system)
    port_b = ClockPort(system)
    port_c = ClockPort(system)
    port_d = ClockPort(system)

    port_a.connect(clock_a)
    port_d.connect(clock_b)

    port_b.connect(port_a)
    port_c.connect(port_d)

    assert port_b.clock is clock_a
    assert port_c.clock is clock_b

    with pytest.raises(MultipleClocksError):
        port_c.connect(port_b)


@pytest.mark.parametrize(
    "start_time, end_time, run_before_start, expected",
    [
        [0.0, None, False, [0.0, 1.0, 2.0, 3.0]],
        [1.5, None, False, [1.5, 2.5, 3.5, 4.5]],
        [0.5, None, True,  [0.5, 1.5, 2.5, 3.5]],
        [0.5, 2.0,  False, [0.5, 1.5]]
    ]
)
def test_tick_generator(start_time,
                        end_time,
                        run_before_start,
                        expected):
    """Test an endless tick generator"""
    system = System()

    clock = Clock(system,
                  period=1.0,
                  start_time=start_time,
                  end_time=end_time,
                  run_before_start=run_before_start)

    tick_generator = clock.tick_generator(not_before=0.0)

    # Fetch at most 4 ticks
    ticks = [tick for _idx, tick in zip(range(4), tick_generator)]

    npt.assert_almost_equal(ticks, expected)
