"""
Tests for the ``modypy.model.events`` module
"""
import pytest

import numpy.testing as npt

from modypy.model import System
from modypy.model.events import MultipleEventSourcesError, EventPort, Clock


def test_event_port():
    """Test the ``EventPort`` class"""

    system = System()

    port_a = EventPort(system)
    port_b = EventPort(system)
    port_c = EventPort(system)
    port_d = EventPort(system)

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


def test_event_connection():
    """
    Test connecting event ports to events.
    """

    system = System()

    clock_a = Clock(system, period=1.0)

    port_a = EventPort(system)
    port_b = EventPort(system)
    port_c = EventPort(system)
    port_d = EventPort(system)

    port_b.connect(clock_a)
    port_d.connect(clock_a)

    port_a.connect(port_b)
    port_c.connect(port_d)
    port_c.connect(port_b)

    assert port_a.source == clock_a
    assert port_b.source == clock_a
    assert port_c.source == clock_a
    assert port_d.source == clock_a


def test_multiple_event_sources_error():
    """
    Test the detection of multiple conflicting event sources on connection.
    """
    system = System()

    clock_a = Clock(system, period=1.0)
    clock_b = Clock(system, period=1.0)

    port_a = EventPort(system)
    port_b = EventPort(system)
    port_c = EventPort(system)
    port_d = EventPort(system)

    port_a.connect(clock_a)
    port_d.connect(clock_b)

    port_b.connect(port_a)
    port_c.connect(port_d)

    assert port_b.source is clock_a
    assert port_c.source is clock_b

    with pytest.raises(MultipleEventSourcesError):
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
    """Test the tick generator of a clock event"""

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


def test_tick_generator_stop_iteration():
    """Test whether the clock tick generator throws a ``StopIteration``
    exception when the current time is after the end time."""

    system = System()

    clock = Clock(system,
                  period=1.0,
                  start_time=0.0,
                  end_time=2.0)

    tick_generator = clock.tick_generator(not_before=3.0)

    with pytest.raises(StopIteration):
        next(tick_generator)
