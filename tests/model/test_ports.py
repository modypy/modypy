"""Tests for ``modypy.model.ports``"""
from unittest.mock import Mock

import pytest

from modypy.model import System, Port, SystemState, PortNotConnectedError
from modypy.model.ports import \
    Port, \
    OutputPort, \
    Signal, \
    InputSignal, \
    ShapeMismatchError, \
    MultipleSignalsError, PortNotConnectedError


def test_port():
    """Test the ``Port`` class"""

    system = System()
    port_a1 = Port(system, shape=3)
    port_a2 = Port(system, shape=3)
    port_a3 = Port(system, shape=3)
    port_b = Port(system, shape=(3, 3))
    port_c = Port(system)

    # Check size calculation
    assert port_a1.size == 3
    assert port_a2.size == 3
    assert port_a3.size == 3
    assert port_b.size == 9
    assert port_c.size == 1

    # Check that the signal is correct
    assert port_a1.signal is None
    assert port_a2.signal is None
    assert port_a3.signal is None
    assert port_b.signal is None
    assert port_c.signal is None

    # Check connections
    port_a1.connect(port_a2)
    port_a3.connect(port_a2)
    assert port_a1.reference == port_a2.reference
    assert port_a2.reference == port_a3.reference


def test_multiple_signals_error():
    """Test occurrence of ``MultipleSignalsError"""

    system = System()
    port_a = Port(system)
    port_b = Port(system)
    signal_a = Signal(system)
    signal_b = Signal(system, value=(lambda x: 0))

    port_a.connect(signal_a)
    port_b.connect(signal_b)
    with pytest.raises(MultipleSignalsError):
        port_a.connect(port_b)


def test_shape_mismatch_error():
    """Test occurrence of ``ShapeMismatchError``"""

    system = System()
    port_a = Port(system, shape=(3, 3))
    port_b = Port(system)
    with pytest.raises(ShapeMismatchError):
        port_a.connect(port_b)


def test_output_port():
    """Test the ``OutputPort`` class"""

    system = System()
    output_port = OutputPort(system, shape=(3, 3))

    # Test the output_slice method
    assert (output_port.output_slice ==
            slice(output_port.output_index,
                  output_port.output_index+output_port.size))

    # Test the output_range method
    assert (output_port.output_range ==
            range(output_port.output_index,
                  output_port.output_index+output_port.size))


def test_signal():
    """Test the ``Signal`` class"""

    system = System()
    signal_a = Signal(system)
    port_a = Port(system)
    port_b = Port(system)
    port_c = Port(system)

    # Test connection port -> signal
    port_a.connect(signal_a)
    port_b.connect(signal_a)
    port_b.connect(port_c)
    assert port_a.signal == signal_a
    assert port_b.signal == signal_a
    assert port_c.signal == signal_a

    # Test the signal_slice method
    assert signal_a.signal_slice == slice(signal_a.signal_index,
                                          signal_a.signal_index+signal_a.size)
    assert port_a.signal_slice == signal_a.signal_slice

    # Test the signal_range method
    assert signal_a.signal_range == range(signal_a.signal_index,
                                          signal_a.signal_index+signal_a.size)
    assert port_a.signal_range == signal_a.signal_range

    # Test the connection of ports with the same signal
    port_a.connect(port_b)


def test_input_signal():
    """Test the ``InputSignal`` class"""

    system = System()
    input_signal = InputSignal(system, shape=(3, 3))

    # Test the input_slice method
    assert (input_signal.input_slice ==
            slice(input_signal.input_index,
                  input_signal.input_index+input_signal.size))

    # Test the input_range method
    assert (input_signal.input_range ==
            range(input_signal.input_index,
                  input_signal.input_index+input_signal.size))

    # Ensure that calling the input signal calls the get_input_value method
    # of the system state.
    system_state = Mock()
    input_signal(system_state)


def test_port_access():
    """Test whether calling a ``Port`` object calls the ``value`` callable
    of the signal using the provider object for connected ports and raises an
    exception for unconnected ports"""

    system = System()

    port = Port(system)
    signal = Signal(system, value=Mock())
    unconnected_port = Port(system)

    port.connect(signal)

    provider = Mock()

    # Check handling of connected ports
    port(provider)
    signal.value.assert_called_with(provider)

    # Check handling of unconnected ports
    with pytest.raises(PortNotConnectedError):
        unconnected_port(provider)


def test_port_not_connected_error():
    """Test the detection of unconnected ports"""

    system = System()
    port = Port(system)

    system_state = SystemState(time=0, system=system)
    with pytest.raises(PortNotConnectedError):
        port(system_state)
