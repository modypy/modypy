"""Tests for ``modypy.model.ports``"""
import pytest

from modypy.model import System
from modypy.model.ports import \
    Port, \
    OutputPort, \
    Signal, \
    InputSignal, \
    ShapeMismatchError, \
    MultipleSignalsError


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
    assert output_port.output_slice == slice(output_port.output_index,
                                             output_port.output_index+output_port.size)


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

    # Test the connection of ports with the same signal
    port_a.connect(port_b)


def test_input_signal():
    """Test the ``InputSignal`` class"""

    system = System()
    input_signal = InputSignal(system, shape=(3, 3))

    # Test the input_slice method
    assert input_signal.input_slice == slice(input_signal.input_index,
                                             input_signal.input_index+input_signal.size)
