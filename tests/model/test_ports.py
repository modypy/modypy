"""Tests for ``modypy.model.ports``"""
from unittest.mock import Mock

import pytest

from modypy.model import System, SystemState
from modypy.model.ports import \
    Port, \
    Signal, \
    InputSignal, \
    ShapeMismatchError, \
    MultipleSignalsError, PortNotConnectedError, signal_method


def test_port():
    """Test the ``Port`` class"""

    port_a1 = Port(shape=3)
    port_a2 = Port(shape=3)
    port_a3 = Port(shape=3)
    port_b = Port(shape=(3, 3))
    port_c = Port()

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

    port_a = Port()
    port_b = Port()
    signal_a = Signal()
    signal_b = Signal(value=(lambda x: 0))

    port_a.connect(signal_a)
    port_b.connect(signal_b)
    with pytest.raises(MultipleSignalsError):
        port_a.connect(port_b)


def test_shape_mismatch_error():
    """Test occurrence of ``ShapeMismatchError``"""

    port_a = Port(shape=(3, 3))
    port_b = Port()
    with pytest.raises(ShapeMismatchError):
        port_a.connect(port_b)


def test_signal():
    """Test the ``Signal`` class"""

    signal_a = Signal()
    port_a = Port()
    port_b = Port()
    port_c = Port()

    # Test connection port -> signal
    port_a.connect(signal_a)
    port_b.connect(signal_a)
    port_b.connect(port_c)
    assert port_a.signal == signal_a
    assert port_b.signal == signal_a
    assert port_c.signal == signal_a

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

    port = Port()
    signal = Signal(value=Mock())
    unconnected_port = Port()

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
    port = Port()

    system_state = SystemState(time=0, system=system)
    with pytest.raises(PortNotConnectedError):
        port(system_state)


def test_signal_method():
    """Test the implementation of signal method"""

    class TestClass:
        @signal_method
        def test_method(self, data):
            pass

    # Ensure that the value of the method in the class does not change
    method_1 = TestClass.test_method
    method_2 = TestClass.test_method
    assert method_1 is method_2

    # Ensure that the method resolves to a unique signal on an object
    test_object_1 = TestClass()
    test_object_2 = TestClass()
    signal_1 = test_object_1.test_method
    signal_2 = test_object_1.test_method
    signal_3 = test_object_2.test_method

    assert isinstance(signal_1, Signal)
    assert signal_1 is signal_2
    assert signal_1 is not signal_3
