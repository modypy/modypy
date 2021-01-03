"""Tests for the simtree.model.blocks module"""
import pytest

from simtree.model import ModelContext, Block
from simtree.model.ports import Port, Signal, output_signal, \
    MultipleSignalsError, ShapeMismatchError


@pytest.fixture
def simple_model():
    """A simple model with two blocks and several signals"""
    class SimpleBlock(Block):
        """A simple block with a few ports and signals"""
        port_a = Port(shape=(3, 3))
        port_b = Port(shape=(3, 3))
        port_c = Port(shape=1)
        port_d = Port(shape=3)

        signal_a = Signal(shape=(3, 3))
        signal_b = Signal(shape=(3, 3))

        @output_signal(shape=3)
        def output_signal_a(self):
            """A simple output signal"""
            return [0, 0, 0]

        @output_signal(shape=3)
        def output_signal_b(self):
            """A simple output signal"""
            return [0, 0, 0]

    context = ModelContext()
    block_a = SimpleBlock(context)

    return block_a


def test_connect_unconnected(simple_model):
    """Test connection of two unconnected signals"""
    block_a = simple_model

    # Connect two unconnected ports
    block_a.port_a = block_a.port_b
    assert (block_a.port_a.reference == block_a.port_b or
            block_a.port_b.reference == block_a.port_a)


def test_connect_to_signal(simple_model):
    """Test connection of a port to a signal"""
    block_a = simple_model

    block_a.port_a = block_a.signal_a
    assert block_a.port_a.signal == block_a.signal_a


def test_connect_to_same_signal(simple_model):
    """Test connection of two ports which are connected to the same signal"""
    block_a = simple_model

    block_a.port_a = block_a.signal_a
    block_a.port_b = block_a.signal_a
    block_a.port_a = block_a.port_b
    assert block_a.port_a.signal == block_a.signal_a
    assert block_a.port_b.signal == block_a.signal_a


def test_connect_to_different_signals(simple_model):
    """Test the connection of two ports which are connected to different signals"""
    block_a = simple_model

    block_a.port_a = block_a.signal_a
    block_a.port_b = block_a.signal_b
    with pytest.raises(MultipleSignalsError):
        block_a.port_a = block_a.port_b


def test_connect_incompatible_ports(simple_model):
    """Test the connection of two ports with incompatible shapes"""
    block_a = simple_model

    with pytest.raises(ShapeMismatchError):
        block_a.port_a = block_a.port_c


def test_connect_output_port(simple_model):
    """Test the connection of a port to an output port"""
    block_a = simple_model

    print(block_a.output_signal_b)
    block_a.port_d = block_a.output_signal_b
    assert block_a.port_d.signal == block_a.output_signal_b
