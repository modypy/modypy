"""Tests for the simtree.model.blocks module"""
import pytest

from simtree.model import System, Block
from simtree.model.ports import Port, Signal, \
    MultipleSignalsError, ShapeMismatchError


class SimpleBlock(Block):
    """A simple block with a few ports and signals"""

    def __init__(self, parent):
        Block.__init__(self, parent)
        self.port_a = Port(self, shape=(3, 3))
        self.port_b = Port(self, shape=(3, 3))
        self.port_c = Port(self, shape=1)
        self.port_d = Port(self, shape=3)

        self.signal_a = Signal(self, shape=(3, 3), function=self.signal_a_func())
        self.signal_b = Signal(self, shape=(3, 3), function=self.signal_b_func())

    def signal_a_func(self):
        pass

    def signal_b_func(self):
        pass


def test_sizes():
    """Test the proper calculation of signal sizes"""
    system = System()
    block_a = SimpleBlock(system)

    assert block_a.port_a.size == 9
    assert block_a.port_c.size == 1
    assert block_a.port_d.size == 3


def test_connect_unconnected():
    """Test connection of two unconnected signals"""
    system = System()
    block_a = SimpleBlock(system)
    block_b = SimpleBlock(system)

    # Connect two unconnected ports
    block_a.port_a.connect(block_b.port_a)
    assert block_a.port_a.reference == block_b.port_a.reference


def test_connect_to_signal():
    """Test connection of a port to a signal"""
    system = System()
    block_a = SimpleBlock(system)
    block_b = SimpleBlock(system)

    block_a.port_a.connect(block_b.signal_a)
    assert block_a.port_a.signal == block_b.signal_a


def test_connect_to_same_signal():
    """Test connection of two ports which are connected to the same signal"""
    system = System()
    block_a = SimpleBlock(system)
    block_b = SimpleBlock(system)

    block_a.port_a.connect(block_a.signal_a)
    block_a.signal_a.connect(block_b.port_a)
    block_a.port_a.connect(block_b.port_a)
    assert block_a.port_a.signal == block_a.signal_a
    assert block_b.port_a.signal == block_a.signal_a


def test_connect_to_different_signals():
    """Test the connection of two ports which are connected to different signals"""
    system = System()
    block_a = SimpleBlock(system)
    block_b = SimpleBlock(system)

    block_a.port_a.connect(block_a.signal_a)
    block_b.port_a.connect(block_b.signal_b)
    with pytest.raises(MultipleSignalsError):
        block_a.port_a.connect(block_b.port_a)


def test_connect_incompatible_ports():
    """Test the connection of two ports with incompatible shapes"""
    system = System()
    block_a = SimpleBlock(system)

    with pytest.raises(ShapeMismatchError):
        block_a.port_a.connect(block_a.port_c)
