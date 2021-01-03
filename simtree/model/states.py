"""
Defines meta-properties for states.
"""
import math

from .blocks import MetaProperty
from .ports import Signal, SignalInstance


class State(MetaProperty):
    """
    A state describes a portion of the state of a block.
    """
    def __init__(self, shape=1, initial_value=None, *args, **kwargs):
        MetaProperty.__init__(self, *args, **kwargs)
        self.shape = shape
        if isinstance(self.shape, int):
            self.size = self.shape
        else:
            self.size = math.prod(self.shape)
        self.initial_value = initial_value
        self.derivative_function = None

    def __get__(self, block, owner):
        if block is None:
            return self
        return self.get_instance(block)

    def instantiate_for_block(self, block):
        instance = self.create_instance(block)
        block.context.register_state_instance(instance)
        setattr(block, "_state_%s" % self.name, instance)

    def create_instance(self, block):
        return StateInstance(block, self)

    def get_instance(self, block):
        return getattr(block, "_state_%s" % self.name)

    def derivative(self, function):
        """Decorator to define the derivative function of the state"""
        self.derivative_function = function
        return function


class StateInstance:
    """
    An instance of a state
    """
    def __init__(self, block, state):
        self.block = block
        self.state = state
        self.state_index = None

    @property
    def size(self):
        return self.state.size


class StateSignal(State, Signal):
    """
    A state signal is a state that is also provided as an output signal.

    By using ``StateSignal`` instead of ``State`` and a separate
    ``output_signal``, the state and the signal have the same name and no
    additional function for the output signal needs to be supplied.
    """
    def __init__(self, shape=1, initial_value=None, *args, **kwargs):
        State.__init__(self, shape, initial_value, *args, **kwargs)
        Signal.__init__(self, shape, *args, **kwargs)
        self.function = (lambda block, time, data: getattr(data, self.name))

    def instantiate_for_block(self, block):
        instance = self.create_instance(block)
        block.context.register_signal_instance(instance)
        block.context.register_state_instance(instance)
        setattr(block, "_state_%s" % self.name, instance)

    def create_instance(self, block):
        return StateSignalInstance(block, self)


class StateSignalInstance(StateInstance, SignalInstance):
    """
    An instance of a state signal.
    """
    def __init__(self, block, state_signal):
        StateInstance.__init__(self, block, state_signal)
        SignalInstance.__init__(self, block, state_signal)
