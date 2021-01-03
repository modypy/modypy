"""
Defines meta-properties for states.
"""
import math

from .block import MetaProperty


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

    def create_instance(self, block):
        state_index = block.context.allocate_states(self.size)
        return StateInstance(block, self, state_index)

    def get_instance(self, block):
        return getattr(block, "_state_%s" % self.name)

    def register_instance(self, block, instance):
        setattr(block, "_state_%s" % self.name, instance)

    def derivative(self, function):
        """Decorator to define the derivative function of the state"""
        self.derivative_function = function
        return function


class StateInstance:
    """
    An instance of a state
    """
    def __init__(self, block, state, state_index):
        self.block = block
        self.state = state
        self.state_index = state_index

# TODO: StateSignal for States that are exported as signals
