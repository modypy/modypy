"""
Provides class ``ModelContext`` for representing the context of a system model
"""
from abc import ABC, abstractmethod


class BlockContainer(ABC):
    @abstractmethod
    def register_block(self, block):
        pass


class ModelContext(BlockContainer):
    """
    This class represents the context for a system model.

    It contains information about the mapping of signals and states to the
    signal and state vector.
    """
    def __init__(self):
        BlockContainer.__init__(self)
        self.signal_instances = set()
        self.signal_line_count = 0
        self.state_instances = set()
        self.state_line_count = 0

    @property
    def context(self):
        """The model context itself"""
        return self

    def register_state_instance(self, state_instance):
        """
        Register a state instance and allocate a number of state lines for it.

        :param state_instance: The state instance to register
        """
        state_instance.state_index = self.state_line_count
        self.state_line_count += state_instance.size
        self.state_instances.add(state_instance)

    def register_signal_instance(self, signal_instance):
        """
        Register a signal instance and allocate a number of signal lines for it.

        :param signal_instance: The signal instance to register
        """
        signal_instance.signal_index = self.signal_line_count
        self.signal_line_count += signal_instance.size
        self.signal_instances.add(signal_instance)
