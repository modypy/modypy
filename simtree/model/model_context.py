"""
Provides class ``ModelContext`` for representing the context of a system model
"""


class ModelContext:
    """
    This class represents the context for a system model.

    It contains information about the mapping of signals and states to the
    signal and state vector.
    """
    def __init__(self):
        self.signal_line_count = 0
        self.state_line_count = 0

    @property
    def context(self):
        """The model context itself"""
        return self

    def allocate_states(self, state_count):
        """
        Allocate a number of states for the state vector.

        :param state_count: The number of states to allocate
        :return: The index of the first state allocated
        """
        state_line_index = self.state_line_count
        self.state_line_count += state_count
        return state_line_index

    def allocate_signals(self, signal_count):
        """
        Allocate a number of signals for the signal vector.

        :param signal_count: The number of signals to allocate
        :return: The index of the first signal allocated
        """
        signal_line_index = self.signal_line_count
        self.signal_line_count += signal_count
        return signal_line_index
