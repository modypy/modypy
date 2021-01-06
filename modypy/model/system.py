"""
Provides classes for constructing systems and block hierarchies.
"""
import numpy as np


class System:
    """
    A system is a group of interacting ``Block`` instances.
    """
    def __init__(self):
        self.num_signals = 0
        self.signals = list()

        self.num_states = 0
        self.states = list()

        self.events = list()

        self.num_inputs = 0
        self.inputs = list()

        self.num_outputs = 0
        self.outputs = list()

    @property
    def system(self):
        """The system itself"""
        return self

    @property
    def initial_condition(self):
        """The initial condition vector for the state of this system"""
        initial_condition = np.zeros(self.num_states)
        for state in self.states:
            start_index = state.state_index
            end_index = start_index + state.size
            initial_condition[start_index:end_index] = \
                state.initial_condition.flatten()
        return initial_condition

    @property
    def initial_input(self):
        """The initial inputs of this system"""
        initial_inputs = np.zeros(self.num_inputs)
        for signal in self.inputs:
            initial_inputs[signal.input_slice] = signal.value.flatten()
        return initial_inputs

    def allocate_signal_lines(self, count):
        """
        Allocate a sequence of consecutive signal lines.

        :param count: The number of signal lines to allocate
        :return: The index of the first signal line allocated
        """
        start_index = self.num_signals
        self.num_signals += count
        return start_index

    def allocate_state_lines(self, count):
        """
        Allocate a sequence of consecutive state lines.

        :param count: The number of state lines to allocate
        :return: The index of the first state line allocated
        """
        start_index = self.num_states
        self.num_states += count
        return start_index

    def allocate_input_lines(self, count):
        """
        Allocate a sequence of consecutive input lines.

        :param count: The number of input lines to allocate
        :return: The index of the first input line allocated
        """
        start_index = self.num_inputs
        self.num_inputs += count
        return start_index

    def allocate_output_lines(self, count):
        """
        Allocate a sequence of consecutive input lines.

        :param count: The number of input lines to allocate
        :return: The index of the first input line allocated
        """
        start_index = self.num_outputs
        self.num_outputs += count
        return start_index

    def register_event(self, event):
        """
        Register an event.

        :return: The index of the event line allocated for the event
        """
        event_index = len(self.events)
        self.events.append(event)
        return event_index

    @property
    def num_events(self):
        """The number of events registered with this system"""
        return len(self.events)


class Block:
    """
    A block describes the interface and behaviour of a part of a system.

    The interface of a block is mainly defined by ports, by which signal
    connections can be established to other parts of the system.

    The behaviour of a block is defined by states.
    """
    def __init__(self, parent):
        self.parent = parent
        self.system = self.parent.system
