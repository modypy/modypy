"""
Provides the ``Evaluator`` class, which can be used to evaluate the individual
aspects (signals, state derivatives, ...) at any given point in time.
"""
import numpy as np

from .system import System
from .ports import Port
from .states import State


class AlgebraicLoopException(RuntimeError):
    """Exception raised when an algebraic loop is encountered."""


class Evaluator:
    """
    This class allows to evaluate the individual aspects (signals, state
    derivatives, ...) of a system at any given time.
    """
    def __init__(self, time, system: System, state=None):
        self.time = time
        self.system = system

        if state is None:
            state = np.zeros(system.num_states)
        self._state = state

        self._state_derivative = np.empty(system.num_states)
        self.valid_state_derivatives = set()

        self._signals = np.zeros(system.num_signals)
        self.valid_signals = set()
        self.signal_evaluation_stack = list()
        self.signal_evaluation_set = set()

        self._event_values = np.zeros(system.num_events)
        self.valid_event_values = set()

    @property
    def state(self):
        """The current state"""
        return self._state

    @property
    def state_derivative(self):
        """The state derivative vector for the complete system"""
        for state_instance in self.system.states:
            # Trigger calculation of the derivative
            self.get_state_derivative(state_instance)
        return self._state_derivative

    @property
    def signals(self):
        """The signal vector for the complete system."""
        for signal_instance in self.system.signals:
            # Trigger calculation of the signal
            self.get_port_value(signal_instance)
        return self._signals

    @property
    def event_values(self):
        """The event vector for the complete system."""
        for event_instance in self.system.events:
            # Trigger calculation of the event function
            self.get_event_value(event_instance)
        return self._event_values

    def get_state_value(self, state: State):
        """
        Determine the value of a given state.

        :param state: The state
        :return:  The value of the state
        """
        start_index = state.state_index
        end_index = start_index + state.size
        return self._state[start_index:end_index].reshape(state.shape)

    def get_port_value(self, port: Port):
        """
        Determine the value of the given port.

        If the value has not yet been calculated, it will be calculated before
        this method returns. If an algebraic loop is encountered during
        calculation, an ``AlgebraicLoopException`` will be raised.

        :param port: The port for which the value shall be determined
        :return: The value of the port
        :raises AlgebraicLoopException: if an algebraic loop is encountered
            while evaluating the value of the signal
        """
        signal = port.signal
        start_index = signal.signal_index
        end_index = start_index + signal.size

        if signal in self.valid_signals:
            # That signal was already evaluated, so just return the value in
            # proper shape.
            return self._signals[start_index:end_index]\
                .reshape(signal.shape)

        # The signal has not yet been evaluated, so we try to do that now
        if signal in self.signal_evaluation_set:
            # The signal is currently being evaluated, but we got here again,
            # so there must be an algebraic loop.
            raise AlgebraicLoopException()

        # Start evaluation of the signal
        self.signal_evaluation_set.add(signal)
        self.signal_evaluation_stack.append(signal)

        # Perform evaluation
        data = DataProvider(self.time,
                            StateProvider(self),
                            PortProvider(self))
        signal_value = signal.function(data)

        # Ensure that the signal has the correct shape
        signal_value = np.asarray(signal_value).reshape(signal.shape)
        # Assign the value to the signal_vector
        self._signals[start_index:end_index] = signal_value.flatten()
        # Mark the signal as valid
        self.valid_signals.add(signal)

        # End evaluation of the signal
        self.signal_evaluation_set.remove(signal)
        self.signal_evaluation_stack.pop()

        # Return the value of the signal
        return signal_value.reshape(signal.shape)

    def get_state_derivative(self, state):
        """
        Get the state derivative of the given state.

        :param state: The state for which the derivative shall be determined
        :return: The state derivative
        :raises AlgebraicLoopException: if an algebraic loop is encountered
            while evaluating the derivative of the state instance
        """
        start_index = state.state_index
        end_index = start_index + state.size
        if state in self.valid_state_derivatives:
            return self._state_derivative[start_index:end_index].reshape(state.shape)
        data = DataProvider(self.time,
                            StateProvider(self),
                            PortProvider(self))
        state_derivative = state.derivative_function(data)
        state_derivative = np.asarray(state_derivative).reshape(state.shape)
        self._state_derivative[start_index:end_index] = state_derivative.flatten()
        self.valid_state_derivatives.add(state)
        return state_derivative

    def get_event_value(self, event):
        """
        Get the value of the event function of the given event

        :param event: The event for which to calculate the value
        :return: The value of the event function
        :raises AlgebraicLoopException: if an algebraic loop is encountered
            while evaluating the value of the event function
        """
        if event in self.valid_event_values:
            return self._event_values[event.event_index]
        data = DataProvider(self.time,
                            StateProvider(self),
                            PortProvider(self))
        event_value = event.event_function(data)
        self._event_values[event.event_index] = event_value
        self.valid_event_values.add(event)
        return event_value


class DataProvider:
    """
    A ``DataProvider`` provides access to the data about the current point in
    time in the simulation. It has the following properties:

    ``time``
        The current time
    ``states``
        The contents of the current states, accessed by indexing using the
        ``State`` objects.
    ``inputs``
        The contents of the current inputs, accessed by indexing using the
        ``Port`` objects.
    """
    def __init__(self, time, states, inputs):
        self.time = time
        self.states = states
        self.inputs = inputs


class StateProvider:
    """
    A ``StateProvider`` provides access to the state via indexing using the
    ``State`` objects.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getitem__(self, state):
        return self.evaluator.get_state_value(state)


class PortProvider:
    """
    A ``PortProvider`` provides access to the signals via indexing using the
    ``Port`` objects.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getitem__(self, port):
        return self.evaluator.get_port_value(port)
