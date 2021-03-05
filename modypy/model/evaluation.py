"""
Provides the ``Evaluator`` class, which can be used to evaluate the individual
aspects (signals, state derivatives, ...) at any given point in time.
"""
import warnings
import numpy as np

from .system import System
from .ports import Port
from .states import State


class AlgebraicLoopError(RuntimeError):
    """Exception raised when an algebraic loop is detected on evaluation"""


class PortNotConnectedError(RuntimeError):
    """Exception when a port is evaluated that is not connected to a signal"""


class Evaluator:
    """This class allows to evaluate the individual aspects (signals, state
    derivatives, ...) of a system at any given time."""

    def __init__(self, time, system: System, state=None, inputs=None):
        self.time = time
        self.system = system

        if state is None:
            state = system.initial_condition.copy()
        self._state = state

        self._state_derivative = np.zeros(system.num_states)
        self.valid_state_derivatives = set()

        self._signals = np.zeros(system.num_signals)
        self.valid_signals = set()
        self.signal_evaluation_stack = list()
        self.signal_evaluation_set = set()

        self._event_values = np.zeros(system.num_events)
        self.valid_event_values = set()

        if inputs is not None:
            for signal in self.system.inputs:
                self._signals[signal.signal_slice] = inputs[signal.input_slice]
                self.valid_signals.add(signal)

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
    def inputs(self):
        """The input vector for the complete system"""
        input_vector = np.empty(self.system.num_inputs)
        for signal in self.system.inputs:
            signal_value = self.get_port_value(signal)
            input_vector[signal.input_slice] = signal_value.flatten()
        return input_vector

    @property
    def signals(self):
        """The signal vector for the complete system."""
        for signal_instance in self.system.signals:
            # Trigger calculation of the signal
            self.get_port_value(signal_instance)
        return self._signals

    @property
    def outputs(self):
        """The output vector for the complete system"""
        output_vector = np.empty(self.system.num_outputs)
        for port in self.system.outputs:
            port_value = self.get_port_value(port)
            output_vector[port.output_slice] = port_value.flatten()
        return output_vector

    @property
    def event_values(self):
        """The event vector for the complete system"""
        for event_instance in self.system.events:
            # Trigger calculation of the event value
            self.get_event_value(event_instance)
        return self._event_values

    def get_state_value(self, state: State):
        """Determine the value of a given state.

        Args:
          state: The state

        Returns:
          The value of the state
        """
        return self._state[state.state_slice].reshape(state.shape)

    def get_port_value(self, port: Port):
        """Determine the value of the given port.

        If the value has not yet been calculated, it will be calculated before
        this method returns. If an algebraic loop is encountered during
        calculation, an ``AlgebraicLoopError`` will be raised.

        Args:
          port: The port for which the value shall be determined

        Returns:
          The value of the port

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the value of the signal
          PortNotConnectedError: if the port is not connected to a signal
        """

        if port.size == 0:
            # An empty port has no value
            return np.empty(port.shape)

        signal = port.signal

        if signal is None:
            # This port is not connected to any signal, so we cannot determine
            # its value.
            raise PortNotConnectedError()

        if signal in self.valid_signals:
            # That signal was already evaluated, so just return the value in
            # proper shape.
            return self._signals[signal.signal_slice]\
                .reshape(signal.shape)

        # The signal has not yet been evaluated, so we try to do that now
        if signal in self.signal_evaluation_set:
            # The signal is currently being evaluated, but we got here again,
            # so there must be an algebraic loop.
            raise AlgebraicLoopError()

        # Start evaluation of the signal
        self.signal_evaluation_set.add(signal)
        self.signal_evaluation_stack.append(signal)

        # Perform evaluation
        data = DataProvider(evaluator=self,
                            time=self.time,
                            states=StateProvider(self),
                            signals=PortProvider(self))
        if callable(signal.value):
            signal_value = signal.value(data)
        else:
            signal_value = signal.value

        # Ensure that the signal has the correct shape
        signal_value = np.asarray(signal_value).reshape(signal.shape)
        # Assign the value to the signal_vector
        self._signals[signal.signal_slice] = signal_value.flatten()
        # Mark the signal as valid
        self.valid_signals.add(signal)

        # End evaluation of the signal
        self.signal_evaluation_set.remove(signal)
        self.signal_evaluation_stack.pop()

        # Return the value of the signal
        return signal_value.reshape(signal.shape)

    def get_state_derivative(self, state):
        """Get the state derivative of the given state.

        Args:
          state: The state for which the derivative shall be determined

        Returns:
          The state derivative

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the derivative of the state instance

        """
        if state in self.valid_state_derivatives:
            return self._state_derivative[state.state_slice]\
                .reshape(state.shape)
        if state.derivative_function is not None:
            data = DataProvider(evaluator=self,
                                time=self.time,
                                states=StateProvider(self),
                                signals=PortProvider(self))
            state_derivative = state.derivative_function(data)
            state_derivative = \
                np.asarray(state_derivative).reshape(state.shape)
            self._state_derivative[state.state_slice] = \
                state_derivative.flatten()
        else:
            state_derivative = self._state_derivative[state.state_slice]
        self.valid_state_derivatives.add(state)
        return state_derivative

    def get_event_value(self, event):
        """Get the value of the event function of the given event

        Args:
          event: The event for which to calculate the value

        Returns:
          The value of the event function

        Raises:
          AlgebraicLoopError: if an algebraic loop is encountered while
            evaluating the value of the event function
        """

        if event in self.valid_event_values:
            return self._event_values[event.event_index]
        data = DataProvider(evaluator=self,
                            time=self.time,
                            states=StateProvider(self),
                            signals=PortProvider(self))
        event_value = event.event_function(data)
        self._event_values[event.event_index] = event_value
        self.valid_event_values.add(event)
        return event_value


class DataProvider:
    """A ``DataProvider`` provides access to the data about the current point in
    time in the simulation. It has the following properties:

    ``time``
        The current time
    ``states``
        The contents of the current states, accessed by indexing using the
        ``State`` objects.
    ``signals``
        The contents of the current signals, accessed by indexing using the
        ``Port`` objects.
    """
    def __init__(self, evaluator, time, states, signals):
        self.evaluator = evaluator
        self.time = time
        self.states = states
        self.signals = signals

        # Forward the relevant functions to the evaluator
        self.get_port_value = evaluator.get_port_value
        self.get_state_value = evaluator.get_state_value
        self.get_event_value = evaluator.get_event_value

    @property
    def inputs(self):
        """Old way of accessing the signals dictionary. Deprecated."""
        warnings.warn(DeprecationWarning("The ``inputs`` property of the "
                                         "``DataProvider`` class is deprecated "
                                         "and will be removed in the future. "
                                         "Use the ``signals`` property "
                                         "instead."))
        return self.signals


class StateProvider:
    """A ``StateProvider`` provides access to the state via indexing using the
    ``State`` objects.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getitem__(self, state):
        return self.evaluator.get_state_value(state)


class PortProvider:
    """A ``PortProvider`` provides access to the signals via indexing using the
    ``Port`` objects.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getitem__(self, port):
        return self.evaluator.get_port_value(port)
