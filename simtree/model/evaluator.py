"""
Provides the ``Evaluator`` class, which can be used to evaluate the individual
aspects (signals, state derivatives, ...) at any given point in time.
"""
import numpy as np

from .model_context import ModelContext
from .ports import PortInstance
from .states import StateInstance


class AlgebraicLoopException(RuntimeError):
    """Exception raised when an algebraic loop is encountered."""


class Evaluator:
    """
    This class allows to evaluate the individual aspects (signals, state
    derivatives, ...) of a system at any given time.
    """
    def __init__(self, time, context: ModelContext, state_vector=None):
        self.time = time
        self.context = context

        if state_vector is None:
            state_vector = np.zeros(context.state_line_count)
        self.state_vector = state_vector
        self._state_derivative = np.empty(context.state_line_count)
        self.valid_state_derivatives = set()

        self._signal_vector = np.zeros(context.signal_line_count)
        self.valid_signals = set()

        self.signal_evaluation_stack = list()
        self.signal_evaluation_set = set()

    @property
    def state_derivative(self):
        """
        The state derivative vector for the complete system.
        """
        for state_instance in self.context.state_instances:
            # Trigger calculation of the derivative
            self.get_state_derivative(state_instance)
        return self._state_derivative

    @property
    def signal_vector(self):
        """
        The signal vector for the complete system.
        """
        for signal_instance in self.context.signal_instances:
            # Trigger calculation of the signal
            self.get_port_value(signal_instance)
        return self._signal_vector

    def get_item_value(self, item):
        """
        Determine the value of a specific item (state instance, port instance,
        ...).

        :param item: The item of which shall be determined
        :return: The value of the item
        """
        if isinstance(item, StateInstance):
            return self.get_state_value(item)
        if isinstance(item, PortInstance):
            return self.get_port_value(item)
        raise AttributeError()

    def get_state_value(self, state_instance):
        """
        Determine the value of a given state instance.

        :param state_instance: The state instance
        :return:  The value of the state instance
        """
        state = state_instance.state
        start_index = state_instance.state_index
        end_index = start_index + state.size
        shape = state.shape
        return self.state_vector[start_index:end_index].reshape(shape)

    def get_port_value(self, port_instance):
        """
        Determine the value of the given port instance.

        If the value has not yet been calculated, it will be calculated before
        this method returns. If an algebraic loop is encountered during
        calculation, an ``AlgebraicLoopException`` will be raised.

        :param port_instance: The port instance for which the value shall be
            determined
        :return: The value of the port instance
        :raises AlgebraicLoopException: if an algebraic loop is encountered
            while evaluating the value of the signal
        """
        signal_instance = port_instance.signal
        signal = signal_instance.port
        start_index = signal_instance.signal_index
        end_index = start_index + signal.size

        if signal_instance in self.valid_signals:
            # That signal was already evaluated, so just return the value in
            # proper shape.
            return self._signal_vector[start_index:end_index]\
                .reshape(signal.shape)

        # The signal has not yet been evaluated, so we try to do that now
        if signal_instance in self.signal_evaluation_set:
            # The signal is currently being evaluated, but we got here again,
            # so there must be an algebraic loop.
            raise AlgebraicLoopException()

        # Start evaluation of the signal
        self.signal_evaluation_set.add(signal_instance)
        self.signal_evaluation_stack.append(signal_instance)

        # Perform evaluation
        block = signal_instance.block
        vector_wrapper = VectorWrapper(block, self)
        signal_value = signal.function(block, self.time, vector_wrapper)

        # Ensure that the signal has the correct shape
        signal_value = np.asarray(signal_value).reshape(signal.shape)
        # Assign the value to the signal_vector
        self._signal_vector[start_index:end_index] = signal_value.flatten()
        # Mark the signal as valid
        self.valid_signals.add(signal_instance)

        # End evaluation of the signal
        self.signal_evaluation_set.remove(signal_instance)
        self.signal_evaluation_stack.pop()

        # Return the value of the signal
        return signal_value.reshape(signal.shape)

    def get_state_derivative(self, state_instance):
        """
        Get the state derivative of the given state instance.

        :param state_instance: The state instance for which the derivative shall
            be determined
        :return: The state derivative
        :raises AlgebraicLoopException: if an algebraic loop is encountered
            while evaluating the derivative of the state instance
        """
        block = state_instance.block
        vector_wrapper = VectorWrapper(block, self)
        state = state_instance.state
        state_derivative = state.derivative_function(block,
                                                     self.time,
                                                     vector_wrapper)
        state_derivative = np.asarray(state_derivative).reshape(state.shape)
        return state_derivative


class VectorWrapper:
    """
    A helper class to pass accesses to states or signals on to the ``Evaluator``
    object.
    """
    def __init__(self, block, evaluator):
        self.block = block
        self.evaluator = evaluator

    def __getattr__(self, name):
        item = getattr(self.block, name)
        return self.evaluator.get_item_value(item)
