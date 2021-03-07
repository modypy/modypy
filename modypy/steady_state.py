"""
Functions and classes for finding the steady state of a system.

To determine a steady state, set up a :class:`SteadyStateConfiguration` object
and pass it to :func:`find_steady_state`.
"""
from functools import partial
from itertools import accumulate, chain
from typing import Union

import numpy as np
import scipy.optimize as opt

from modypy.model import Evaluator, Port, System
from modypy.model.evaluation import DataProvider


class SteadyStateConfiguration:
    """Represents the configuration for the steady state determination

    Attributes:
        system
            The system for which a steady state analysis shall be carried out
        time
            The system time for which the steady state analysis shall be carried
            out.
        objective
            An objective to minimize. This is either `None` (default), a
            callable or a `Port`. If no objective is specified, any steady state
            that satisfies the other constraints may be returned.
        initial_condition
            The initial estimate for the states. The default is the initial
            condition of the system.
        initial_input
            The initial estimate for the inputs. The default is the initial
            values specified for the inputs in the system.
        input_bounds
            An array of shape (n,2), where n is the number of inputs for the
            system. Each entry ``input_bounds[k]`` is a tuple ``(lb, ub)``, with
            ``lb`` giving the lower and ``ub`` giving the upper bound for the
            respective input line. The initial value for all bounds is
            ``(-np.inf, np.inf)``, representing an unconstrained input line.
        state_bounds
            An array of shape (n,2), where n is the number of states for the
            system. The format is the same as for `input_bounds`.
        signal_bounds
            An array of shape (n,2), where n is the number of signals for the
            system. The format is the same as for `input_bounds`.
        steady_states
            An array of shape (n,), where n is the number of states for the
            system. The entry ``steady_states[k]`` is a boolean indicating
            whether the state ``k`` shall be steady, i.e. whether its derivative
            shall be zero. By default, all states are set to be steady.
        solver_options
            Dictionary with additional keyword options for the solver.
    """

    def __init__(self,
                 system: System,
                 time: float = 0,
                 objective: Union[callable, Port] = None):
        self.system = system
        self.time = time
        self.objective = objective
        # Set up the initial state estimates
        self.initial_condition = self.system.initial_condition
        # Set up the initial input estimates
        self.initial_input = self.system.initial_input
        # Set up the initial input bounds
        self.input_bounds = np.full(shape=(self.system.num_inputs, 2),
                                    fill_value=(-np.inf, np.inf))
        # Set up the initial state bounds
        self.state_bounds = np.full(shape=(self.system.num_states, 2),
                                    fill_value=(-np.inf, np.inf))
        # Set up the initial signal bounds
        # We initialize these to NaN so that we can skip the calculation
        # of unbounded signal values
        self.signal_bounds = np.full(shape=(self.system.num_signals, 2),
                                     fill_value=(np.nan, np.nan))
        # Flags indicating which states need to be steady
        # (by default, all states are steady states)
        self.steady_states = [True, ] * self.system.num_states
        # Set up the dictionary for solver options
        self.solver_options = dict()


def find_steady_state(config: SteadyStateConfiguration):
    """Run the steady-state determination

    Args:
        config: The configuration for the steady-state analysis

    Returns:
        An :class:`OptimizeResult <scipy.optimize.OptimizeResult>` object with
        additional fields.

        state: ndarray
            The state part of the solution
        inputs: ndarray
            The input part of the solution
        evaluator: modypy.model.evaluation.Evaluator
            An :class:`Evaluator <modypy.model.evaluation.Evaluator>` object,
            configured to evaluate the system at the determined steady-state
    """

    # Set up the initial estimate
    x0 = np.concatenate((config.initial_condition, config.initial_input))
    # Set up the bounds
    bounds = np.concatenate((config.state_bounds, config.input_bounds))

    # Set up the constraints
    constraints = list()

    if (~np.isnan(config.signal_bounds)).any():
        # Set up the signal constraint
        constraints.append(_SignalConstraint(config))

    if config.objective is not None:
        # We have an actual objective function, so we can use the steady-state
        # constraint as actual constraint.
        if any(config.steady_states):
            # Set up the state derivative constraint
            constraints.append(_StateDerivativeConstraint(config))

        # Translate the objective function
        if isinstance(config.objective, Port):
            objective_function = partial(_port_objective_function, config)
        elif callable(config.objective):
            objective_function = partial(_general_objective_function, config)
        else:
            raise ValueError("The objective function must be either a Port or "
                             "a callable")
    elif any(config.steady_states):
        # No objective function was specified, but we can use the steady-state
        # constraint function. The value of this function is intended to be
        # zero, so the minimum value of its square is zero.
        steady_state_constraint = _StateDerivativeConstraint(config)
        objective_function = steady_state_constraint.evaluate_squared
    else:
        # We have neither an objective function to minimize nor do we have any
        # state that is intended to be steady. We cannot use the signal
        # constraints to minimize, as these may specify ranges instead of a
        # target value. We cannot do anything about this.
        raise ValueError("Either an objective function or at least one steady "
                         "state is required")

    result = opt.minimize(fun=objective_function,
                          x0=x0,
                          method="trust-constr",
                          bounds=bounds,
                          constraints=constraints,
                          options=config.solver_options)

    result.config = config
    result.state = result.x[:config.system.num_states]
    result.inputs = result.x[config.system.num_states:]
    result.evaluator = Evaluator(time=config.time,
                                 system=config.system,
                                 state=result.state,
                                 inputs=result.inputs)

    return result


class _StateDerivativeConstraint(opt.NonlinearConstraint):
    """Represents the steady-state constraints on the state derivatives"""

    def __init__(self, config: SteadyStateConfiguration):
        self.config = config
        # Steady-state constraints can be defined on the level of individual
        # state components. To optimize evaluation, we only evaluate derivatives
        # of those states that have at least one of their components
        # constrained.
        self.constrained_states = [
            state for state in self.config.system.states
            if any(self.config.steady_states[state.state_slice])]

        # We will build a vector of the constrained derivatives, and for that
        # we assign offsets for the states in that vector
        self.state_offsets = \
            [0, ] + list(accumulate(state.size
                                    for state in self.constrained_states))

        num_states = self.state_offsets[-1]

        # Now we set up the bounds for each of these
        ub = np.full(num_states, np.inf)
        ub[self.config.steady_states] = 0
        lb = -ub

        opt.NonlinearConstraint.__init__(self,
                                         fun=self.evaluate,
                                         lb=lb,
                                         ub=ub)

    def evaluate(self, x):
        """Determine the value of the derivatives of the vector of constrained
        states"""

        state = x[:self.config.system.num_states]
        inputs = x[self.config.system.num_states:]
        evaluator = Evaluator(time=self.config.time,
                              system=self.config.system,
                              state=state,
                              inputs=inputs)
        num_states = self.state_offsets[-1]
        derivative_vector = np.empty(num_states)
        for state, state_offset in zip(self.constrained_states,
                                       self.state_offsets):
            derivative_vector[state_offset:state_offset + state.size] = \
                evaluator.get_state_derivative(state)
        return derivative_vector

    def evaluate_squared(self, x):
        """Determine the 2-norm of the derivatives vector of constrained
        states"""

        return np.sum(np.square(self.evaluate(x)))


class _SignalConstraint(opt.NonlinearConstraint):
    """Represents the constraints on signals"""

    def __init__(self, config: SteadyStateConfiguration):
        self.config = config

        # Signal constraints can be defined on individual signal components.
        # To optimize evaluation, we only evaluate those signals that have at
        # least one of their components constrained.
        self.constrained_signals = [
            signal for signal in self.config.system.signals
            if (~np.isnan(self.config.signal_bounds[signal.signal_slice])).any()
        ]

        # We will build a vector of the constrained signals, and for that
        # we assign offsets for the signals in that vector
        self.signal_offsets = \
            [0, ] + list(accumulate(signal.size
                                    for signal in self.constrained_signals))

        # Now we set up the bounds for each of these
        indices = list(
            chain.from_iterable(signal.signal_range
                                for signal in self.constrained_signals))
        bounds = self.config.signal_bounds[indices]
        # Ensure that no nans are in the bounds
        np.nan_to_num(bounds[:, 0],
                      posinf=np.inf,
                      neginf=-np.inf,
                      nan=-np.inf,
                      copy=False)
        np.nan_to_num(bounds[:, 1],
                      posinf=np.inf,
                      neginf=-np.inf,
                      nan=np.inf,
                      copy=False)

        opt.NonlinearConstraint.__init__(self,
                                         fun=self.evaluate,
                                         lb=bounds[:, 0],
                                         ub=bounds[:, 1])

    def evaluate(self, x):
        """Calculate the vector of constrained signals"""

        state = x[:self.config.system.num_states]
        inputs = x[self.config.system.num_states:]
        evaluator = Evaluator(time=self.config.time,
                              system=self.config.system,
                              state=state,
                              inputs=inputs)
        num_signals = self.signal_offsets[-1]
        signal_vector = np.empty(num_signals)
        for signal, signal_offset in zip(self.constrained_signals,
                                         self.signal_offsets):
            signal_vector[signal_offset:signal_offset + signal.size] = \
                evaluator.get_port_value(signal)
        return signal_vector


def _port_objective_function(config: SteadyStateConfiguration, x):
    """Implementation of the objective function for ports

    Args:
        config: The configuration for the steady-state determination
        x: The vector of the current values of states and input

    Returns:
        The current value of the objective port
    """

    state = x[:config.system.num_states]
    inputs = x[config.system.num_states:]
    evaluator = Evaluator(time=config.time,
                          system=config.system,
                          state=state,
                          inputs=inputs)
    return evaluator.get_port_value(config.objective)


def _general_objective_function(config: SteadyStateConfiguration, x):
    """Implementation of the general objective function

    This calls the objective function with a `DataProvider` as single parameter.

    Args:
        config: The configuration for the steady-state determination
        x: The vector of the current values of states and input

    Returns:
        The current value of the objective function
    """

    state = x[:config.system.num_states]
    inputs = x[config.system.num_states:]
    evaluator = Evaluator(time=config.time,
                          system=config.system,
                          state=state,
                          inputs=inputs)
    data_provider = DataProvider(evaluator=evaluator,
                                 time=config.time)
    return config.objective(data_provider)
