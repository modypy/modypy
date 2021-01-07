"""
Provides functions to determine the steady state of a system and the jacobi
matrix for linearizing the system around a steady state.
"""
import numpy as np
from scipy.optimize import root
from scipy.misc import central_diff_weights

from modypy.model.evaluation import Evaluator


def find_steady_state(system,
                      time,
                      method="lm",
                      solver_options=None,
                      **kwargs):
    """
    Find the constrained steady state of a system.

    A system is said to be in a *steady state* if its state does not change over
    time. As there maybe multiple steady states, additional constraints may be
    required.

    These constraints are expressed by enforcing the value of specified signals
    to be zero. Inputs can be specified by using the ``InputSignal`` class.

    The search begins at the initial values of signals and states.

    Note that for time-dependent systems, ``find_steady_state`` can only identify
    the steady state at a specific time.

    This function uses ``scipy.optimize.root`` for finding the root of the
    constraint functions.

    If the search is successful, the values of all input signals in the system
    will be set to the respective input value identified for the steady state.

    NOTE: This function currently does not honor clocks.

    :param system: The system to analyze.
    :param time: The time at which to determine the steady state. Default: 0
    :param method: The solver method to use. Refer to the documentation for
        `scipy.optimize.root` for more information on the available methods.
    :param solver_options: The options to pass to the solver.
    :return: sol, x0 -
        sol: An `scipy.optimize.OptimizeResult` object
        x0: The state vector at which the steady state occurs
    """

    if system.num_inputs > system.num_outputs:
        raise ValueError(
            "The system must have at least as many constraints as inputs")

    if system.num_states + system.num_inputs == 0:
        raise ValueError(
            "Cannot find steady-state in system without states and inputs")

    initial_value = np.concatenate((system.initial_condition,
                                    system.initial_input))

    sol = root(fun=(lambda x: _system_function(system, time, x)),
               x0=initial_value,
               method=method,
               options=solver_options,
               **kwargs)

    states = sol.x[:system.num_states]
    inputs = sol.x[system.num_states:]

    return sol, states, inputs


def system_jacobian(system,
                    time,
                    x_ref,
                    u_ref,
                    step=0.1,
                    order=3,
                    single_matrix=False):
    """
    Numerically determine the jacobian of the system at the given state and
    input setting.

    The function uses polygonal interpolation of the given order on the
    components of the system derivative and output function, chosing
    interpolation points in distance `step` from the given state and input.

    This can be used in conjunction with `find_steady_state` to determine
    an LTI approximating the behaviour around the steady state.

    NOTE: This function currently does not honor clocks.

    :param system: The system to be analysed
    :param time: The time at which the system shall be considered
    :param x_ref: The state vector for which the jacobian shall be determined
    :param u_ref: The input vector for which the jacobian shall be determined
    :param step: The step size for numerical differentiation
    :param order: The order of the interpolating polynomial
    :param single_matrix: Flag indicating whether a single matrix shall be
        returned. The default is `False`.
    :return: The jacobian, if ``single_matrix`` is ``True``.
    :return: ``system_matrix``, ``input_matrix``, ``output_matrix``,
        ``feed_through_matrix``, representing the LTI system at the given state
        and input.
    """

    if system.num_states + system.num_inputs == 0:
        raise ValueError("Cannot linearize system without states and inputs")
    if order == 3:
        weights = np.array([-1, 0, 1]) / 2.0
    elif order == 5:
        weights = np.array([1, -8, 0, 8, -1]) / 12.0
    elif order == 7:
        weights = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
    elif order == 9:
        weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
    else:
        weights = central_diff_weights(order, 1)

    num_invars = system.num_states + system.num_inputs
    num_outvars = system.num_states + system.num_outputs
    half_offset = order >> 1

    jac = np.zeros((num_outvars, num_invars))
    x_ref0 = np.concatenate((x_ref, u_ref), axis=None)

    for var_ind in range(num_invars):
        x_step = step * np.eye(N=1, M=num_invars, k=var_ind).flatten()
        for k in range(order):
            x_k = x_ref0 + (k - half_offset) * x_step
            y_k = _system_function(system, time, x_k)
            jac[:, var_ind] += weights[k] * y_k
        jac[:, var_ind] /= step

    if single_matrix:
        return jac
    return jac[:system.num_states, :system.num_states], \
        jac[:system.num_states, system.num_states:], \
        jac[system.num_states:, :system.num_states], \
        jac[system.num_states:, system.num_states:]


def _system_function(system, time, x_ref):
    states = x_ref[:system.num_states]
    inputs = x_ref[system.num_states:]

    evaluator = Evaluator(time=time, system=system, state=states, inputs=inputs)

    return np.concatenate((evaluator.state_derivative, evaluator.outputs))
