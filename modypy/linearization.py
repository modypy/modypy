"""
Provides functions to determine the jacobi matrix for linearizing the system
around a given state with specified inputs.
"""
import numpy as np
from scipy.optimize import root
from scipy.misc import central_diff_weights

from modypy.model.evaluation import Evaluator


def system_jacobian(system,
                    time,
                    x_ref,
                    u_ref,
                    step=0.1,
                    order=3,
                    single_matrix=False):
    """Numerically determine the jacobian of the system at the given state and
    input setting.

    The function uses polygonal interpolation of the given order on the
    components of the system derivative and output function, chosing
    interpolation points in distance `step` from the given state and input.

    This can be used in conjunction with `find_steady_state` to determine
    an LTI approximating the behaviour around the steady state.

    NOTE: This function currently does not honor clocks.

    Args:
      system: The system to be analysed
      time: The time at which the system shall be considered
      x_ref: The state vector for which the jacobian shall be determined
      u_ref: The input vector for which the jacobian shall be determined
      step: The step size for numerical differentiation (Default value = 0.1)
      order: The order of the interpolating polynomial (Default value = 3)
      single_matrix: Flag indicating whether a single matrix shall be
        returned. The default is `False`.

    Returns:
      The jacobian, if ``single_matrix`` is ``True``.

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
    """

    Args:
      system:
      time:
      x_ref:

    Returns:

    """
    states = x_ref[:system.num_states]
    inputs = x_ref[system.num_states:]

    evaluator = Evaluator(time=time, system=system, state=states, inputs=inputs)

    return np.concatenate((evaluator.state_derivative, evaluator.outputs))
