"""
Blocks for stiff body dynamics
"""
import numpy as np
from numpy.linalg import solve

from simtree.blocks import LeafBlock


class RigidBody6DOFFlatEarth(LeafBlock):
    """
    A block representing the motion of a rigid body in 6 degrees of freedom
    relative to a flat earth reference frame.

    The flat earth reference frame assumes the x-axis pointing to the north,
    the y-axis pointing to the east, and the z-axis pointing down.

    The body reference frame assumes the x-axis pointing to the front, the
    y-axis pointing to the right, and the z-axis pointing down.

    Both reference frames are right-handed systems.

    The block assumes a constant mass and constant moment of inertia.

    The block accepts the applied forces and moments in the body reference
    frame as input, in this order.

    It provides as output (in this order)
        - the velocity in the earth reference frame,
        - the position in the earth reference frame,
        - the coordinate transformation matrix from the body to the earth
          reference frame,
        - the velocity in the body reference frame, and
        - the angular rates in the body reference frame.
    """
    def __init__(self,
                 mass,
                 moment_of_inertia,
                 gravity=None,
                 initial_velocity_earth=None,
                 initial_position_earth=None,
                 initial_transformation=None,
                 initial_angular_rates_earth=None,
                 **kwargs):
        if gravity is None:
            gravity = np.c_[0, 0, 9.81]
        if initial_velocity_earth is None:
            initial_velocity_earth = np.zeros(3)
        if initial_position_earth is None:
            initial_position_earth = np.zeros(3)
        if initial_transformation is None:
            initial_transformation = np.eye(3, 3)
        if initial_angular_rates_earth is None:
            initial_angular_rates_earth = np.zeros(3)
        initial_condition = np.concatenate(
            (initial_velocity_earth,
             initial_position_earth,
             initial_transformation,
             initial_angular_rates_earth),
            axis=None
        )
        LeafBlock.__init__(self,
                           num_inputs=3+3,
                           num_states=3+3+9+3,
                           num_outputs=3+3+9+3+3,
                           feedthrough_inputs=[],
                           initial_condition=initial_condition,
                           **kwargs)
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.gravity = gravity

    def state_update_function(self, time, state, inputs):
        del time  # unused

        # Get the relevant inputs
        forces_body = inputs[0:3]
        moments_body = inputs[3:6]

        # Get the linear and angular accelerations in body frame
        lin_accel_body = forces_body / self.mass
        ang_accel_body = solve(self.moment_of_inertia, moments_body)

        # Get the relevant portions of the state
        vel_earth = state[0:3]
        trans_matrix = state[6:15].reshape(3, 3)
        omega_earth = state[15:18]

        # Transform the linear and angular acceleration into the earth frame
        lin_accel_earth = trans_matrix @ lin_accel_body + self.gravity
        ang_accel_earth = trans_matrix @ ang_accel_body

        # Determine the derivative of the transformation_matrix
        omega_skew_matrix = np.asarray([
            [0,               -omega_earth[2], +omega_earth[1]],
            [+omega_earth[2], 0,               -omega_earth[0]],
            [-omega_earth[1], +omega_earth[0], 0]
        ])
        trans_matrix_deriv = omega_skew_matrix @ trans_matrix

        # Return the state derivative
        return np.concatenate(
            (
                # Change of velocity in earth reference frame
                lin_accel_earth,
                # Change of position in earth reference frame
                vel_earth,
                # Change of coordinate transformation matrix
                trans_matrix_deriv,
                # Change of angular velocity in earth reference frame
                ang_accel_earth
            ),
            axis=None
        )

    @staticmethod
    def output_function(time, state, inputs):
        del time  # unused
        del inputs  # unused

        # Get the relevant portions of the state
        vel_earth = state[0:3]
        pos_earth = state[3:6]
        trans_matrix = state[6:15].reshape(3, 3)
        omega_earth = state[15:18]

        # Calculate velocity and angular velocity in body reference frame
        vel_body = solve(trans_matrix, vel_earth)
        omega_body = solve(trans_matrix, omega_earth)

        # Return all the outputs
        return np.concatenate(
            (
                # velocity in earth reference frame
                vel_earth,
                # position in earth reference frame
                pos_earth,
                # coordinate transformation matrix
                trans_matrix,
                # velocity in body reference frame
                vel_body,
                # angular rates in the body reference frame
                omega_body
            ),
            axis=None
        )


class DirectCosineToEuler(LeafBlock):
    """
    A block translating a direct cosine matrix to Euler angles.

    This block has a 3x3 direct cosine matrix (DCM) as input and
    provides the euler angles (psi, theta, phi, in radians) for yaw,
    pitch and roll as output.
    """
    # yaw: psi
    # pitch: theta
    # roll: phi
    def __init__(self, **kwargs):
        LeafBlock.__init__(self,
                           num_inputs=9,
                           num_outputs=3,
                           **kwargs)

    @staticmethod
    def output_function(time, inputs):
        del time  # unused
        dcm = np.asarray(inputs).reshape(3,3)
        psi = np.arctan2(dcm[1,0], dcm[0,0])
        theta = np.arctan2(-dcm[2,0], np.sqrt(dcm[0,0]**2 + dcm[1,0]**2))
        phi = np.arctan2(dcm[2,1], dcm[2,2])

        return np.r_[psi, theta, phi]
