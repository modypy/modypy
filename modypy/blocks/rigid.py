"""
Blocks for stiff body dynamics
"""
import numpy as np
from modypy.model import Block, Port, signal_method, State
from numpy import linalg


class RigidBody6DOFFlatEarth(Block):
    """A block representing the motion of a rigid body in 6 degrees of freedom
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

    def __init__(
        self,
        owner,
        mass,
        moment_of_inertia,
        initial_velocity_earth=None,
        initial_position_earth=None,
        initial_transformation=None,
        initial_angular_rates_earth=None,
    ):
        Block.__init__(self, owner)
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia

        self.forces_body = Port(shape=3)
        self.moments_body = Port(shape=3)

        if initial_transformation is None:
            initial_transformation = np.eye(3)

        self.velocity_earth = State(
            self,
            shape=3,
            derivative_function=self.velocity_earth_dot,
            initial_condition=initial_velocity_earth,
        )
        self.position_earth = State(
            self,
            shape=3,
            derivative_function=self.position_earth_dot,
            initial_condition=initial_position_earth,
        )
        self.omega_earth = State(
            self,
            shape=3,
            derivative_function=self.omega_earth_dot,
            initial_condition=initial_angular_rates_earth,
        )
        self.dcm = State(
            self,
            shape=(3, 3),
            derivative_function=self.dcm_dot,
            initial_condition=initial_transformation,
        )

    def velocity_earth_dot(self, data):
        """Calculates the acceleration in the earth reference frame"""
        forces_earth = self.dcm(data) @ self.forces_body(data)
        accel_earth = forces_earth / self.mass
        return accel_earth

    def position_earth_dot(self, data):
        """Calculates the velocity in the earth reference frame"""
        return self.velocity_earth(data)

    def omega_earth_dot(self, data):
        """Calculate the angular acceleration in the earth reference frame"""
        moments_earth = self.dcm(data) @ self.moments_body(data)
        ang_accel_earth = linalg.solve(self.moment_of_inertia, moments_earth)
        return ang_accel_earth

    def dcm_dot(self, data):
        """Calculate the derivative of the direct cosine matrix"""
        omega_earth = self.omega_earth(data)
        skew_sym_matrix = np.array(
            [
                [0, -omega_earth[2], omega_earth[1]],
                [omega_earth[2], 0, -omega_earth[0]],
                [-omega_earth[1], omega_earth[0], 0],
            ]
        )
        return skew_sym_matrix @ self.dcm(data)

    # pylint does not recognize the modifications to the signal_method decorator
    # pylint: disable=no-value-for-parameter
    @signal_method(shape=(3, 3))
    def dcm_inverse(self, data):
        """The inverse of the direct cosine matrix"""
        return np.swapaxes(self.dcm(data), 0, 1)

    # pylint does not recognize the modifications to the signal_method decorator
    # pylint: disable=no-value-for-parameter
    @signal_method(shape=3)
    def velocity_body(self, data):
        """Calculate the velocity in the body reference frame"""

        return np.einsum(
            "ij...,j...->i...",
            self.dcm_inverse(data),
            self.velocity_earth(data),
        )

    # pylint does not recognize the modifications to the signal_method decorator
    # pylint: disable=no-value-for-parameter
    @signal_method(shape=3)
    def omega_body(self, data):
        """Calculate the angular velocity in the body reference frame"""

        return np.einsum(
            "ij...,j...->i...", self.dcm_inverse(data), self.omega_earth(data)
        )


class DirectCosineToEuler(Block):
    """A block translating a direct cosine matrix to Euler angles.

    This block has a 3x3 direct cosine matrix (DCM) as input and
    provides the euler angles (psi, theta, phi, in radians) for yaw,
    pitch and roll as output.
    """

    # yaw: psi
    # pitch: theta
    # roll: phi
    def __init__(self, parent):
        Block.__init__(self, parent)

        self.dcm = Port(shape=(3, 3))

    @signal_method
    def yaw(self, data):
        """Calculate the yaw angle for the given direct cosine matrix"""
        dcm = self.dcm(data)
        return np.arctan2(dcm[1, 0], dcm[0, 0])

    @signal_method
    def pitch(self, data):
        """Calculate the pitch angle for the given direct cosine matrix"""
        dcm = self.dcm(data)
        return np.arctan2(-dcm[2, 0], np.sqrt(dcm[0, 0] ** 2 + dcm[1, 0] ** 2))

    @signal_method
    def roll(self, data):
        """Calculate the roll angle for the given direct cosine matrix"""
        dcm = self.dcm(data)
        return np.arctan2(dcm[2, 1], dcm[2, 2])
