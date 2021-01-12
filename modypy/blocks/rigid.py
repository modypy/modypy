"""
Blocks for stiff body dynamics
"""
import numpy as np
import numpy.linalg as linalg

from modypy.model import Block, Port, Signal, SignalState


class RigidBody6DOFFlatEarth(Block):
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
                 owner,
                 mass,
                 moment_of_inertia,
                 initial_velocity_earth=None,
                 initial_position_earth=None,
                 initial_transformation=None,
                 initial_angular_rates_earth=None):
        Block.__init__(self, owner)
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia

        self.forces_body = Port(self, shape=3)
        self.moments_body = Port(self, shape=3)

        if initial_transformation is None:
            initial_transformation = np.eye(3)

        self.velocity_earth = SignalState(self,
                                          shape=3,
                                          derivative_function=self.velocity_earth_dot,
                                          initial_condition=initial_velocity_earth)
        self.position_earth = SignalState(self,
                                          shape=3,
                                          derivative_function=self.position_earth_dot,
                                          initial_condition=initial_position_earth)
        self.omega_earth = SignalState(self,
                                       shape=3,
                                       derivative_function=self.omega_earth_dot,
                                       initial_condition=initial_angular_rates_earth)
        self.dcm = SignalState(self,
                               shape=(3, 3),
                               derivative_function=self.dcm_dot,
                               initial_condition=initial_transformation)

        self.velocity_body = Signal(self,
                                    shape=3,
                                    value=self.velocity_body_output)
        self.omega_body = Signal(self,
                                 shape=3,
                                 value=self.omega_body_output)

    def velocity_earth_dot(self, data):
        """Calculates the acceleration in the earth reference frame"""
        forces_earth = data.states[self.dcm] @ data.signals[self.forces_body]
        accel_earth = forces_earth / self.mass
        return accel_earth

    def position_earth_dot(self, data):
        """Calculates the velocity in the earth reference frame"""
        return data.states[self.velocity_earth]

    def omega_earth_dot(self, data):
        """Calculate the angular acceleration in the earth reference frame"""
        moments_earth = data.states[self.dcm] @ data.signals[self.moments_body]
        ang_accel_earth = linalg.solve(self.moment_of_inertia, moments_earth)
        return ang_accel_earth

    def dcm_dot(self, data):
        """Calculate the derivative of the direct cosine matrix"""
        omega_earth = data.states[self.omega_earth]
        skew_sym_matrix = np.array([
            [0, -omega_earth[2], omega_earth[1]],
            [omega_earth[2], 0, -omega_earth[0]],
            [-omega_earth[1], omega_earth[0], 0]
        ])
        return skew_sym_matrix @ data.states[self.dcm]

    def velocity_body_output(self, data):
        """Calculate the velocity in the body reference frame"""
        dcm = data.states[self.dcm]
        velocity_earth = data.states[self.velocity_earth]
        return dcm.T @ velocity_earth

    def omega_body_output(self, data):
        """Calculate the angular velocity in the body reference frame"""
        dcm = data.states[self.dcm]
        omega_earth = data.states[self.omega_earth]
        return dcm.T @ omega_earth


class DirectCosineToEuler(Block):
    """
    A block translating a direct cosine matrix to Euler angles.

    This block has a 3x3 direct cosine matrix (DCM) as input and
    provides the euler angles (psi, theta, phi, in radians) for yaw,
    pitch and roll as output.
    """
    # yaw: psi
    # pitch: theta
    # roll: phi
    def __init__(self, parent):
        Block.__init__(self, parent)

        self.dcm = Port(self, shape=(3, 3))
        self.yaw = Signal(self, shape=1, value=self.calculate_yaw)
        self.pitch = Signal(self, shape=1, value=self.calculate_pitch)
        self.roll = Signal(self, shape=1, value=self.calculate_roll)

    def calculate_yaw(self, data):
        """Calculate the yaw angle for the given direct cosine matrix"""
        dcm = data.signals[self.dcm]
        return np.arctan2(dcm[1, 0], dcm[0, 0])

    def calculate_pitch(self, data):
        """Calculate the pitch angle for the given direct cosine matrix"""
        dcm = data.signals[self.dcm]
        return np.arctan2(-dcm[2, 0], np.sqrt(dcm[0, 0]**2 + dcm[1, 0]**2))

    def calculate_roll(self, data):
        """Calculate the roll angle for the given direct cosine matrix"""
        dcm = data.signals[self.dcm]
        return np.arctan2(dcm[2, 1], dcm[2, 2])
