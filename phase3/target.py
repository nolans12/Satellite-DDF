import numpy as np
from astropy import units as u
from numpy import typing as npt

from common import dataclassframe
from phase3 import collection


class Target:
    def __init__(
        self,
        target_id: str,
        coords: npt.NDArray,
        heading: float,
        speed: float,
        color: str,
        climb_rate: float = 0,
        change_AoA: bool = False,
    ):
        """Target class that moves linearly around the earth with constant angular velocity.

        Args:
            name: Name of the target.
            target_id: ID of the target.
            coords: Initial position [latitude, longitude, altitude].
            heading: Heading direction in degrees.
            speed: Speed in km/h.
            uncertainty: Uncertainty in the coordinates, heading, and speed of the target. [lat (deg), lon (deg), alt (km), heading (deg), speed (km/min)]
            color: Color representation of the target.
            climb_rate: Climbing rate in km/h. Defaults to 0.
            change_AoA: Whether the target should change Angle of Attack. Defaults to False.
        """

        # Initialize the target's parameters
        self.target_id = target_id
        self.color = color

        # Convert the spherical coordinates and heading into a state vector
        # state = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        range = coords[2] + 6378  # range from center of earth in km
        rangeRate = climb_rate  # constant altitude [km/min]
        self.change_AoA = (
            change_AoA  # True if target should change Angle of Attack TODO: Implement
        )

        elevation = np.deg2rad(coords[0])  # [rad] latitude where 0 is equator
        azimuth = np.deg2rad(coords[1])  # [rad] longitude where 0 prime meridian

        # Angular rates are speed in direction of heading TODO: Check this
        elevation_rate = speed / range * np.cos(np.deg2rad(heading))  # [rad/min]
        azimuth_rate = speed / range * np.sin(np.deg2rad(heading))  # [rad/min]

        # Initialize the state (guess of target position and velocity in spherical coordinates)
        self.initial_state = np.array(
            [range, rangeRate, elevation, elevation_rate, azimuth, azimuth_rate]
        )

        # Define the initial state vector X
        self.X = self.initial_state
        self.pos = np.array([0, 0, 0])  # Target's position in ECI [x y z]
        self.vel = np.array([0, 0, 0])

        self._state_hist = dataclassframe.DataClassFrame(clz=collection.State)

    def insert_propagation(self, state: collection.State) -> None:
        """
        Insert a state into the state history.

        Args:
            state: The state to insert.
        """
        # Convert position and velocity to range, elevation, and azimuth
        # TODO: Test me
        range = np.linalg.norm([state.x, state.y, state.z])
        rangeRate = (
            np.dot([state.x, state.y, state.z], [state.vx, state.vy, state.vz]) / range
        )
        elevation = np.arcsin(state.z / range)
        elevationRate = (
            state.z * state.vz
            + range
            * np.cos(elevation)
            * np.dot([state.x, state.y], [state.vx, state.vy])
        ) / range
        azimuth = np.arctan2(state.y, state.x)
        azimuthRate = (state.y * state.vy - state.x * state.vx) / (
            range * np.cos(elevation)
        )

        self.X = np.array(
            [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        )

        self.pos = np.array([state.x, state.y, state.z])
        self.vel = np.array([state.vx, state.vy, state.vz])

        self._state_hist.append(state)

    def propagate(
        self,
        time_step: u.Quantity[u.minute],
        time: float,
        initial_state: npt.NDArray | None = None,
    ) -> None:
        """Linearly propagate target state in spherical coordinates then transform back to ECI.

        Args:
            time_step: Time step for propagation.
            time: Current time.
        """

        # Determine the time step
        dt = time_step.value

        # Determine the current state of target:
        # X = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        X = initial_state if initial_state is not None else self.X

        # Define the state transition matrix A
        A = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # Propagate the current state using the state transition matrix
        x_dot = np.dot(A, X)
        X = X + x_dot * dt

        # Save the updated spherical state
        self.X = X

        # Extract state components for reading simplicity
        range = X[0]
        rangeRate = X[1]
        elevation = X[2]
        elevationRate = X[3]
        azimuth = X[4]
        azimuthRate = X[5]

        # Convert current spherical state to Cartesian coordinate frame
        x = range * np.cos(elevation) * np.cos(azimuth)
        y = range * np.cos(elevation) * np.sin(azimuth)
        z = range * np.sin(elevation)

        # Approximate velocities conversion (simplified version)
        # TODO: Implement more accurate conversion? Do we need it?
        vx = (
            rangeRate * np.cos(elevation) * np.cos(azimuth)
            - range * elevationRate * np.sin(elevation) * np.cos(azimuth)
            - range * azimuthRate * np.cos(elevation) * np.sin(azimuth)
        )

        vy = (
            rangeRate * np.cos(elevation) * np.sin(azimuth)
            - range * elevationRate * np.sin(elevation) * np.sin(azimuth)
            + range * azimuthRate * np.cos(elevation) * np.cos(azimuth)
        )

        vz = rangeRate * np.sin(elevation) + range * elevationRate * np.cos(elevation)

        # Save target's position and velocity in ECI coordinates for EKF error calculation
        self.pos = np.array([x, y, z])
        self.vel = np.array([vx, vy, vz])
        self._state_hist.append(
            collection.State(
                time=time,
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
            )
        )

    def get_state(self, time: float) -> collection.State:
        """
        Get the state of the target at a specified time.

        Args:
            time: The time to get the state at.

        Returns:
            The state of the target at the specified time.
        """
        df = self._state_hist.loc[self._state_hist['time'] == time]
        return self._state_hist.to_dataclasses(df)[0]
