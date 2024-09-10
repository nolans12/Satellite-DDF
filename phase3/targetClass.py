from import_libraries import *
from numpy import typing as npt


class Target:
    def __init__(
        self,
        name: str,
        targetID: int,
        coords: npt.NDArray,
        heading: float,
        speed: float,
        color: str,
        uncertainty: npt.NDArray = np.array([0, 0, 0, 0, 0]),
        climbrate: float = 0,
        changeAoA: bool = False,
        tqReq: int = 0,
    ):
        """Target class that moves linearly around the earth with constant angular velocity.

        Args:
            name: Name of the target.
            targetID: ID of the target.
            coords: Initial position [latitude, longitude, altitude].
            heading: Heading direction in degrees.
            speed: Speed in km/h.
            uncertainty: Uncertainty in the coordinates, heading, and speed of the target. [lat (deg), lon (deg), alt (km), heading (deg), speed (km/min)]
            color: Color representation of the target.
            climbrate: Climbing rate in km/h. Defaults to 0.
            changeAoA: Whether the target should change Angle of Attack. Defaults to False.
        """

        # Initialize the target's parameters
        self.tqReq = tqReq
        self.targetID = targetID
        self.name = name
        self.color = color
        self.time = 0

        # Take the initial coords, heading, speed and add the uncertainty to them
        self.initialParams = np.array([coords[0], coords[1], coords[2], heading, speed])
        self.initialCovMat = np.diag(uncertainty**2)

        # Now sample from the uncertainty matrix to get the initial parameters
        initialParameters = np.random.multivariate_normal(
            self.initialParams, self.initialCovMat
        )

        # Now back out the original parameters
        coords = initialParameters[0:3]
        heading = initialParameters[3]
        speed = initialParameters[4]

        # Convert the spherical coordinates and heading into a state vector
        # state = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        range = coords[2] + 6378  # range from center of earth in km
        rangeRate = climbrate  # constant altitude [km/min]
        self.changeAoA = (
            changeAoA  # True if target should change Angle of Attack TODO: Implement
        )

        elevation = np.deg2rad(coords[0])  # [rad] latitude where 0 is equator
        azimuth = np.deg2rad(coords[1])  # [rad] longitude where 0 prime meridian

        # Angular rates are speed in direction of heading TODO: Check this
        elevationRate = speed / range * np.cos(np.deg2rad(heading))  # [rad/min]
        azimuthRate = speed / range * np.sin(np.deg2rad(heading))  # [rad/min]

        # Initialize the state (guess of target position and velocity in spherical coordinates)
        self.initialState = np.array(
            [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        )

        # Define the initial state vector X
        self.X = self.initialState

        self.hist = defaultdict(
            dict
        )  # Contains time, xyz, and velocity history in ECI [x xdot y ydot z zdot]

    def propagate(
        self, time_step: float, time: float, initialState: npt.NDArray | None = None
    ) -> None:
        """Linearly propagate target state in spherical coordinates then transform back to ECI.

        Args:
            time_step (float or np.float64): Time step for propagation.
            time (float): Current time.
            initialState (np.array, optional): Initial guess for the state. Defaults to None.

        Returns:
            np.array: Updated state guess if initialState is provided. # TODO: update history instead
        """
        # Determine the time step
        if isinstance(time_step, (float, np.float64)):
            dt = time_step
        else:
            dt = time_step.value

        # Determine the current state of target:
        # X = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        X = initialState if initialState is not None else self.X

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

        # Approximate velocities conversion (simplified version), TODO: Implement more accurate conversion? Do we need it?
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

        # Update the target's position and velocity in ECI coordinates for EKF error calculation
        self.pos = np.array([x, y, z])
        self.vel = np.array([vx, vy, vz])

        # Update the target's dictionary with the current state for plotting
        self.hist[self.time] = np.array(
            [
                self.pos[0],
                self.vel[0],
                self.pos[1],
                self.vel[1],
                self.pos[2],
                self.vel[2],
            ]
        )
