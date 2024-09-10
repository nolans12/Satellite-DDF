from import_libraries import *


class target:
    def __init__(
        self,
        name,
        targetID,
        coords,
        heading,
        speed,
        color,
        uncertainty=np.array([0, 0, 0, 0, 0, 0]),
        climbrate=0,
        changeAoA=False,
    ):
        """Target class that moves linearly around the earth with constant angular velocity.

        Args:
            name (str): Name of the target.
            targetID (int): ID of the target.
            coords (np.array): Initial position [latitude, longitude, altitude].
            heading (float): Heading direction in degrees.
            speed (float): Speed in km/h.
            color (str): Color representation of the target.
            uncertainty (np.array): Uncertainty in the initial state. Defaults to np.array([0, 0, 0, 0, 0, 0]).
            climbrate (float): Climbing rate in km/h. Defaults to 0.
            changeAoA (bool): Whether the target should change Angle of Attack. Defaults to False.
        """

        # Initialize the target's parameters
        self.targetID = targetID
        self.name = name
        self.color = color
        self.time = 0

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

        # Set the covariance (uncertainty in the initial guess in spherical coordinates)
        self.initialCovariance = np.array(
            [
                [uncertainty[0] ** 2, 0, 0, 0, 0, 0],
                [0, uncertainty[1] ** 2, 0, 0, 0, 0],
                [0, 0, np.deg2rad(uncertainty[2]) ** 2, 0, 0, 0],
                [0, 0, 0, np.deg2rad(uncertainty[3]) ** 2, 0, 0],
                [0, 0, 0, 0, np.deg2rad(uncertainty[4]) ** 2, 0],
                [0, 0, 0, 0, 0, np.deg2rad(uncertainty[5]) ** 2],
            ]
        )

        # Define the initial state vector X by sampling from the initialState and initialCovariance
        self.X = np.random.multivariate_normal(
            self.initialState, self.initialCovariance
        )

        self.hist = defaultdict(
            dict
        )  # Contains time, xyz, and velocity history in ECI [x xdot y ydot z zdot]

    def propagate(self, time_step, time, initialState=None):
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

        # if initialState is not None: TODO: what is this for?
        #     return np.array([x, y, z, vx, vy, vz])
