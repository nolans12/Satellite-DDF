from import_libraries import *


class sensor:
    """
    Class representing a bearings only sensor.
    """

    def __init__(self, fov, bearingsError, name, detectChance=0, resolution=720):
        """
        Initialize a sensor instance.

        Args:
            fov (float): Field of view of the sensor [deg].
            bearingsError (float): Error associated with bearings [deg].
            name (str): Name of the sensor.
            detectChance (float, optional): Detection probability chance (default is 0).
            resolution (int, optional): Resolution of the sensor (default is 720).
        """
        self.fov = fov
        self.bearingsError = bearingsError
        self.detectChance = detectChance
        self.name = name
        self.resolution = resolution
        self.projBox = np.array([0, 0, 0])

    def get_measurement(self, sat, targ):
        """
        Get sensor measurement of a target if visible within sensor's field of view.

        Args:
            sat (Satellite): Satellite object.
            targ (Target): Target object.

        Returns:
            np.ndarray: Sensor measurement if target is visible, else returns 0.
        """

        # Get the current projection box of the sensor
        self.visible_projection(sat)
        if self.inFOV(sat, targ):  # check if target is in the field of view
            detect = np.random.uniform(
                0, 1
            )  # generate random number to determine detection
            if detect < self.detectChance:  # check if target is detected
                return None
            else:
                return self.sensor_model(
                    sat, targ
                )  # return the noisy sensor measurement for target
        else:
            return None

    def sensor_model(self, sat, targ):
        """
        Simulate sensor measurement with added error.

        Args:
            sat (Satellite): Satellite object.
            targ (Target): Target object.

        Returns:
            np.ndarray: Simulated sensor measurement (in-track and cross-track angles).
        """
        # Get True Relative Target Position in terms of bearings angles
        in_track_truth, cross_track_truth = self.convert_to_bearings(sat, targ.pos)

        # Add Sensor Error in terms of Gaussian Noise [deg]
        in_track_meas = in_track_truth + np.random.normal(0, self.bearingsError[0])
        cross_track_meas = cross_track_truth + np.random.normal(
            0, self.bearingsError[1]
        )

        # Return the noisy sensor measurement
        return np.array([in_track_meas, cross_track_meas])

    def convert_to_bearings(self, sat, meas_ECI):
        """
        Convert satellite and target ECI positions to bearings angles.

        Args:
            sat (Satellite): Satellite object.
            meas_ECI (np.ndarray): Target position vector in ECI coordinates.

        Returns:
            Tuple[float, float]: In-track and cross-track angles between satellite and target.
        """
        sensor_measurement = self.transform_eci_to_bearings(
            sat.orbit.r.value, sat.orbit.v.value, meas_ECI
        )

        return float(sensor_measurement[0]), float(sensor_measurement[1])

    def transform_eci_to_bearings(self, r_value, v_value, meas_ECI):
        """
        Transform ECI coordinates to bearings angles.

        Args:
            r_value (jnp.ndarray): Current satellite position vector.
            v_value (jnp.ndarray): Current satellite velocity vector.
            meas_ECI (jnp.ndarray): Target position vector in ECI coordinates.

        Returns:
            jnp.ndarray: In-track and cross-track angles between satellite and target.
        """
        # Find Sensor Frame
        rVec = self.normalize(jnp.array(r_value))  # find unit radial vector
        vVec = self.normalize(jnp.array(v_value))  # find unit in-track vector
        wVec = self.normalize(
            jnp.cross(r_value, v_value)
        )  # find unit cross-track vector

        # Create transformation matrix T
        T = jnp.stack([vVec.T, wVec.T, rVec.T])

        # Rotate satellite and target into sensor frame
        sat_pos = jnp.array(r_value)  # get satellite position
        x_sat_sens, y_sat_sens, z_sat_sens = (
            T @ sat_pos
        )  # rotate satellite into sensor frame

        meas_ECI_sym = jnp.array(meas_ECI)  # get noisy measurement
        x_targ_sens, y_targ_sens, z_targ_sens = (
            T @ meas_ECI_sym
        )  # rotate measurement into sensor frame

        # Get the relative bearings from sensor to target
        satVec = jnp.array(
            [x_sat_sens, y_sat_sens, z_sat_sens]
        )  # get satellite vector in sensor frame

        # Get the In-Track and Cross-Track angles
        targVec_inTrack = satVec - jnp.array(
            [x_targ_sens, 0, z_targ_sens]
        )  # get in-track component
        in_track_angle = jnp.arctan2(
            jnp.linalg.norm(jnp.cross(targVec_inTrack, satVec)),
            jnp.dot(targVec_inTrack, satVec),
        )  # calculate in-track angle

        # Do a sign check for in-track angle
        # If targVec_inTrack is negative, switch
        if x_targ_sens < 0:
            in_track_angle = -in_track_angle

        targVec_crossTrack = satVec - jnp.array(
            [0, y_targ_sens, z_targ_sens]
        )  # get cross-track component
        cross_track_angle = jnp.arctan2(
            jnp.linalg.norm(jnp.cross(targVec_crossTrack, satVec)),
            jnp.dot(targVec_crossTrack, satVec),
        )  # calculate cross-track angle

        # If targVec_inTrack is negative, switch
        if x_targ_sens < 0:
            in_track_angle = -in_track_angle

        # If targVec_crossTrack is negative, switch
        if y_targ_sens < 0:
            cross_track_angle = -cross_track_angle

        # Do a sign check for cross-track angle
        # If targVec_crossTrack is negative, switch
        if y_targ_sens < 0:
            cross_track_angle = -cross_track_angle

        in_track_angle_deg = in_track_angle * 180 / jnp.pi  # convert to degrees
        cross_track_angle_deg = cross_track_angle * 180 / jnp.pi  # convert to degrees

        # Return the relative bearings from sensor to target
        return jnp.array([in_track_angle_deg, cross_track_angle_deg])

    def jacobian_ECI_to_bearings(self, sat, meas_ECI_full):
        """
        Compute the Jacobian matrix H used in a Kalman filter for the sensor. Describes
        sensitivity of the sensor measurements to changes in predicted state of the target.

        Args:
            sat (Satellite): Satellite object.
            meas_ECI_full (np.ndarray): Full ECI measurement vector [x, vx, y, vy, z, vz].

        Returns:
            np.ndarray: Jacobian matrix H.
        """
        # Extract predited position from the full ECI measurement vector
        pred_position = jnp.array(
            [meas_ECI_full[0], meas_ECI_full[2], meas_ECI_full[4]]
        )

        # Use reverse automatic differentiation since more inputs 3 than outputs 2
        jacobian = jax.jacrev(
            lambda x: self.transform_eci_to_bearings(
                sat.orbit.r.value, sat.orbit.v.value, x
            )
        )(pred_position)

        # Initialize a new Jacobian matrix with zeros
        new_jacobian = jnp.zeros((2, 6))

        # Populate the new Jacobian matrix with the relevant values
        for i in range(3):
            new_jacobian = new_jacobian.at[:, 2 * i].set(jacobian[:, i])

        return new_jacobian

    def inFOV(self, sat, targ):
        """
        Check if the target is within the satellite's field of view (FOV).

        Parameters:
            sat (Satellite): The satellite object.
            targ (Target): The target object.

        Returns:
            bool: True if the target is within the FOV, False otherwise.
        """
        # Get the target position
        l0 = targ.pos

        # Create the polygon object using the satellite position and FOV box points
        # The polygon consists of the satellite position and four corners of the projection box
        poly = np.array(
            [
                sat.orbit.r.value,
                self.projBox[0],
                self.projBox[1],
                self.projBox[2],
                self.projBox[3],
            ]
        )

        # Create a Delaunay triangulation of the polygon points
        delaunay = Delaunay(poly)

        # Check if the target position is within the Delaunay triangulation
        # The find_simplex method returns -1 if the point is outside the triangulation
        if delaunay.find_simplex(l0) >= 0:
            # Target is within the FOV
            return True
        else:
            # Target is outside the FOV
            return False

    def visible_projection(self, sat):
        """
        Compute the projection box of the sensor based on satellite position and FOV.

        Args:
            sat (Satellite): Satellite object.

        Returns:
            Updated history of current FOV projection box.
        """
        # Get the current xyz position of the satellite
        x, y, z = sat.orbit.r.value

        # Now get the projection_vectors
        proj_vecs = self.projection_vectors(sat)

        # Now find where the projection vectors intersect with the earth, given that they start at the satellite
        points = []
        for vec in proj_vecs:
            intersection = self.sphere_line_intersection(
                [0, 0, 0], 6378, [x, y, z], vec
            )  # Find the intersection of the line from the satellite to the earth
            points.append(intersection)  # add the intersection point to the list

        self.projBox = np.array(points)  # update the projection box
        return

    def projection_vectors(self, sat):
        """
        Compute the direction vectors that define the projection box based on FOV and satellite position.
        Each direction vector is 45 degrees diagonally from radial vector forming a polygon.

        Args:
            sat (Satellite): Satellite object.

        Returns:
            np.ndarray: Array containing four direction vectors of the projection box that define a polygon.
        """
        # Get the current xyz position of the satellite
        sat_pos = sat.orbit.r.value
        sat_dist = np.linalg.norm(sat_pos)

        # Find Sensor Frame
        rVec = self.normalize(np.array(sat.orbit.r.value))  # find unit radial vector
        vVec = self.normalize(np.array(sat.orbit.v.value))  # find unit in-track vector
        wVec = self.normalize(
            np.cross(sat.orbit.r.value, sat.orbit.v.value)
        )  # find unit cross-track vector

        # Create transformation matrix T
        T = jnp.stack([vVec.T, wVec.T, rVec.T])

        # Rotate satellite into sensor frame
        x_sat_sens, y_sat_sens, z_sat_sens = T @ sat_pos

        # Define the rotation axes for the four directions of square FOV in sensor frame
        rotation_axes = [
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        ]

        # Initialize list to store the new direction vectors
        dir_new_list = []

        # Calculate distance between radial vector and edges of FOV
        change = sat_dist * np.tan(np.radians(self.fov / 2))

        # Loop over the four directions
        for axis in rotation_axes:
            perp_vec = np.cross(
                [x_sat_sens, y_sat_sens, z_sat_sens], axis
            )  # Calculate the perpendicular vector for this direction
            perp_vec = perp_vec / np.linalg.norm(
                perp_vec
            )  # Normalize the perpendicular vector

            # Scale the perpendicular vector by the distance and add satellite position to get the new dir vector
            x_new = x_sat_sens + change * perp_vec[0]
            y_new = y_sat_sens + change * perp_vec[1]
            z_new = z_sat_sens + change * perp_vec[2]

            # Normalize the new position to get the new direction vector
            dir_new = -np.array([x_new, y_new, z_new]) / np.linalg.norm(
                [x_new, y_new, z_new]
            )  #  TODO: should this be normalized? Rounding Error?

            # Take the inverse of T and rotate back to ECI
            dir_new = sat_dist * np.dot(np.linalg.inv(T), dir_new)

            # Add the new direction vector to the list
            dir_new_list.append(dir_new)

        return np.array(dir_new_list)

    def sphere_line_intersection(
        self, sphere_center, sphere_radius, line_point, line_direction
    ):
        """
        Calculate the intersection point of a projection vector with a sphere. Used to find the
        the point where the satellite projection vector intersects with the Earth to define a
        polygonal area visible to the sensor.

        Parameters:
            sphere_center (tuple): The (x, y, z) coordinates of the sphere center.
            sphere_radius (float): The radius of the sphere.
            line_point (tuple): The (x, y, z) coordinates of a point on the line.
            line_direction (tuple): The (dx, dy, dz) direction vector of the line.

        Returns:
            np.ndarray: The intersection point, or None if there is no intersection.
        """
        # Unpack sphere parameters
        x0, y0, z0 = sphere_center
        r = sphere_radius

        # Unpack line parameters
        x1, y1, z1 = line_point
        dx, dy, dz = line_direction

        # Compute coefficients for the quadratic equation
        a = dx**2 + dy**2 + dz**2
        b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
        c = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - r**2

        # Compute discriminant
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            # No intersection -> vector does not intersect with the earth
            return None
        elif discriminant == 0:
            # One intersection -> vector is tangent to the earth
            t = -b / (2 * a)
            intersection_point = np.array([x1 + t * dx, y1 + t * dy, z1 + t * dz])
            return intersection_point
        else:
            # Two intersections -> vector intersects with the earth at two points (take the closest one)
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            intersection_point1 = np.array([x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz])
            intersection_point2 = np.array([x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz])

            # Calculate distances
            dist1 = np.linalg.norm(intersection_point1 - line_point)
            dist2 = np.linalg.norm(intersection_point2 - line_point)

            if dist1 < dist2:
                return intersection_point1
            else:
                return intersection_point2

    def normalize(self, vec):
        """
        Normalize a vector.

        Args:
            vec (jnp.ndarray): Input vector.

        Returns:
            jnp.ndarray: Normalized vector.
        """
        return vec / jnp.linalg.norm(vec)
