from typing import cast

import jax
import numpy as np
from numpy import typing as npt
from poliastro import twobody
from scipy import spatial

from common import linalg


class Sensor:
    """A bearings only sensor."""

    def __init__(
        self,
        fov: float,
        bearingsError: npt.NDArray,
        name: str,
        detectChance: float = 0,
        resolution: int = 720,
    ):
        """
        Initialize a sensor instance.

        Args:
            fov: Field of view of the sensor [deg].
            bearingsError: Error associated with bearings [deg].
            name: Name of the sensor.
            detectChance: Detection probability chance (default is 0).
            resolution: Resolution of the sensor (default is 720).
        """
        self.fov = fov
        self.bearingsError = bearingsError
        self.detectChance = detectChance
        self.name = name
        self.resolution = resolution

    def get_measurement(
        self, sat_orbit: twobody.Orbit, targ_pos: npt.NDArray
    ) -> npt.NDArray | None:
        """
        Get sensor measurement of a target if visible within sensor's field of view.

        Args:
            sat: Satellite object.
            targ: Target object.

        Returns:
            Sensor measurement if target is visible, else returns None.
        """

        # Get the current projection box of the sensor
        projection_box = self.get_projection_box(sat_orbit)
        if self._in_FOV(
            projection_box, targ_pos
        ):  # check if target is in the field of view
            detect = np.random.uniform(
                0, 1
            )  # generate random number to determine detection
            if detect < self.detectChance:  # check if target is detected
                return None
            else:
                return self._sensor_model(
                    sat_orbit, targ_pos
                )  # return the noisy sensor measurement for target
        else:
            return None

    def _sensor_model(
        self, sat_orbit: twobody.Orbit, target_pos: npt.NDArray
    ) -> npt.NDArray:
        """
        Simulate sensor measurement with added error.

        Args:
            sat_orbit: Orbit of the satellite.
            target_pos: Ground truth target position.

        Returns:
            Simulated sensor measurement (in-track and cross-track angles).
        """
        # Get True Relative Target Position in terms of bearings angles
        in_track_truth, cross_track_truth = self._convert_to_bearings(
            sat_orbit, target_pos
        )

        # Add Sensor Error in terms of Gaussian Noise [deg]
        in_track_meas = in_track_truth + np.random.normal(0, self.bearingsError[0])
        cross_track_meas = cross_track_truth + np.random.normal(
            0, self.bearingsError[1]
        )

        # Return the noisy sensor measurement
        return np.array([in_track_meas, cross_track_meas])

    def _convert_to_bearings(
        self, sat_orbit: twobody.Orbit, meas_ECI: npt.NDArray
    ) -> tuple[float, float]:
        """
        Convert satellite and target ECI positions to bearings angles.

        Args:
            sat: Satellite object.
            meas_ECI: Target position vector in ECI coordinates.

        Returns:
            In-track and cross-track angles between satellite and target.
        """
        sensor_measurement = self._transform_eci_to_bearings(
            sat_orbit.r.value, sat_orbit.v.value, meas_ECI
        )

        return float(sensor_measurement[0]), float(sensor_measurement[1])  # type: ignore

    def _transform_eci_to_bearings(
        self, r_value: npt.NDArray, v_value: npt.NDArray, meas_ECI: npt.NDArray
    ) -> npt.NDArray:
        """
        Transform ECI coordinates to bearings angles.

        Args:
            r_value: Current satellite position vector.
            v_value: Current satellite velocity vector.
            meas_ECI: Target position vector in ECI coordinates.

        Returns:
            np.ndarray: In-track and cross-track angles between satellite and target.
        """
        # Find Sensor Frame
        rVec = self._normalize(r_value)  # find unit radial vector
        vVec = self._normalize(v_value)  # find unit in-track vector
        wVec = self._normalize(
            np.cross(r_value, v_value)
        )  # find unit cross-track vector

        # Create transformation matrix T
        T = np.stack([vVec.T, wVec.T, rVec.T])

        # Rotate satellite and target into sensor frame
        sat_pos = np.array(r_value)  # get satellite position
        x_sat_sens, y_sat_sens, z_sat_sens = (
            T @ sat_pos
        )  # rotate satellite into sensor frame

        meas_ECI_sym = np.array(meas_ECI)  # get noisy measurement
        x_targ_sens, y_targ_sens, z_targ_sens = (
            T @ meas_ECI_sym
        )  # rotate measurement into sensor frame

        # Get the relative bearings from sensor to target
        satVec = np.array(
            [x_sat_sens, y_sat_sens, z_sat_sens]
        )  # get satellite vector in sensor frame

        # Get the In-Track and Cross-Track angles
        targVec_inTrack = satVec - np.array(
            [x_targ_sens, 0, z_targ_sens]
        )  # get in-track component
        in_track_angle = np.arctan2(
            np.linalg.norm(np.cross(targVec_inTrack, satVec)),
            np.dot(targVec_inTrack, satVec),
        )  # calculate in-track angle

        # Do a sign check for in-track angle
        # If targVec_inTrack is negative, switch
        if x_targ_sens < 0:
            in_track_angle = -in_track_angle

        targVec_crossTrack = satVec - np.array(
            [0, y_targ_sens, z_targ_sens]
        )  # get cross-track component
        cross_track_angle = np.arctan2(
            np.linalg.norm(np.cross(targVec_crossTrack, satVec)),
            np.dot(targVec_crossTrack, satVec),
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

        in_track_angle_deg = in_track_angle * 180 / np.pi  # convert to degrees
        cross_track_angle_deg = cross_track_angle * 180 / np.pi  # convert to degrees

        # Return the relative bearings from sensor to target
        return np.array([in_track_angle_deg, cross_track_angle_deg])

    def jacobian_ECI_to_bearings(
        self, sat_orbit: twobody.Orbit, meas_ECI_full: npt.NDArray
    ) -> npt.NDArray:
        """
        Compute the Jacobian matrix H used in a Kalman filter for the sensor. Describes
        sensitivity of the sensor measurements to changes in predicted state of the target.

        Args:
            sat: Satellite object.
            meas_ECI_full: Full ECI measurement vector [x, vx, y, vy, z, vz].

        Returns:
            Jacobian matrix H.
        """
        # Extract predited position from the full ECI measurement vector
        pred_position = np.array([meas_ECI_full[0], meas_ECI_full[2], meas_ECI_full[4]])

        # Use reverse automatic differentiation since more inputs 3 than outputs 2
        jacobian = jax.jacrev(
            lambda x: self._transform_eci_to_bearings(
                sat_orbit.r.value, sat_orbit.v.value, x
            )
        )(pred_position)

        # Initialize a new Jacobian matrix with zeros
        new_jacobian = np.zeros((2, 6))

        # Populate the new Jacobian matrix with the relevant values
        for i in range(3):
            new_jacobian = new_jacobian[:, 2 * i] = jacobian[:, i]

        return cast(npt.NDArray, new_jacobian)

    def _in_FOV(self, projection_box: npt.NDArray, targ_pos: npt.NDArray) -> bool:
        """
        Check if the target is within the satellite's field of view (FOV).

        Parameters:
            sat: The satellite object.
            targ: The target object.

        Returns:
            True if the target is within the FOV, False otherwise.
        """
        # Get the target position
        l0 = targ_pos

        # Create a Delaunay triangulation of the polygon points
        delaunay = spatial.Delaunay(projection_box)

        # Check if the target position is within the Delaunay triangulation
        # The find_simplex method returns -1 if the point is outside the triangulation
        return delaunay.find_simplex(l0) >= 0  # type: ignore

    def get_projection_box(self, sat_orbit: twobody.Orbit) -> npt.NDArray:
        """
        Compute the projection box of the sensor based on satellite position and FOV.

        Args:
            sat (Satellite): Satellite object.

        Returns:
            Updated history of current FOV projection box.
        """
        # Get the current xyz position of the satellite
        x, y, z = sat_orbit.r.value

        # Now get the projection vectors
        proj_vecs = self._projection_vectors(sat_orbit)

        # Now find where the projection vectors intersect with the earth, given that they start at the satellite
        points = []
        for vec in proj_vecs:
            intersection = linalg.sphere_line_intersection(
                np.array([0, 0, 0]), np.array([x, y, z]), vec
            )  # Find the intersection of the line from the satellite to the earth
            points.append(intersection)  # add the intersection point to the list

        # Create the polygon object using the satellite position and FOV box points
        # The polygon consists of the satellite position and four corners of the projection box
        return np.array([sat_orbit.r.value] + points)

    def _projection_vectors(self, sat_orbit: twobody.Orbit) -> npt.NDArray:
        """
        Compute the direction vectors that define the projection box based on FOV and satellite position.
        Each direction vector is 45 degrees diagonally from radial vector forming a polygon.

        Args:
            sat: Satellite object.

        Returns:
            Array containing four direction vectors of the projection box that define a polygon.
        """
        # Get the current xyz position of the satellite
        sat_pos = sat_orbit.r.value
        sat_dist = np.linalg.norm(sat_pos)

        # Find Sensor Frame
        rVec = self._normalize(np.array(sat_orbit.r.value))  # find unit radial vector
        vVec = self._normalize(np.array(sat_orbit.v.value))  # find unit in-track vector
        wVec = self._normalize(
            np.cross(sat_orbit.r.value, sat_orbit.v.value)
        )  # find unit cross-track vector

        # Create transformation matrix T
        T = np.stack([vVec.T, wVec.T, rVec.T])

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

    def _normalize(self, vec: npt.NDArray) -> npt.NDArray:
        """Normalize a vector

        Args:
            vec: Input vector.

        Returns a normalized vector.
        """
        return vec / np.linalg.norm(vec)
