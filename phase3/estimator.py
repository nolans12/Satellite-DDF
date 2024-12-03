from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import typing as jnpt
from numpy import typing as npt
from scipy import optimize

from common import dataclassframe
from phase3 import collection


class Estimator:
    """Base EFK Estimator"""

    def __init__(self):
        """
        Initialize the BaseEstimator object.

        Initalizes the data frame for storing the data
        """

        # Meas data dictionary, keyed by targetID into a measurement data frame
        self.initialization_data = {}

        # Estimation data
        self.estimation_data = dataclassframe.DataClassFrame(clz=collection.Estimate)

    def _EKF_initialize(self, measurement: collection.Measurement) -> None:
        """
        Initialize the EKF using the measurement.
        """

        # Take the bearings measurement and shoot it down to the Earth, then take that as initial state
        target_id = measurement.target_id
        time = measurement.time
        sat_pos = measurement.sat_state
        r = sat_pos[0:3]
        v = sat_pos[3:6]
        alpha = measurement.alpha
        beta = measurement.beta

        # Get the transform from ECI to the sensor frame
        rVec = r / np.linalg.norm(r)
        vVec = v / np.linalg.norm(v)
        wVec = np.cross(rVec, vVec)
        wVec = wVec / np.linalg.norm(wVec)
        T = np.stack([vVec, wVec, rVec])

        sat_pos_sens = T @ r

        # Just get the distance from Earth
        h = np.linalg.norm(sat_pos_sens) - 6378

        # Get the magnitude of in track and cross track, use tangent
        in_track = h * np.tan(alpha * np.pi / 180)
        cross_track = h * np.tan(beta * np.pi / 180)

        # Get the position of targ estimate
        pos_est_sens = np.array([in_track, cross_track, 6378])

        # Now take estimate back into ECI
        pos_est = T.T @ pos_est_sens

        # Now, take this position and use it as the initial state
        est_init = np.array([pos_est[0], 0, pos_est[1], 0, pos_est[2], 0])

        # Initialize the covariance
        P_init = np.eye(6) * 1000

        # Save the initial values
        self.save_current_estimation_data(
            target_id, time, est_init, P_init, np.zeros(2), np.eye(2)
        )

    def EKF_predict(self, measurements: list[collection.Measurement]) -> None:
        """
        Predict the next state of the target using the state transition function.
        """

        # For each targetID, recieved, either predict or initialize
        for measurement in measurements:

            target_id = measurement.target_id

            # Check, does a filter exist for this targetID?
            if (
                self.estimation_data.empty
                or target_id not in self.estimation_data.target_id.values
                or self.estimation_data.iloc[-1].time + 5
                < measurement.time  # old estimate is too old, 5 minute buffer?
            ):
                # print(f"Initializing target {target_id}")
                self._EKF_initialize(measurement)
                continue

            # If we have an estimate, predict it!

            # Get most recent estimate and covariance
            latest_estimate = self.estimation_data[
                self.estimation_data.target_id == target_id
            ].iloc[-1]
            time_prior = latest_estimate.time
            est_prior = latest_estimate.estimate
            P_prior = latest_estimate.covariance

            # Calculate time difference since last estimate
            dt = measurement.time - time_prior
            # print(f"Predicting target {target_id} with dt = {dt}")

            # Predict next state using state transition function
            est_pred = self.state_transition(est_prior, dt)

            # Evaluate Jacobian of state transition function
            F = self.state_transition_jacobian(est_prior, dt)

            # Predict process noise associated with state transition
            Q = np.diag([0.1, 0.01, 0.1, 0.01, 0.1, 0.01])

            # Predict covariance
            P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

            self.save_current_estimation_data(
                target_id, measurement.time, est_pred, P_pred, np.zeros(2), np.eye(2)
            )

    def EKF_update(self, measurements: list[collection.Measurement]) -> None:
        """
        Update the estimate of the target using the measurement.

        Args:
            measurements: List of measurements containing targetID, time, satellite state, and bearings
        """

        # Get unique target IDs from measurements
        target_ids = set(m.target_id for m in measurements)

        # Process each target ID separately
        for target_id in target_ids:
            # Get measurements for this target
            target_measurements = [m for m in measurements if m.target_id == target_id]

            if not target_measurements:
                continue

            # Get the prior estimate and covariance
            latest_estimate = self.estimation_data[
                self.estimation_data.target_id == target_id
            ].iloc[-1]

            meas_time = target_measurements[0].time

            # Does the current time equal the time of the latest estimate?
            # If so, then latest was just a prediction, so we can remove it and replace it with the update (so dont get double count in data)
            if meas_time == latest_estimate.time:
                self.estimation_data = self.estimation_data.iloc[:-1]

            est_pred = latest_estimate.estimate
            P_pred = latest_estimate.covariance

            # Assume that the measurements are in the form of [alpha, beta] for each satellite
            num_measurements = 2 * len(target_measurements)

            # Prepare for measurements and update the estimate
            z = np.zeros((num_measurements, 1))  # Stacked vector of measurements
            H = np.zeros((num_measurements, 6))  # Jacobian of the sensor model
            R = np.zeros((num_measurements, num_measurements))  # Sensor noise matrix
            innovation = np.zeros((num_measurements, 1))

            # Iterate over satellites to get measurements and update matrices
            for i, measurement in enumerate(target_measurements):
                z[2 * i : 2 * i + 2] = np.reshape(
                    [measurement.alpha, measurement.beta], (2, 1)
                )  # Measurement stack
                H[2 * i : 2 * i + 2, 0:6] = self.jacobian_eci_to_bearings(
                    measurement, est_pred
                )  # Jacobian of the sensor model
                R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = (
                    measurement.R_mat
                )  # Sensor noise matrix
                z_pred = self.eci_to_bearings(
                    measurement.sat_state,
                    np.array([est_pred[0], est_pred[2], est_pred[4]]),
                )  # Predicted measurements
                innovation[2 * i : 2 * i + 2] = z[2 * i : 2 * i + 2] - np.reshape(
                    z_pred, (2, 1)
                )  # Innovations

            # Calculate innovation covariance
            innovation_cov = np.dot(H, np.dot(P_pred, H.T)) + R

            # Solve for Kalman gain
            K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovation_cov)))

            # Correct prediction
            est = est_pred + np.reshape(np.dot(K, innovation), (6))

            # Correct covariance
            P = P_pred - np.dot(K, np.dot(H, P_pred))

            # Save data
            self.save_current_estimation_data(
                target_id, meas_time, est, P, innovation, innovation_cov
            )

    def state_transition(self, estPrior: npt.NDArray, dt: float) -> jnpt.ArrayLike:
        """
        State Transition Function: Converts ECI Cartesian state to spherical coordinates,
        propagates state over time using Runge-Kutta 4th order integration, and converts
        back to Cartesian. Inputs current state and time step, returns next state.

        Parameters:
        - estPrior: Current state in Cartesian coordinates [x, vx, y, vy, z, vz].
        - dt: Time step for integration.

        Returns:
        - Next state in Cartesian coordinates [x, vx, y, vy, z, vz].
        """
        x, vx, y, vy, z, vz = estPrior

        # Convert to Spherical Coordinates
        range = jnp.sqrt(x**2 + y**2 + z**2)
        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x)

        # Calculate the Range Rate
        rangeRate = (x * vx + y * vy + z * vz) / range

        # Calculate Elevation Rate
        elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / (
            (x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2)
        )

        # Calculate Azimuth Rate
        azimuthRate = (x * vy - y * vx) / (x**2 + y**2)

        # Previous Spherical State
        prev_spherical_state = jnp.array(
            [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        )

        # Define function to compute derivatives for Runge-Kutta method
        def derivatives(spherical_state: jnpt.ArrayLike) -> jnpt.ArrayLike:
            """
            Computes derivatives of spherical state variables.

            Parameters:
            - spherical_state (array-like): Spherical state variables [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate].

            Returns:
            - array-like: Derivatives [rangeRate, rangeAccel, elevationRate, elevationAccel, azimuthRate, azimuthAccel].
            """
            r, rdot, elevation, elevationRate, azimuth, azimuthRate = spherical_state

            rangeRate = rdot
            rangeAccel = 0  # No acceleration assumed in this simplified model
            elevationRate = elevationRate
            elevationAccel = 0  # No acceleration assumed in this simplified model
            azimuthRate = azimuthRate
            azimuthAccel = 0  # No acceleration assumed in this simplified model

            return jnp.array(
                [
                    rangeRate,
                    rangeAccel,
                    elevationRate,
                    elevationAccel,
                    azimuthRate,
                    azimuthAccel,
                ]
            )

        # Runge-Kutta 4th order integration
        k1 = derivatives(prev_spherical_state)
        k2 = derivatives(prev_spherical_state + 0.5 * dt * k1)
        k3 = derivatives(prev_spherical_state + 0.5 * dt * k2)
        k4 = derivatives(prev_spherical_state + dt * k3)

        next_spherical_state = prev_spherical_state + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

        # Extract components from spherical state
        range = next_spherical_state[0]
        rangeRate = next_spherical_state[1]
        elevation = next_spherical_state[2]
        elevationRate = next_spherical_state[3]
        azimuth = next_spherical_state[4]
        azimuthRate = next_spherical_state[5]

        # Convert back to Cartesian coordinates
        x = range * jnp.cos(elevation) * jnp.cos(azimuth)
        y = range * jnp.cos(elevation) * jnp.sin(azimuth)
        z = range * jnp.sin(elevation)

        # Approximate velocities conversion (simplified version)
        vx = (
            rangeRate * jnp.cos(elevation) * jnp.cos(azimuth)
            - range * elevationRate * jnp.sin(elevation) * jnp.cos(azimuth)
            - range * azimuthRate * jnp.cos(elevation) * jnp.sin(azimuth)
        )

        vy = (
            rangeRate * jnp.cos(elevation) * jnp.sin(azimuth)
            - range * elevationRate * jnp.sin(elevation) * jnp.sin(azimuth)
            + range * azimuthRate * jnp.cos(elevation) * jnp.cos(azimuth)
        )

        vz = rangeRate * jnp.sin(elevation) + range * elevationRate * jnp.cos(elevation)

        # Return the next state in Cartesian coordinates
        return jnp.array([x, vx, y, vy, z, vz])

    def state_transition_jacobian(
        self, estPrior: npt.NDArray, dt: float
    ) -> jnpt.ArrayLike:
        """
        Calculates the Jacobian matrix of the state transition function.

        Parameters:
        - estPrior: Current state in Cartesian coordinates [x, vx, y, vy, z, vz].
        - dt: Time step for integration.

        Returns:
        - Jacobian matrix of the state transition function.
        """
        jacobian = jax.jacfwd(lambda x: self.state_transition(x, dt))(
            jnp.array(estPrior)
        )

        return jacobian

    def eci_to_bearings(
        self, sat_state: npt.NDArray, est_ECI_pos: npt.NDArray
    ) -> jnpt.ArrayLike:
        """
        Transform ECI coordinates to bearings angles.

        Args:
            sat_state: Current satellite state vector. [x, y, z, vx, vy, vz]
            est_ECI: Target position vector in ECI coordinates [x, y, z].

        Returns:
            np.ndarray: In-track and cross-track angles between satellite and target.
        """

        rVec = jnp.array(sat_state[0:3])
        vVec = jnp.array(sat_state[3:6])

        rUnit = rVec / jnp.linalg.norm(rVec)
        vUnit = vVec / jnp.linalg.norm(vVec)
        wUnit = jnp.cross(rUnit, vUnit) / jnp.linalg.norm(jnp.cross(rUnit, vUnit))

        T = jnp.stack([vUnit, wUnit, rUnit])

        x_sat_sens, y_sat_sens, z_sat_sens = (
            T @ rVec
        )  # rotate satellite into sensor frame

        meas_ECI_sym = jnp.array(est_ECI_pos)  # get noisy measurement
        x_targ_sens, y_targ_sens, z_targ_sens = (
            T @ meas_ECI_sym
        )  # rotate target into sensor frame

        satVec = jnp.array([x_sat_sens, y_sat_sens, z_sat_sens])

        targVec_inTrack = satVec - jnp.array(
            [x_targ_sens, 0, z_targ_sens]
        )  # get in-track component
        in_track_angle = jnp.arctan2(
            jnp.linalg.norm(jnp.cross(targVec_inTrack, satVec)),
            jnp.dot(targVec_inTrack, satVec),
        )  # calculate in-track angle

        # If targVec_inTrack is negative, switch
        in_track_angle = jnp.where(x_targ_sens < 0, -in_track_angle, in_track_angle)

        targVec_crossTrack = satVec - jnp.array(
            [0, y_targ_sens, z_targ_sens]
        )  # get cross-track component
        cross_track_angle = jnp.arctan2(
            jnp.linalg.norm(jnp.cross(targVec_crossTrack, satVec)),
            jnp.dot(targVec_crossTrack, satVec),
        )  # calculate cross-track angle

        # If targVec_crossTrack is negative, switch
        cross_track_angle = jnp.where(
            y_targ_sens < 0, -cross_track_angle, cross_track_angle
        )

        in_track_angle_deg = in_track_angle * 180 / jnp.pi  # convert to degrees
        cross_track_angle_deg = cross_track_angle * 180 / jnp.pi  # convert to degrees

        return jnp.array([in_track_angle_deg, cross_track_angle_deg])

    def jacobian_eci_to_bearings(
        self, measurement: collection.Measurement, est_ECI: npt.NDArray
    ) -> npt.NDArray:
        """
        Calculate the Jacobian matrix of the ECI to bearings transformation.
        """

        # Get the satellite state
        sat_state = measurement.sat_state

        # Get the position estimate
        est_ECI_pos = jnp.array([est_ECI[0], est_ECI[2], est_ECI[4]])

        jacobian = jax.jacrev(lambda x: self.eci_to_bearings(sat_state, x))(est_ECI_pos)

        # Initialize a new Jacobian matrix with zeros
        new_jacobian = jnp.zeros((2, 6))

        # Populate the new Jacobian matrix with the relevant values
        for i in range(3):
            new_jacobian = new_jacobian.at[:, 2 * i].set(jacobian[:, i])

        return cast(npt.NDArray, new_jacobian)

    def CI(self, estimates: list[collection.Estimate]) -> None:
        """
        Helper function to perform Covariance Intersection fusion on a list of estimates.

        Args:
            estimates: List of Estimate objects to fuse

        Updates the current estimate with all inputted estimates
        """

        # Get all target_ids in the list
        target_ids = [est.target_id for est in estimates]

        # Then, iteratively do CI for each target_id sent
        for target_id in target_ids:

            # Get all estimates for this target
            target_estimates = [est for est in estimates if est.target_id == target_id]

            # Check, do i have an estimate for this target? If so, add it to the list
            if (
                not self.estimation_data.empty
                and target_id in self.estimation_data.target_id.values
            ):
                # Get the current estimate
                current_estimate = self.estimation_data[
                    self.estimation_data.target_id == target_id
                ].iloc[-1]
                target_estimates.append(current_estimate)

            # Now, loop through all estimates and iteratively do CI
            start_estimate = target_estimates[0]
            if len(target_estimates) == 1:
                fused_estimate = start_estimate  # If theres only 1 estimate in list, just take it as the fused
            else:
                fused_estimate = start_estimate
                # Otherwise, iteratively do CI
                for next_estimate in target_estimates[1:]:
                    fused_estimate = self.CI_helper(fused_estimate, next_estimate)

            # If we have an existing estimate for this target, override the most recent one
            if (
                not self.estimation_data.empty
                and target_id in self.estimation_data.target_id.values
            ):
                # Drop the old estimate
                index_to_delete = self.estimation_data[
                    self.estimation_data.target_id == target_id
                ].index[-1]
                self.estimation_data = self.estimation_data.drop(index=index_to_delete)

            # Add the fused estimate
            self.save_current_estimation_data(
                target_id,
                fused_estimate.time,
                fused_estimate.estimate,
                fused_estimate.covariance,
                np.zeros(2),
                np.eye(2),
            )

    def CI_helper(
        self, est1: collection.Estimate, est2: collection.Estimate
    ) -> collection.Estimate:
        """
        Helper function to perform CI fusion on two estimates.
        """

        # Figure out which time is newer, or are they the same?
        if est1.time > est2.time:
            newer_est = est1
            older_est = est2
            dt = newer_est.time - older_est.time
        else:  # est2.time >= est1.time
            newer_est = est2
            older_est = est1
            dt = newer_est.time - older_est.time

        est_new = newer_est.estimate
        cov_new = newer_est.covariance

        # Predict the older estimate (might be dt = 0, doesnt matter)
        est_prior = self.state_transition(older_est.estimate, dt)
        F = self.state_transition_jacobian(est_prior, dt)
        cov_prior = np.dot(F, np.dot(older_est.covariance, F.T))

        # Minimize the covariance determinant
        omega_opt = optimize.minimize(
            self.det_of_fused_covariance,
            [0.5],
            args=(cov_prior, cov_new),
            bounds=[(0, 1)],
        ).x

        # Compute the fused covariance
        cov_fused = np.linalg.inv(
            omega_opt * np.linalg.inv(cov_prior)
            + (1 - omega_opt) * np.linalg.inv(cov_new)
        )
        est_fused = cov_fused @ (
            omega_opt * np.linalg.inv(cov_prior) @ est_prior
            + (1 - omega_opt) * np.linalg.inv(cov_new) @ est_new
        )

        return collection.Estimate(
            target_id=est1.target_id,
            time=newer_est.time,
            estimate=est_fused,
            covariance=cov_fused,
            innovation=np.zeros(2),
            innovation_covariance=np.eye(2),
            track_uncertainty=self.calcTrackError(est_fused, cov_fused),
        )

    def det_of_fused_covariance(
        self, omega: float, cov1: npt.NDArray, cov2: npt.NDArray
    ) -> float:
        """
        Calculate the determinant of the fused covariance matrix.

        Args:
            omega: Weight of the first covariance matrix
            cov1: Covariance matrix of the first estimate
            cov2: Covariance matrix of the second estimate

        Returns:
            Determinant of the fused covariance matrix
        """
        P = np.linalg.inv(
            omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2)
        )
        return np.linalg.det(P)

    def calcTrackError(self, est: npt.NDArray, cov: npt.NDArray) -> float:
        """
        Calculate the track quality metric for the current estimate.

        Args:
        - est (array-like): Current estimate for the target ECI state = [x, vx, y, vy, z, vz].
        - cov (array-like): Current covariance matrix for the target ECI state.

        Returns:
        - float: Track quality metric.
        """
        # Since we are guaranteeing that the state will always live within the covariance estimate
        # Why not just use the diag of the covariance matrix, then calculate the 2 sigma bounds of xyz
        # Then take norm of xyz max as the error

        # Calculate the 2 sigma bounds of the position
        diag = np.diag(cov)

        # Calculate the std
        posStd = np.array([np.sqrt(diag[0]), np.sqrt(diag[2]), np.sqrt(diag[4])])

        # Now calculate the 2 sigma bounds of the position
        pos2Sigma = 2 * posStd

        # Now get the norm of the 2 sigma bounds xyz
        posError_1side = np.linalg.norm(pos2Sigma)

        # Multiple this value by 2 to get the total error the state could be in
        posError = 2 * posError_1side

        return posError

    def save_current_estimation_data(
        self,
        targetID: int,
        time: float,
        est: npt.NDArray,
        cov: npt.NDArray,
        innovation: npt.NDArray,
        innovationCov: npt.NDArray,
    ) -> None:
        """
        Save the current estimation data for the target.
        """
        new_estimate = pd.DataFrame(
            [
                collection.Estimate(
                    target_id=targetID,
                    time=time,
                    estimate=est,
                    covariance=cov,
                    innovation=innovation,
                    innovation_covariance=innovationCov,
                    track_uncertainty=self.calcTrackError(est, cov),
                )
            ]
        )

        self.estimation_data = pd.concat(
            [self.estimation_data, new_estimate], ignore_index=True
        )
