from collections import defaultdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import typing as jnpt
from numpy import typing as npt
from scipy import optimize
from scipy import stats

from phase3 import comm

if TYPE_CHECKING:
    # Circular import resolution
    from phase3 import satellite

from phase3 import target


class BaseEstimator:
    def __init__(self, targetPriorities: dict[int, int]):
        """
        Initialize the BaseEstimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """

        # Define the targetPriorities dictioanry
        self.targetPriorities = targetPriorities

        # Define the targets to track
        self.targs = targetPriorities.keys()

        # Define history vectors for each extended Kalman filter
        self.estHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of Kalman estimates in ECI coordinates
        self.estPredHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of Kalman estimates in ECI coordinates
        self.covarianceHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of covariance matrices
        self.covariancePredHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of predicted covariance matrices
        self.innovationHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of innovations
        self.innovationCovHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of innovation covariances
        self.neesHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of NEES (Normalized Estimation Error Squared)
        self.nisHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of NIS (Normalized Innovation Squared)
        self.trackErrorHist = {
            targetID: defaultdict(dict) for targetID in targetPriorities.keys()
        }  # History of track quality metric

    def EKF_initialize(self, target: target.Target, envTime: float) -> None:
        prior_pos = np.array(
            [target.pos[0], target.pos[1], target.pos[2]]
        ) + np.random.normal(0, 15, 3)
        prior_vel = np.array(
            [target.vel[0], target.vel[1], target.vel[2]]
        ) + np.random.normal(0, 1.5, 3)
        est_prior = np.array(
            [
                prior_pos[0],
                prior_vel[0],
                prior_pos[1],
                prior_vel[1],
                prior_pos[2],
                prior_vel[2],
            ],
            dtype=np.float32,
        )
        # Initial covariance matrix
        P_prior = np.array(
            [
                [2500, 0, 0, 0, 0, 0],
                [0, 100, 0, 0, 0, 0],
                [0, 0, 2500, 0, 0, 0],
                [0, 0, 0, 100, 0, 0],
                [0, 0, 0, 0, 2500, 0],
                [0, 0, 0, 0, 0, 100],
            ]
        )

        # Store initial values and return for first iteration
        self.save_current_estimation_data(
            target.targetID, envTime, est_prior, P_prior, np.zeros(2), np.eye(2)
        )

    def EKF_pred(self, targetID: int, envTime: float) -> None:
        # Get most recent estimate and covariance
        time_prior = max(self.estHist[targetID].keys())
        est_prior = self.estHist[targetID][time_prior]
        P_prior = self.covarianceHist[targetID][time_prior]

        # Calculate time difference since last estimate
        dt = envTime - time_prior

        # ### Also reset the filter if its been a certain amount of time since the last estimate
        # if dt > 30:
        #     prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
        #     prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 1.5, 3)
        #     est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
        #     # Initial covariance matrix
        #     P_prior = np.array([[2500, 0, 0, 0, 0, 0],
        #                         [0, 100, 0, 0, 0, 0],
        #                         [0, 0, 2500, 0, 0, 0],
        #                         [0, 0, 0, 100, 0, 0],
        #                         [0, 0, 0, 0, 2500, 0],
        #                         [0, 0, 0, 0, 0, 100]])

        #     dt = 0 # Reset the time difference since were just reinitializing at this time

        # Predict next state using state transition function
        est_pred = self.state_transition(est_prior, dt)

        # Evaluate Jacobian of state transition function
        F = self.state_transition_jacobian(est_prior, dt)

        # Predict process noise associated with state transition
        Q = np.diag([0.1, 0.01, 0.1, 0.01, 0.1, 0.01])

        # Predict covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

        self.save_current_estimation_data(
            targetID, envTime, est_pred, P_pred, np.zeros(2), np.eye(2)
        )
        self.estPredHist[targetID][envTime] = est_pred
        self.covariancePredHist[targetID][envTime] = P_pred

    def EKF_update(
        self,
        sats: list['satellite.Satellite'],
        measurements,
        targetID: int,
        envTime: float,
    ) -> None:

        if not isinstance(measurements[0], np.ndarray):
            return

        # Get the prior estimate and covariance
        time_prior = max(self.estHist[targetID].keys())
        est_pred = self.estHist[targetID][time_prior]
        P_pred = self.covarianceHist[targetID][time_prior]

        # Assume that the measurements are in the form of [alpha, beta] for each satellite
        numMeasurements = 2 * len(measurements)

        # Prepare for measurements and update the estimate
        z = np.zeros((numMeasurements, 1))  # Stacked vector of measurements
        H = np.zeros((numMeasurements, 6))  # Jacobian of the sensor model
        R = np.zeros((numMeasurements, numMeasurements))  # Sensor noise matrix
        innovation = np.zeros((numMeasurements, 1))

        # Iterate over satellites to get measurements and update matrices
        for i, sat in enumerate(sats):
            z[2 * i : 2 * i + 2] = np.reshape(
                measurements[i][:], (2, 1)
            )  # Measurement stack
            H[2 * i : 2 * i + 2, 0:6] = sat.sensor.jacobian_ECI_to_bearings(
                sat, est_pred
            )  # Jacobian of the sensor model
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = (
                np.eye(2) * sat.sensor.bearingsError**2
            )  # Sensor noise matrix

            z_pred = np.array(
                sat.sensor.convert_to_bearings(
                    sat, np.array([est_pred[0], est_pred[2], est_pred[4]])
                )
            )  # Predicted measurements
            innovation[2 * i : 2 * i + 2] = z[2 * i : 2 * i + 2] - np.reshape(
                z_pred, (2, 1)
            )  # Innovations

        # Calculate innovation covariance
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R

        # Solve for Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

        # Correct prediction
        est = est_pred + np.reshape(np.dot(K, innovation), (6))

        # Correct covariance
        P = P_pred - np.dot(K, np.dot(H, P_pred))

        # Save data
        self.save_current_estimation_data(
            targetID, envTime, est, P, innovation, innovationCov
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
        def derivatives(spherical_state: npt.NDArray) -> jnpt.ArrayLike:
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

        # print(f"Position Error: {posError}")

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

        Args:
        - targetID: Target ID.
        - time: Current environment time.
        - est: Current estimate for the target ECI state = [x, vx, y, vy, z, vz].
        - cov: Current covariance matrix for the target ECI state.
        - innovation: Innovation vector.
        - innovationCov: Innovation covariance matrix.
        """

        self.estHist[targetID][time] = est
        self.covarianceHist[targetID][time] = cov
        self.innovationHist[targetID][time] = innovation
        self.innovationCovHist[targetID][time] = innovationCov

        # Calculate Track Quaility Metric
        trackError = self.calcTrackError(est, cov)
        self.trackErrorHist[targetID][time] = trackError


### Central Estimator Class
class CentralEstimator(BaseEstimator):
    def __init__(self, targPriorities: dict[int, int]):
        """
        Initialize Central Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)

        self.R_factor = 1  # Factor to scale the sensor noise matrix

    def central_EKF_initialize(
        self, target: target.Target, envTime: float
    ) -> None:
        """
        Centralized Extended Kalman Filter initialization step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_initialize(target, envTime)

    def central_EKF_pred(self, targetID: int, envTime: float) -> None:
        """
        Centralized Extended Kalman Filter prediction step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_pred(targetID, envTime)

    def central_EKF_update(
        self,
        sats: list['satellite.Satellite'],
        measurements,
        targetID: int,
        envTime: float,
    ) -> None:
        """
        Centralized Extended Kalman Filter update step.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_update(sats, measurements, targetID, envTime)


### Independent Estimator Class
class IndeptEstimator(BaseEstimator):
    def __init__(self, targPriorities: dict[int, int]):
        """
        Initialize Independent Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)

        self.R_factor = 1

    def local_EKF_initialize(self, target: target.Target, envTime: float) -> None:
        """
        Local Extended Kalman Filter initialization step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_initialize(target, envTime)

    def local_EKF_pred(self, targetID: int, envTime: float) -> None:
        """
        Local Extended Kalman Filter prediction step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_pred(targetID, envTime)

    def local_EKF_update(
        self,
        sats: list['satellite.Satellite'],
        measurements,
        targetID: int,
        envTime: float,
    ) -> None:
        """
        Local Extended Kalman Filter update step.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_update(sats, measurements, targetID, envTime)


### CI Estimator Class
class CiEstimator(BaseEstimator):
    def __init__(self, targPriorities: dict[int, int]):
        """
        Initialize DDF Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)

        self.R_factor = 1  # Factor to scale the sensor noise matrix

    def ci_EKF_initialize(self, target: target.Target, envTime: float) -> None:
        """
        Covariance Intersection Extended Kalman Filter initialization step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_initialize(target, envTime)

    def ci_EKF_pred(self, targetID: int, envTime: float) -> None:
        """
        Covariance Intersection Extended Kalman Filter prediction step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_pred(targetID, envTime)

    def ci_EKF_update(
        self,
        sats: list['satellite.Satellite'],
        measurements,
        targetID: int,
        envTime: float,
    ) -> None:
        """
        Covariance Intersection Extended Kalman Filter update step.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_update(sats, measurements, targetID, envTime)

    def CI(self, sat: 'satellite.Satellite', comms) -> None:
        """
        Covariance Intersection function to conservatively combine received estimates and covariances
        into updated target state and covariance.

        Args:
            sat: Satellite object that is receiving information.
            commNode (dict): Communication node containing queued data from satellites.

        Returns:
            estPrior (array-like): Updated current state in Cartesian coordinates [x, vx, y, vy, z, vz].
            covPrior (array-like): Updated current covariance matrix.
        """

        commNode = comms.G.nodes[sat]

        # Check if there is any information in the queue:
        if len(commNode['estimate_data']) == 0:
            # If theres no information in the queue, return
            return

        # There is information in the queue, get the newest info
        time_sent = max(commNode['estimate_data'].keys())

        # Check all the targets that are being talked about in the queue
        for targetID in commNode['estimate_data'][time_sent].keys():

            # Is that target something this satellite should be tracking? Or just ferrying data?
            if targetID not in sat.targetIDs:
                continue

            # For each target we should track, loop through all the estimates and covariances
            for i in range(len(commNode['estimate_data'][time_sent][targetID]['est'])):
                senderName = commNode['estimate_data'][time_sent][targetID]['sender'][i]
                est_sent = commNode['estimate_data'][time_sent][targetID]['est'][i]
                cov_sent = commNode['estimate_data'][time_sent][targetID]['cov'][i]

                # Check if satellite has an estimate and covariance for this target already
                if not self.estHist[targetID] and not self.covarianceHist[targetID]:
                    # If not, use the sent estimate and covariance to initialize
                    self.estHist[targetID][time_sent] = est_sent
                    self.covarianceHist[targetID][time_sent] = cov_sent
                    self.trackErrorHist[targetID][time_sent] = self.calcTrackError(
                        est_sent, cov_sent
                    )
                    continue

                # If satellite has an estimate and covariance for this target already, check if we should CI
                time_prior = max(self.estHist[targetID].keys())

                # If the send time is older than the prior estimate, discard the sent estimate
                if time_sent < time_prior:
                    continue

                # Now check, does the satellite need help on this target?
                if not not sat.ciEstimator.trackErrorHist[targetID][
                    time_prior
                ]:  # An estimate exists for this target
                    if (
                        sat.ciEstimator.trackErrorHist[targetID][time_prior]
                        < sat.targPriority[targetID]
                    ):  # Is the estimate good enough already?
                        # If the track quality is good, don't do CI
                        continue

                # We will now use the estimate and covariance that were sent, so we should store this
                comms.used_comm_data[targetID][sat.name][senderName][time_sent] = (
                    est_sent.size * 2 + cov_sent.size / 2
                )

                # If the time between the sent estimate and the prior estimate is greater than 5 minutes, discard the prior
                if time_sent - time_prior > 5:
                    self.estHist[targetID][time_sent] = est_sent
                    self.covarianceHist[targetID][time_sent] = cov_sent
                    continue

                # Else, let's do CI
                est_prior = self.estHist[targetID][time_prior]
                cov_prior = self.covarianceHist[targetID][time_prior]

                # Propagate the prior estimate and covariance to the new time
                dt = time_sent - time_prior
                est_prior = self.state_transition(est_prior, dt)
                F = self.state_transition_jacobian(est_prior, dt)
                cov_prior = np.dot(F, np.dot(cov_prior, F.T))

                # Minimize the covariance determinant
                omega_opt = optimize.minimize(
                    self.det_of_fused_covariance,
                    [0.5],
                    args=(cov_prior, cov_sent),
                    bounds=[(0, 1)],
                ).x

                # Compute the fused covariance
                cov1 = cov_prior
                cov2 = cov_sent
                cov_prior = np.linalg.inv(
                    omega_opt * np.linalg.inv(cov1)
                    + (1 - omega_opt) * np.linalg.inv(cov2)
                )
                est_prior = cov_prior @ (
                    omega_opt * np.linalg.inv(cov1) @ est_prior
                    + (1 - omega_opt) * np.linalg.inv(cov2) @ est_sent
                )

                # Save the fused estimate and covariance
                self.estHist[targetID][time_sent] = est_prior
                self.covarianceHist[targetID][time_sent] = cov_prior
                self.trackErrorHist[targetID][time_sent] = self.calcTrackError(
                    est_prior, cov_prior
                )

    def det_of_fused_covariance(
        self, omega: float, cov1: npt.NDArray, cov2: npt.NDArray
    ) -> float:
        """
        Calculate the determinant of the fused covariance matrix.

        Args:
            omega (float): Weight of the first covariance matrix.
            cov1 (np.ndarray): Covariance matrix of the first estimate.
            cov2 (np.ndarray): Covariance matrix of the second estimate.

        Returns:
            float: Determinant of the fused covariance matrix.
        """
        omega = omega[0]  # Ensure omega is a scalar
        P = np.linalg.inv(
            omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2)
        )
        return np.linalg.det(P)


### Event Triggered Estimator Class
class EtEstimator(BaseEstimator):
    def __init__(self, targPriorities, shareWith = None):
        """
        Initialize Event Triggered Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        - shareWith (list): List of satellite objects to share information with.
        """
        super().__init__(targPriorities)

        self.shareWith = shareWith # use this attribute to match common EKF with neighbor
        self.synchronizeFlag = {targetID: defaultdict(dict) for targetID in targPriorities.keys()}  # Flag to synchronize filters
        self.R_factor = 1

        # ET Parameters
        self.delta_alpha = 1
        self.delta_beta = 1
        self.delta = 10


    def et_EKF_initialize(self, target, envTime):
        """
        Event Triggered Extended Kalman Filter initialization step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_initialize(target, envTime)


    def et_EKF_pred(self, targetID, envTime):
        """
        Event Triggered Extended Kalman Filter prediction step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_pred(targetID, envTime)


    def et_EKF_update(self, sats, measurements, targetID, envTime):
        """
        Event Triggered Extended Kalman Filter update step.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_update(sats, measurements, targetID, envTime)


    def event_trigger_processing(self, sat, envTime, comms):
        """
        Event Triggered Estimator processing step for new measurements received from other agents

        Args:
        - sat (object): Satellite object.
        - envTime (float): Current environment time.
        - comms (object): Communications object.

        """
        commNode = comms.G.nodes[sat]

        if envTime in commNode['received_measurements']:
            self.process_new_measurements(sat, envTime, comms)


    def event_trigger_updating(self, sat, envTime, comms):
        """
        Event Triggered Estimator Updating step for new measurments sent to other agents

        Args:
        - sat (object): Satellite object.
        - envTime (float): Current environment time.
        - comms (object): Communications object.
        """

        commNode = comms.G.nodes[sat]

        if envTime in commNode['sent_measurements']:
            self.update_common_filters(sat, envTime, comms)


    def process_new_measurements(self, sat, envTime, comms):
        '''
        This should process new measurements for this satellites commNode and update the local and common filters

        Args:
        - sat (object): Satellite object.
        - envTime (float): Current environment time.
        - comms (object): Communications object.
        '''
        # Get the communcation node for the satellite
        commNode = comms.G.nodes[sat]
        time_sent = max(commNode['received_measurements'].keys()) # Get the newest info on this target
        for targetID in commNode['received_measurements'][time_sent].keys(): # Get a target ID
            for i in range(len(commNode['received_measurements'][time_sent][targetID]['sender'])): # number of messages on this target

                # Get the sender and the message
                sender = commNode['received_measurements'][time_sent][targetID]['sender'][i] # who sent the message
                alpha, beta = commNode['received_measurements'][time_sent][targetID]['meas'][i] # what was the message

                # Get your local EKF
                localEKF = sat.etEstimators[0]

                # Get the commonEKF shared with sender
                commonEKF = None
                for each_etestimator in sat.etEstimators: # find the common filter that is shared with the sender
                    if each_etestimator.shareWith == sender.name:
                        commonEKF = each_etestimator
                        break

                # If your pairwise filters need to be sychronized, do that first
                if commonEKF.synchronizeFlag[targetID][envTime] == True:
                    self.synchronize_filters(sat, sender, targetID, envTime, comms)
                    continue

                # Grab the most recent local prediction for the target
                est_pred = localEKF.estPredHist[targetID][envTime]
                cov_pred = localEKF.covariancePredHist[targetID][envTime]

                # Run Prediction Step on this target for common fitler
                commonEKF.et_EKF_pred(targetID, envTime)

                # Proccess the new measurement from sender with the local and common filter
                measVec_size = 50

                # In-Track Measurement
                if not np.isnan(alpha): # TODO: This is wrong, explicit takes wrong local estimate
                    self.explicit_measurement_update(sat, sender, alpha, 'IT', targetID, envTime, filter = localEKF) # update local filter
                    self.explicit_measurement_update(sat, sender, alpha, 'IT', targetID, envTime, filter = commonEKF) # update our common filter
                else:
                    self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'IT', 'both', targetID, envTime, filters=[localEKF, commonEKF]) # update my local and common filter
                    measVec_size -= 20

                # Cross-Track Measurement
                if not np.isnan(beta):
                    self.explicit_measurement_update(sat, sender, beta, 'CT', targetID, envTime, filter=localEKF) # update local filter
                    self.explicit_measurement_update(sat, sender, beta, 'CT', targetID, envTime, filter=commonEKF) # update our common filter
                else:
                    self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'CT', 'both', targetID, envTime, filters=[localEKF, commonEKF]) # update my local and common filter
                    measVec_size -= 20

                # Calculate Local Track Quaility Metric
                est = localEKF.estHist[targetID][envTime]
                cov = localEKF.covarianceHist[targetID][envTime]
                localEKF.trackErrorHist[targetID][envTime] = localEKF.calcTrackError(est, cov)


                # Calculate Common Track Quaility Metric
                est = commonEKF.estHist[targetID][envTime]
                cov = commonEKF.covarianceHist[targetID][envTime]
                commonEKF.trackErrorHist[targetID][envTime] = commonEKF.calcTrackError(est, cov)

                # comms.used_comm_et_data_values[targetID][sat.name][sender.name][time_sent] = np.array([alpha, beta])
                # comms.used_comm_et_data[targetID][sat.name][sender.name][time_sent] = measVec_size



    def update_common_filters(self, sat, envTime, comms):
        commNode = comms.G.nodes[sat]
        time_sent = max(commNode['sent_measurements'].keys()) # Get the newest info on this target
        for targetID in commNode['sent_measurements'][time_sent].keys():
            for i in range(len(commNode['sent_measurements'][time_sent][targetID]['receiver'])): # number of messages on this target?

                receiver = commNode['sent_measurements'][time_sent][targetID]['receiver'][i]

                localEKF = sat.etEstimators[0]
                commonEKF = None
                for et_estimator in sat.etEstimators:
                    if et_estimator.shareWith == receiver.name:
                        commonEKF = et_estimator
                        break


                est_pred = localEKF.estHist[targetID][envTime]
                cov_pred = localEKF.covarianceHist[targetID][envTime]

                # Run Prediction Step on this target for common fitler
                commonEKF.et_EKF_pred(targetID, envTime)

                # Proccess the new measurement from sender with the local and common filter
                alpha, beta = commNode['sent_measurements'][time_sent][targetID]['meas'][i]
                if not np.isnan(alpha):
                    self.explicit_measurement_update(sat, sat, alpha, 'IT', targetID, envTime, filter=commonEKF) # update our common filter
                else:
                    self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'IT', 'common', targetID, envTime, filters=[localEKF,commonEKF])

                if not np.isnan(beta):
                    self.explicit_measurement_update(sat, sat, beta, 'CT', targetID, envTime, filter=commonEKF) # update our common filter
                else:
                    self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'CT', 'common', targetID, envTime, filters=[localEKF, commonEKF])

                # Calculate Common Track Quaility Metric
                est = commonEKF.estHist[targetID][envTime]
                cov = commonEKF.covarianceHist[targetID][envTime]
                commonEKF.trackErrorHist[targetID][envTime] = commonEKF.calcTrackError(est, cov)


    def explicit_measurement_update(self, sat, sender, measurement, type, targetID, envTime, filter = None):
        '''
        Explicit measurement updates  the estimate with a measurement from a sender satellite.
        Note satellites can send themselves measurements for local filter.

        Args:
        - sat (object): Satellite object that is receiving information.
        - sender (object): Sender satellite object that sent information
        - measurements (np.array): Explicit measurements from the sender satellite: [alpha, 0] or [0, beta]
        - targetID (int): Target ID.
        - envTime (float): Current environment time.

        Returns:
        - Updated estimate and covariance for the target ECI state = [x, vx, y, vy, z, vz].
        '''
        if type == 'IT':
            scalarIdx = 0
        elif type == 'CT':
            scalarIdx = 1

        # The recieving filter most recent Estimate and Covariance
        #prior_time = max(filter.estHist[targetID].keys())
        est_prev = filter.estHist[targetID][envTime]
        P_prev = filter.covarianceHist[targetID][envTime]

        # In Track or Cross Track Measurement actually sent by the sender
        z = measurement

        # The measurement I think you should have got based on assuming our estimates are consistent
        z_pred = np.array(sender.sensor.convert_to_bearings(sender, np.array([est_prev[0], est_prev[2], est_prev[4]])))  # Predicted measurements
        z_pred = z_pred[scalarIdx]

        # The uncertaininty in the measurement you should have got based on my/our estimate
        H = sender.sensor.jacobian_ECI_to_bearings(sender, est_prev)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))

        # Sensor Noise Matrix
        R = sender.sensor.bearingsError[scalarIdx]**2 * self.R_factor # Sensor noise matrix scaled by 1000x

        # Compute innovation
        innovation = z - z_pred

        # Calculate innovation covariance
        innovationCov = (H @ P_prev @ H.T + R)**-1

        # Solve for Kalman gain
        K = np.reshape(P_prev @ H.T * innovationCov, (6,1))

        # Correct prediction
        est = est_prev + np.reshape(K * innovation, (6))

        # Correct covariance
        P = P_prev - K @ H @ P_prev

        # Save Data into filter
        filter.estHist[targetID][envTime] = est
        filter.covarianceHist[targetID][envTime] = P
        filter.trackErrorHist[targetID][envTime] = self.calcTrackError(est, P)


    def implicit_measurement_update(self, sat, sender, local_est_pred, local_P_pred, type, update, targetID, envTime, filters = [None, None]):
        """
        Implicit measurement update function to update the estimate without a measurement.
        Fuses the local estimate with implicit information shared from a paired satellite.

        Args:
        - sat (object): Satellite object that is receiving information.
        - target (object): Target object.
        - envTime (float): Current environment time.

        Returns:
        - np.array: Updated estimate for the target ECI state = [x, vx, y, vy, z, vz].
        """
        if type == 'IT':
            scalarIdx = 0
            delta = self.delta_alpha # Threshold Factor

        elif type == 'CT':
            scalarIdx = 1
            delta = self.delta_beta

        # Grab the best current local Estimate and Covariance
        localEKF = filters[0]
        local_time = max(localEKF.estHist[targetID].keys())
        local_est_curr = localEKF.estHist[targetID][local_time]
        local_cov_curr = localEKF.covarianceHist[targetID][local_time]

        # Grab the shared best local Estimate and Covariance
        commonEKF = filters[1]
        common_time = max(commonEKF.estHist[targetID].keys())
        common_est_curr = commonEKF.estHist[targetID][common_time]
        common_cov_curr = commonEKF.covarianceHist[targetID][common_time]

        # Compute Expected Value of Implicit Measurement
        mu = np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_curr[0], local_est_curr[2], local_est_curr[4]]))) - np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        mu = mu[scalarIdx]

        # Compute Alpha
        alpha = np.array(sender.sensor.convert_to_bearings(sender, np.array([common_est_curr[0], common_est_curr[2], common_est_curr[4]]))) - np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        alpha = alpha[scalarIdx]

        # Compute the Sensor Jacobian
        H = sender.sensor.jacobian_ECI_to_bearings(sender, local_est_curr)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))

        # Compute Sensor Noise Matrix
        R = sender.sensor.bearingsError[scalarIdx]**2 * self.R_factor # Sensor noise matrix scaled by 1000x

        # Define innovation covariance
        Qe = H @ local_P_pred @ H.T + R

        # Compute Expected Value of Implicit Measurement
        vminus = (-delta + alpha - mu)/np.sqrt(Qe)
        vplus = (delta + alpha - mu)/np.sqrt(Qe)

        # Probability Density Function
        phi1 = stats.norm.pdf(vminus)
        phi2 = stats.norm.pdf(vplus)

        # Cumulative Density Function
        Q1 = 1 - stats.norm.cdf(vminus)
        Q2 = 1 - stats.norm.cdf(vplus)

        # Compute Expected Value of Measurement
        zbar = (phi1 - phi2)/(Q1 - Q2) * np.sqrt(Qe)

        # Compute Expected Variance of Implicit Measurement
        nu = ((phi1 - phi2)/(Q1 - Q2))**2 - (vminus * phi1 - vplus * phi2) / (Q1 - Q2)

        # Compute Kalman Gain
        K = np.reshape(local_cov_curr @ H.T * (H @ local_cov_curr @ H.T + R)**-1, (6,1))

        # Update Estimate
        est = local_est_curr + np.reshape(K * zbar, (6))

        # Update Covariance
        cov = local_cov_curr - nu * K @ H @ local_cov_curr

        # Save Data into both filters
        if update == 'both':
            localEKF.estHist[targetID][envTime] = est
            localEKF.covarianceHist[targetID][envTime] = cov
            localEKF.trackErrorHist[targetID][envTime] = self.calcTrackError(est, cov)

            commonEKF.estHist[targetID][envTime] = est
            commonEKF.covarianceHist[targetID][envTime] = cov
            commonEKF.trackErrorHist[targetID][envTime] = self.calcTrackError(est, cov)

        elif update == 'common':
            commonEKF.estHist[targetID][envTime] = est
            commonEKF.covarianceHist[targetID][envTime] = cov
            commonEKF.trackErrorHist[targetID][envTime] = self.calcTrackError(est, cov)

    def event_trigger(self, sat, neighbor, targetID, time):
        """
        Event Trigger function to determine if an explict or implicit
        measurement update is needed.

        Args:
        - sat (object): Satellite object that is receiving information.
        - target (object): Target object.

        Returns:
        - send_alpha (float): Alpha value to send to neighbor - NaN if not needed.
        - send_beta (float): Beta value to send to neighbor - NaN if not needed.
        """
        # Get the most recent measurement on the target
        alpha, beta = sat.measurementHist[targetID][time]

        # Get my commonEKF with this neighbor
        commonEKF = None
        for each_etEstimator in sat.etEstimators:
            if each_etEstimator.shareWith == neighbor.name:
                commonEKF = each_etEstimator
                break

        # Get neighbors commonEKF with me
        neighbor_commonEKF = None
        for each_etEstimator in neighbor.etEstimators:
            if each_etEstimator.shareWith == sat.name:
                neighbor_commonEKF = each_etEstimator
                break


        commonEKF.synchronizeFlag[targetID][time] = True
        neighbor_commonEKF.synchronizeFlag[targetID][time] = True
        # Search backwards through dictionary to check if there are 5 measurements sent to this neighbor
        count = 2
        for lastTime in reversed(list(sat.measurementHist[targetID].keys())): # starting now, go back in time
            if isinstance(sat.measurementHist[targetID][lastTime], np.ndarray): # if the satellite took a measurement at this time
                count -= 1 # increment count

            if count == 0: # if there are 5 measurements sent to this neighbor, no need to synchronize
                commonEKF.synchronizeFlag[targetID][time] = False
                neighbor_commonEKF.synchronizeFlag[targetID][time] = False
                break # break out of loop

        # Predict the common estimate to the current time a measurement was taken
        if len(commonEKF.estHist[targetID]) == 1:
            return alpha, beta

        commonEKF.et_EKF_pred(targetID, time)

        # Get the most recent estimate and covariance
        pred_est = commonEKF.estHist[targetID][time]
        pred_cov = commonEKF.covarianceHist[targetID][time]

        # Predict the Measurement that the neighbor would think I made
        pred_alpha, pred_beta = sat.sensor.convert_to_bearings(sat, np.array([pred_est[0], pred_est[2], pred_est[4]]))

        # Compute the Innovation
        innovation = np.array([alpha, beta]) - np.array([pred_alpha, pred_beta])

        # Compute the Innovation Covariance
        H = sat.sensor.jacobian_ECI_to_bearings(sat, pred_est)

        # Compute the Sensor Noise Matrix
        R =  np.eye(2) * sat.sensor.bearingsError

        # Compute the Innovation Covariance
        innovationCov = H @ pred_cov @ H.T + R

        # Event Trigger
        et = innovation.T @ np.linalg.inv(innovationCov) @ innovation

        # Is my measurment surprising based on the innovation covariance?
        send_alpha = np.nan
        send_beta = np.nan

        if(et > self.delta):
            send_alpha = alpha
            send_beta = beta

        return send_alpha, send_beta


    def synchronize_filters(self, sat, neighbor, targetID, envTime, comms):
        """
        Synchronize the local and common filters between two agents.

        Args:
        - sat (object): Satellite object that is receiving information.
        - neighbor (object): Neighbor satellite object.
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        """
        ### Try just synchronizing the common information filters

        # Sat common information filter with neighbor
        localEKF = sat.etEstimators[0]
        local_time = max(localEKF.estHist[targetID].keys())
        local_est = localEKF.estHist[targetID][local_time]
        local_cov = localEKF.covarianceHist[targetID][local_time]

        # Neighbor common information filter with sat
        neighbor_localEKF = neighbor.etEstimators[0]
        neighbor_time = max(neighbor_localEKF.estHist[targetID].keys())
        neighbor_est = neighbor_localEKF.estHist[targetID][neighbor_time]
        neighbor_cov = neighbor_localEKF.covarianceHist[targetID][neighbor_time]

        omega_opt = optimize.minimize(self.det_of_fused_covariance, [0.5], args=(local_cov, neighbor_cov), bounds=[(0, 1)]).x
        cov_fused = np.linalg.inv(omega_opt * np.linalg.inv(local_cov) + (1 - omega_opt) * np.linalg.inv(neighbor_cov))
        est_fused = cov_fused @ (omega_opt * np.linalg.inv(local_cov) @ local_est + (1 - omega_opt) * np.linalg.inv(neighbor_cov) @ neighbor_est)

        # Find commonEKFs
        for each_etEstimator in sat.etEstimators:
            if each_etEstimator.shareWith == neighbor.name:
                local_commonEKF = each_etEstimator
                break

        for each_etEstimator in neighbor.etEstimators:
            if each_etEstimator.shareWith == sat.name:
                neighbor_commonEKF = each_etEstimator
                break

        # Save the result to local and common EKFs
        localEKF.estHist[targetID][envTime] = est_fused
        localEKF.covarianceHist[targetID][envTime] = cov_fused
        localEKF.trackErrorHist[targetID][envTime] = localEKF.calcTrackError(est_fused, cov_fused)

        local_commonEKF.estHist[targetID][envTime] = est_fused
        local_commonEKF.covarianceHist[targetID][envTime] = cov_fused
        local_commonEKF.trackErrorHist[targetID][envTime] = local_commonEKF.calcTrackError(est_fused, cov_fused)

        neighbor_localEKF.estHist[targetID][envTime] = est_fused
        neighbor_localEKF.covarianceHist[targetID][envTime] = cov_fused
        neighbor_localEKF.trackErrorHist[targetID][envTime] = neighbor_localEKF.calcTrackError(est_fused, cov_fused)

        neighbor_commonEKF.estHist[targetID][envTime] = est_fused
        neighbor_commonEKF.covarianceHist[targetID][envTime] = cov_fused
        neighbor_commonEKF.trackErrorHist[targetID][envTime] = neighbor_commonEKF.calcTrackError(est_fused, cov_fused)



    def det_of_fused_covariance(self, omega, cov1, cov2):
        """
        Calculate the determinant of the fused covariance matrix.

        Args:
            omega (float): Weight of the first covariance matrix.
            cov1 (np.ndarray): Covariance matrix of the first estimate.
            cov2 (np.ndarray): Covariance matrix of the second estimate.

        Returns:
            float: Determinant of the fused covariance matrix.
        """
        omega = omega[0]  # Ensure omega is a scalar
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)


### Ground Station Estimator Class
class GsEstimator(BaseEstimator):
    def __init__(self, targetPriorities: dict[int, int]):
        """
        Initialize Ground Station Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetPriorities)

        self.R_factor = 1

    def gs_EKF_initialize(self, target: target.Target, envTime: float) -> None:
        """
        Ground Station Extended Kalman Filter initialization step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_initialize(target, envTime)

    def gs_EKF_pred(self, targetID: int, envTime: float) -> None:
        """
        Ground Station Extended Kalman Filter prediction step.

        Args:
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_pred(targetID, envTime)

    def gs_EKF_update(
        self,
        sats: list['satellite.Satellite'],
        measurements,
        targetID: int,
        envTime: float,
    ) -> None:
        """
        Ground Station Extended Kalman Filter update step.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.
        """
        super().EKF_update(sats, measurements, targetID, envTime)

    def gs_CI(
        self,
        targetID: int,
        est_sent: npt.NDArray,
        cov_sent: npt.NDArray,
        time_sent: float,
    ) -> None:
        """
        Covaraince interesection function to conservatively combine received estimates and covariances on a target.

        Args:
        - targetID (int): Target ID.
        - est_sent (np.ndarray): Sent estimate.
        - cov_sent (np.ndarray): Sent covariance.
        - time_sent (float): Time the estimate was sent.
        """

        # First, check does the estimator already have a est and cov for this target?
        if not self.estHist[targetID] and not self.covarianceHist[targetID]:
            # If not, use the sent estimate and covariance to initialize
            self.estHist[targetID][time_sent] = est_sent
            self.covarianceHist[targetID][time_sent] = cov_sent
            self.trackErrorHist[targetID][time_sent] = self.calcTrackError(est_sent, cov_sent)
            return

        # Now, if the estimator already has an estimate and covariance for this target, check if we should CI
        time_prior = max(self.estHist[targetID].keys())

        # If the send time is older than the prior estimate, discard the sent estimate
        if time_sent < time_prior:
            return

        if time_sent - time_prior > 5:
            self.estHist[targetID][time_sent] = est_sent
            self.covarianceHist[targetID][time_sent] = cov_sent
            self.trackErrorHist[targetID][time_sent] = self.calcTrackError(est_sent, cov_sent)
            return

        # Else do CI
        est_prior = self.estHist[targetID][time_prior]
        cov_prior = self.covarianceHist[targetID][time_prior]

        # Propagate the prior estimate and covariance to the new time
        dt = time_sent - time_prior
        est_prior = self.state_transition(est_prior, dt)
        F = self.state_transition_jacobian(est_prior, dt)
        cov_prior = np.dot(F, np.dot(cov_prior, F.T))

        # Minimize the covariance determinant
        omega_opt = optimize.minimize(self.det_of_fused_covariance, [0.5], args=(cov_prior, cov_sent), bounds=[(0, 1)]).x

        # Compute the fused covariance
        cov1 = cov_prior
        cov2 = cov_sent
        cov_prior = np.linalg.inv(omega_opt * np.linalg.inv(cov1) + (1 - omega_opt) * np.linalg.inv(cov2))
        est_prior = cov_prior @ (omega_opt * np.linalg.inv(cov1) @ est_prior + (1 - omega_opt) * np.linalg.inv(cov2) @ est_sent)

        # Save the fused estimate and covariance
        self.estHist[targetID][time_sent] = est_prior
        self.covarianceHist[targetID][time_sent] = cov_prior
        self.trackErrorHist[targetID][time_sent] = self.calcTrackError(est_prior, cov_prior)

    def det_of_fused_covariance(self, omega, cov1, cov2):
        """
        Calculate the determinant of the fused covariance matrix.

        Args:
            omega (float): Weight of the first covariance matrix.
            cov1 (np.ndarray): Covariance matrix of the first estimate.
            cov2 (np.ndarray): Covariance matrix of the second estimate.

        Returns:
            float: Determinant of the fused covariance matrix.
        """
        omega = omega[0] # Ensure omega is a scalar
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)
