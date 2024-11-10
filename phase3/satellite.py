import numpy as np
import pandas as pd
from astropy import units as u
from numpy import typing as npt

from common import dataclassframe
from phase3 import collection
from phase3 import comms
from phase3 import estimator
from phase3 import orbit
from phase3 import sensor
from phase3 import target


class Satellite:
    def __init__(
        self,
        name: str,
        orbit: orbit.Orbit,
        color: str,
    ) -> None:
        self.name = name
        self.color = color

        self.__network = None

        self._orbit_params = orbit
        self.orbit = self._orbit_params.to_poliastro()

        self._target_intents: dict[int, int] = {}
        self._targetIDs: list[int] = []

        self._state_hist = dataclassframe.DataClassFrame(clz=collection.State)

        self._sensor: sensor.Sensor | None = None

    @property
    def pos(self) -> npt.NDArray:
        return self.orbit.r.to_value(u.km)

    @property
    def _network(self) -> comms.Comms:
        assert self.__network is not None
        return self.__network

    def initialize(self, network: comms.Comms) -> None:
        self.__network = network

    def _get_neighbors(self) -> list[str]:
        return self._network.get_neighbors(self.name)

    def propagate(self, time_step: u.Quantity[u.s], time: float):
        """
        Propagate the satellite's orbit forward in time by the given time step.

        Args:
            time_step (u.Quantity[u.s]): Time step to propagate the orbit by.
            time (float): Current time.
        """

        # Propagate the orbit
        self.orbit = self.orbit.propagate(time_step)

        # Save into the data frame
        self._state_hist.append(
            collection.State(
                time=time,
                x=self.orbit.r.value[0],
                y=self.orbit.r.value[1],
                z=self.orbit.r.value[2],
                vx=self.orbit.v.value[0],
                vy=self.orbit.v.value[1],
                vz=self.orbit.v.value[2],
            )
        )

    def get_projection_box(self) -> npt.NDArray | None:
        """
        Get the projection box of the sensor.

        Returns:
            The projection box of the sensor if it exists, otherwise None.
        """
        if self._sensor is not None:
            return self._sensor.get_projection_box(self.orbit)


class SensingSatellite(Satellite):
    def __init__(self, *args, sensor: sensor.Sensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounty = (
            {}
        )  # targetID: satID (name), who this sat should talk to if they see the target
        self._sensor = sensor

        # Data frame for measurements
        self._measurement_hist = dataclassframe.DataClassFrame(
            clz=collection.Measurement
        )

    def collect_measurements(
        self, target_id: int, target_ground_truth_pos: npt.NDArray, time: float
    ) -> None:
        """
        Collect measurements from the sensor for a specified target.

        The satellite will use its sensor class to collect a measurement on the target.
        It then stores the measurement in its measurement history.

        Args:
            target_ground_truth_pos: Ground truth position of the target (used for simulating the measurement).

        Returns:
            Flag indicating whether measurements were successfully collected or not.
        """
        assert self._sensor is not None

        # Check if this satellite can see the target, if so: get the measurement, if not: return False
        measurement = self._sensor.get_measurement(self.orbit, target_ground_truth_pos)

        # If the measurement is an np.ndarray of in-track, cross-track measurements
        if measurement is not None:

            # Get the [x, y, z, vx, vy, vz] sat state
            sat_state = np.array(
                [
                    self.orbit.r.value[0],
                    self.orbit.r.value[1],
                    self.orbit.r.value[2],
                    self.orbit.v.value[0],
                    self.orbit.v.value[1],
                    self.orbit.v.value[2],
                ]
            )

            self._measurement_hist.append(
                collection.Measurement(
                    target_id=target_id,
                    time=time,
                    alpha=measurement[0],
                    beta=measurement[1],
                    sat_state=sat_state,
                    meas_noise=self._sensor.R,
                )
            )

    def send_meas_to_fusion(self, target_id: int, time: float) -> None:
        """
        Send measurements from the sensing satellites to the fusion satellites.
        """

        # Get all measurements for this target_id
        measurements = self.get_measurements(target_id=target_id, time=time)

        if measurements:
            # Check, does this sat have custody of this target?
            if target_id in self.bounty:
                # If so, send the measurements to the fusion satellite
                sat_id = self.bounty[target_id]
                self._network.send_measurements_path(
                    measurements,
                    self.name,
                    sat_id,
                    time=time,
                    size=50 * len(measurements),
                )
            else:
                print(f"Sat {self.name} does not have a bounty for target {target_id}")
                # Just send to nearest fusion satellite

                neighbors = self._get_neighbors()
                nearest_fusion_sat = min(
                    neighbors, key=lambda x: self._network.get_distance(self.name, x)
                )
                self._network.send_measurements_path(
                    measurements,
                    self.name,
                    nearest_fusion_sat,
                    time=time,
                    size=50 * len(measurements),
                )

    def get_measurements(
        self, target_id: int, time: float | None = None
    ) -> list[collection.Measurement]:
        """
        Get all measurements for a specified target.

        Args:
            target_id: The target ID to get measurements for.
            time: The time at which to get measurements. If None, all measurements are returned.

        Returns:
            A list of measurements for the target.
        """
        if time is not None:
            df = self._measurement_hist.loc[
                (self._measurement_hist['target_id'] == target_id)
                & (self._measurement_hist['time'] == time)
            ]
        else:
            df = self._measurement_hist.loc[
                self._measurement_hist['target_id'] == target_id
            ]
        return self._measurement_hist.to_dataclasses(df)


class FusionSatellite(Satellite):
    def __init__(
        self, *args, local_estimator: estimator.BaseEstimator | None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.custody = {}  # targetID: Boolean, who this sat has custody of
        self._estimator = local_estimator
        self._et_estimators: list['estimator.EtEstimator'] = []

    def process_measurements(self, time: float) -> None:
        """
        Process the measurements from the fusion satellite.
        """

        # Find the set of measurements that were sent to this fusion satellite at this time step (use self._network.measurements data frame)
        data_received = self._network.receive_measurements(self.name, time)

        if not data_received:
            return

        # Get unique target IDs from received data
        target_ids = {meas.target_id for meas in data_received}

        for target_id in target_ids:
            # Get all measurements for this targetID
            meas_for_target = [
                meas for meas in data_received if meas.target_id == target_id
            ]

            # Send the measurements to the estimator
            self._estimator.EKF_predict(meas_for_target)
            self._estimator.EKF_update(meas_for_target, [self])

            # Also, figure out if custody of this target needs to be updated?
            test = 1

            # im here, do logic for custody

    def update_estimator(
        self, measurement: npt.NDArray, target: target.Target, time: float
    ) -> None:  # TODO: fix this such that dont require target
        """Update the independent estimator for the satellite.

        The satellite will update its independent estimator using the measurement provided.
        This will call the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            measurement (object): Measurement data obtained from the sensor.
            target (object): Target object containing targetID and other relevant information.
            time (float): Current time at which the measurement is taken.
        """
        # This assertion checks that the independent estimator exists
        # It raises an AssertionError if self.indeptEstimator is None
        assert self._estimator is not None, 'Independent estimator is not initialized'
        target_id = target.target_id

        # Predict step will initialize if needed
        self._estimator.EKF_pred(measurement, target_id, time)

        # Update with measurement if we have one
        if measurement is not None:
            self._estimator.EKF_update([self], [measurement], target.target_id, time)

    def filter_CI(self, data_received: pd.DataFrame) -> None:
        """
        Update the satellite estimator using covariance intersection data that was sent to it.
        """

        # Use the estimator.CI function to update the estimator with any data recieved, at that time step
        # Want to only fuse data that is newer than the latest estimate the estimator has on that target

        for targetID in self._targetIDs:

            # Get the latest estimate time for this target

            # Does the estimator have any data?
            if not self._estimator.estimation_data.empty:
                # Does the targetID exist in the estimator?
                if targetID in self._estimator.estimation_data['targetID'].values:
                    latest_estimate_time = self._estimator.estimation_data[
                        self._estimator.estimation_data['targetID'] == targetID
                    ]['time'].max()
                else:
                    # If the targetID does not exist in the estimator, then the latest estimate time is negative infinity
                    latest_estimate_time = float('-inf')
            else:
                # If the estimator is empty, then the latest estimate time is negative infinity
                latest_estimate_time = float('-inf')

            # Get all data received for this target
            data_for_target = data_received[data_received['targetID'] == targetID]

            # Now, loop through all data for the target, and only fuse data that is newer than the latest estimate
            for _, row in data_for_target.iterrows():
                if row['time'] >= latest_estimate_time:
                    self._estimator.CI(
                        targetID, row['data'][0], row['data'][1], row['time']
                    )
