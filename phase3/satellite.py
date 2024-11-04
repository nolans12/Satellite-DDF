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


class SensingSatellite(Satellite):
    def __init__(self, *args, sensor: sensor.Sensor, **kwargs):
        super().__init__(*args, **kwargs)
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

        # Check if this satellite can see the target, if so: get the measurement, if not: return False
        measurement = self._sensor.get_measurement(self.orbit, target_ground_truth_pos)

        # If the measurement is an np.ndarray of in-track, cross-track measurements
        if measurement is not None:
            self._measurement_hist.append(
                collection.Measurement(
                    target_id=target_id,
                    time=time,
                    alpha=measurement[0],
                    beta=measurement[1],
                )
            )


class FusionSatellite(Satellite):
    def __init__(self, *args, local_estimator: 'estimator.BaseEstimator', **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator = local_estimator
        self._et_estimators: list['estimator.EtEstimator'] = []

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
        assert self._estimator is not None, "Independent estimator is not initialized"
        target_id = target.target_id

        if self._estimator.estimation_data.empty:
            # The estimator contains zero data in it (first target)
            self._estimator.EKF_initialize(target, time)
        else:
            # Check, does the targetID already exist in the estimator?
            if target_id in self._estimator.estimation_data['targetID'].values:
                # If estimate exists, predict and update
                self._estimator.EKF_pred(target_id, time)
                self._estimator.EKF_update([self], [measurement], target_id, time)
            else:
                # If no estimate exists, initialize
                self._estimator.EKF_initialize(target, time)

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
