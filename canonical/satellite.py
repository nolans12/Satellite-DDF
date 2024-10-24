from astropy import units as u
from numpy import typing as npt

# from canonical import estimator  # TODO: Doesn't exist
from canonical import comms
from canonical import orbit
from canonical import sensor


class Satellite:
    def __init__(
        self,
        name: str,
        orbit: orbit.Orbit,
        local_estimator: 'estimator.BaseEstimator',
        color: str,
    ) -> None:
        self.name = name
        self.color = color

        self.__network = None

        self._orbit_params = orbit
        self.orbit = self._orbit_params.to_poliastro()

        self._estimator = local_estimator
        self._et_estimators: list['estimator.EtEstimator'] = []

        self._target_intents: dict[int, int] = {}

        self._orbit_hist: dict[float, npt.NDArray] = {}
        self._vel_hist: dict[float, npt.NDArray] = {}

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

    def update_estimator(
        self, measurement: npt.NDArray | None, target_id: int, time: float
    ) -> None:
        """Update the local estimator for the satellite.

        The satellite will update its estimator using the measurement provided.
        This will call the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            measurement: Measurement data obtained from the sensor.
            target_id: Target object containing targetID and other relevant information.
            time: Current time at which the measurement is taken.
        """
        # Update the independent estimator using the measurement
        self._estimator.predict(target_id, time)
        # TODO: Fix this line
        self._estimator.update([self], [measurement], target_id, time)


class FusionSatellite(Satellite):
    pass


class SensingSatellite(Satellite):
    def __init__(self, *args, sensor: sensor.Sensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._sensor = sensor

        # Target ID -> Time -> Measurement
        self._measurement_hist: dict[int, dict[float, npt.NDArray]] = {}

    def collect_measurements_and_filter(
        self, target_id: int, target_ground_truth_pos: npt.NDArray, time: float
    ) -> bool:
        """
        Collect measurements from the sensor for a specified target and update local filters.

        The satellite will use its sensor class to collect a measurement on the target.
        It then stores the measurement in its measurement history and updates its local filters.
        Updating the local filters calls the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            target_ground_truth_pos: Ground truth position of the target (used for simulating the measurement).

        Returns:
            Flag indicating whether measurements were successfully collected or not.
        """
        collected_measurement = False

        # Check if this satellite can see the target, if so: get the measurement, if not: return False
        measurement = self._sensor.get_measurement(self.orbit, target_ground_truth_pos)

        # If the measurement is an np.ndarray of in-track, cross-track measurements
        if measurement is not None:
            collected_measurement = True

            # Save the measurement
            self._measurement_hist[target_id][time] = measurement

        self.update_estimator(measurement, target_id, time)

        return collected_measurement
