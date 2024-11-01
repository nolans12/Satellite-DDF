import pandas as pd
from astropy import units as u
from numpy import typing as npt

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

        self._state_hist = pd.DataFrame(
            columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        )

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
        new_row = pd.DataFrame(
            {
                'time': [time],
                'x': [self.orbit.r.value[0]],
                'y': [self.orbit.r.value[1]],
                'z': [self.orbit.r.value[2]],
                'vx': [self.orbit.v.value[0]],
                'vy': [self.orbit.v.value[1]],
                'vz': [self.orbit.v.value[2]],
            }
        )
        self._state_hist = pd.concat([self._state_hist, new_row], ignore_index=True)


class SensingSatellite(Satellite):
    def __init__(self, *args, sensor: sensor.Sensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._sensor = sensor

        # Data frame for measurements
        self._measurement_hist = pd.DataFrame(
            columns=['time', 'targetID', 'measurement']
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

            # Save the measurement
            self._measurement_hist = pd.concat(
                [
                    self._measurement_hist,
                    pd.DataFrame(
                        {
                            'time': [time],
                            'targetID': [target_id],
                            'measurement': [measurement],
                        }
                    ),
                ],
                ignore_index=True,
            )


class FusionSatellite(Satellite):
    def __init__(self, *args, local_estimator: 'estimator.BaseEstimator', **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator = local_estimator
        self._et_estimators: list['estimator.EtEstimator'] = []

    def update_estimator(
        self, measurement, target: target.Target, time: float
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
        target_id = target.targetID

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


# class Satellite:
# def __init__(
#     self,
#     orbit: orbit.Orbit,
#     sensor: 'sensor.Sensor',
#     name: str,
#     color: str,
# ):
#     """Initialize a Satellite object.

#     Args:
#         a: Semi-major axis of the satellite's orbit.
#         ecc: Eccentricity of the satellite's orbit.
#         inc: Inclination of the satellite's orbit in degrees.
#         raan: Right ascension of the ascending node in degrees.
#         argp: Argument of periapsis in degrees.
#         nu: True anomaly in degrees.
#         sensor: Sensor used by the satellite.
#         name: Name of the satellite.
#         color: Color of the satellite for visualization.
#     """

#     # Sensor to use
#     self.sensor = sensor

#     # Other parameters
#     self.name = name
#     self.color = color
#     self._orbit_params = orbit

#     # Set the estimators to None on initalization
#     self.estimator: 'estimator.Estimator | None' = None

#     self.targPriority: dict[int, int] = {}
#     self.targetIDs: list[int] = []

#     # Create the poliastro orbit
#     self.orbit = self._orbit_params.to_poliastro()
#     self.orbitHist = defaultdict(dict)  # contains time and xyz of orbit history
#     self.velHist = defaultdict(dict)  # contains time and xyz of velocity history
#     self.time = 0

#     self.stateHist = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
#     self.measurementHist = pd.DataFrame(columns=['time', 'targetID', 'measurement'])

# def propagate(self, time_step: u.Quantity[u.s], time: float):
#     """
#     Propagate the satellite's orbit forward in time by the given time step.

#     Args:
#         time_step (u.Quantity[u.s]): Time step to propagate the orbit by.
#         time (float): Current time.
#     """

#     # Update the time
#     self.time = time

#     # Propagate the orbit
#     self.orbit = self.orbit.propagate(time_step)

#     # Save into the data frame
#     new_row = pd.DataFrame(
#         {
#             'time': [self.time],
#             'x': [self.orbit.r.value[0]],
#             'y': [self.orbit.r.value[1]],
#             'z': [self.orbit.r.value[2]],
#             'vx': [self.orbit.v.value[0]],
#             'vy': [self.orbit.v.value[1]],
#             'vz': [self.orbit.v.value[2]],
#         }
#     )
#     self.stateHist = pd.concat([self.stateHist, new_row], ignore_index=True)

# def collect_measurements_and_filter(self, targets: list[target.Target]) -> bool:
# """
# For all targets, collect avaliable measurements and update the estimator.

# Args:
#     targets: List of Target objects containing targetID and the state (so can take noisy measurement)
# """

# # Check if this satellite can see the target, if so: get the measurement, if not: return False
# for target in targets:
#     if target.targetID in self.targetIDs:  # Told you should track this target
#         measurement = self.sensor.get_measurement(self, target)

#         # Save this measurement inot the data drame (even if it is None)
#         new_row = pd.DataFrame(
#             {
#                 'time': [self.time],
#                 'targetID': [target.targetID],
#                 'measurement': [measurement],
#             }
#         )
#         self.measurementHist = pd.concat(
#             [self.measurementHist, new_row], ignore_index=True
#         )

#         # Update the local filters with predictions and measurement if a measurement is collected
#         if self.estimator and measurement is not None:
#             self.update_estimator(measurement, target, self.time)


# def update_estimator(self, measurement, target: target.Target, time: float) -> None:
# """Update the independent estimator for the satellite.

# The satellite will update its independent estimator using the measurement provided.
# This will call the EKF functions to update the state and covariance estimates based on the measurement.

# Args:
#     measurement (object): Measurement data obtained from the sensor.
#     target (object): Target object containing targetID and other relevant information.
#     time (float): Current time at which the measurement is taken.
# """
# # This assertion checks that the independent estimator exists
# # It raises an AssertionError if self.indeptEstimator is None
# assert self.estimator is not None, "Independent estimator is not initialized"
# targetID = target.targetID

# if self.estimator.estimation_data.empty:
#     # The estimator contains zero data in it (first target)
#     self.estimator.EKF_initialize(target, time)
# else:
#     # Check, does the targetID already exist in the estimator?
#     if targetID in self.estimator.estimation_data['targetID'].values:
#         # If estimate exists, predict and update
#         self.estimator.EKF_pred(targetID, time)
#         self.estimator.EKF_update([self], [measurement], targetID, time)
#     else:
#         # If no estimate exists, initialize
#         self.estimator.EKF_initialize(target, time)

# def filter_CI(self, data_received: pd.DataFrame) -> None:
# """
# Update the satellite estimator using covariance intersection data that was sent to it.
# """

# # Use the estimator.CI function to update the estimator with any data recieved, at that time step
# # Want to only fuse data that is newer than the latest estimate the estimator has on that target

# for targetID in self.targetIDs:

#     # Get the latest estimate time for this target

#     # Does the estimator have any data?
#     if not self.estimator.estimation_data.empty:
#         # Does the targetID exist in the estimator?
#         if targetID in self.estimator.estimation_data['targetID'].values:
#             latest_estimate_time = self.estimator.estimation_data[
#                 self.estimator.estimation_data['targetID'] == targetID
#             ]['time'].max()
#         else:
#             # If the targetID does not exist in the estimator, then the latest estimate time is negative infinity
#             latest_estimate_time = float('-inf')
#     else:
#         # If the estimator is empty, then the latest estimate time is negative infinity
#         latest_estimate_time = float('-inf')

#     # Get all data received for this target
#     data_for_target = data_received[data_received['targetID'] == targetID]

#     # Now, loop through all data for the target, and only fuse data that is newer than the latest estimate
#     for _, row in data_for_target.iterrows():
#         if row['time'] >= latest_estimate_time:
#             self.estimator.CI(
#                 targetID, row['data'][0], row['data'][1], row['time']
#             )

# # def update_et_estimator(
# #     self, measurement, target: target.Target, time: float
# # ) -> None:
# #     """Update the ET filters for the satellite.

# #     The satellite will update its ET filters using the measurement provided.
# #     This will call the ET functions to update the state and covariance estimates based on the measurement.

# #     Args:
# #         measurement (object): Measurement data obtained from the sensor.
# #         target (object): Target object containing targetID and other relevant information.
# #         time (float): Current time at which the measurement is taken.
# #     """
# #     targetID = target.targetID

# #     # Update the ET filters using the ET estimator
# #     local_et_estimator = self.etEstimators[
# #         0
# #     ]  # get the local et estimator for this satellite

# #     if len(local_et_estimator.estHist[targetID]) < 1:
# #         local_et_estimator.et_EKF_initialize(target, time)
# #         return

# #     local_et_estimator.et_EKF_pred(targetID, time)
# #     local_et_estimator.et_EKF_update([self], [measurement], targetID, time)
