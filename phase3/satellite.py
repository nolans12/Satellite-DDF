from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy import units as u
from poliastro import bodies
from poliastro import twobody

if TYPE_CHECKING:
    from phase3 import estimator
    from phase3 import sensor

from phase3 import orbit
from phase3 import target

## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit


class Satellite:
    def __init__(
        self,
        orbit: orbit.Orbit,
        sensor: 'sensor.Sensor',
        name: str,
        color: str,
    ):
        """Initialize a Satellite object.

        Args:
            a: Semi-major axis of the satellite's orbit.
            ecc: Eccentricity of the satellite's orbit.
            inc: Inclination of the satellite's orbit in degrees.
            raan: Right ascension of the ascending node in degrees.
            argp: Argument of periapsis in degrees.
            nu: True anomaly in degrees.
            sensor: Sensor used by the satellite.
            name: Name of the satellite.
            color: Color of the satellite for visualization.
        """

        # Sensor to use
        self.sensor = sensor

        # Other parameters
        self.name = name
        self.color = color
        self._orbit_params = orbit

        # Set the estimators to None on initalization
        self.indeptEstimator: 'estimator.IndeptEstimator | None' = None
        self.ciEstimator: 'estimator.CiEstimator | None' = None
        self.etEstimators: list['estimator.EtEstimator'] = []

        self.targPriority: dict[int, int] = {}
        self.targetIDs: list[int] = []
        self.measurementHist: dict[int, defaultdict] = {}

        # Create the poliastro orbit
        self.orbit = self._orbit_params.to_poliastro()
        self.orbitHist = defaultdict(dict)  # contains time and xyz of orbit history
        self.velHist = defaultdict(dict)  # contains time and xyz of velocity history
        self.time = 0

        self.stateHist = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    def propagate(self, time_step: u.Quantity[u.s], time: float):

        # Update the time
        self.time = time

        # Propagate the orbit
        self.orbit = self.orbit.propagate(time_step)

        # Save into the data frame
        new_row = pd.DataFrame(
            {
                'time': [self.time],
                'x': [self.orbit.r.value[0]],
                'y': [self.orbit.r.value[1]],
                'z': [self.orbit.r.value[2]],
                'vx': [self.orbit.v.value[0]],
                'vy': [self.orbit.v.value[1]],
                'vz': [self.orbit.v.value[2]],
            }
        )
        self.stateHist = pd.concat([self.stateHist, new_row], ignore_index=True)

    def collect_measurements_and_filter(self, target: target.Target) -> bool:
        """
        Collect measurements from the sensor for a specified target and update local filters.
        The satellite will use its sensor class to collect a measurement on the target.
        It then stores the measurement in its measurement history and updates its local filters.
        Updating the local filters calls the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            target: Target object containing targetID and other relevant information.

        Returns:
            Flag indicating whether measurements were successfully collected (1) or not (0).
        """

        # Assume no measurement is collected on this target
        collectedFlag = False

        # Check if this satellite can see the target, if so: get the measurement, if not: return False
        if target.targetID in self.targetIDs:

            measurement = self.sensor.get_measurement(self, target)

            # If the measurement is an np.ndarray of in-track, cross-track measurements
            if isinstance(measurement, np.ndarray):
                collectedFlag = True

                # Save the measurement
                self.measurementHist[target.targetID][self.time] = measurement

                # Update the local filters with predictions and measurement if a measurement is collected
                if self.indeptEstimator:
                    self.update_indept_estimator(measurement, target, self.time)

                if self.ciEstimator:
                    self.update_ci_estimator(measurement, target, self.time)

            if self.etEstimators:
                self.update_et_estimator(measurement, target, self.time)

        return collectedFlag

    def update_indept_estimator(
        self, measurement, target: target.Target, time: float
    ) -> None:
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
        assert (
            self.indeptEstimator is not None
        ), "Independent estimator is not initialized"
        targetID = target.targetID

        if self.indeptEstimator.estimation_data.empty:
            # The estimator contains zero data in it (first target)
            self.indeptEstimator.local_EKF_initialize(target, time)
        else:
            # Check, does the targetID already exist in the estimator?
            if targetID in self.indeptEstimator.estimation_data['targetID'].values:
                # If estimate exists, predict and update
                self.indeptEstimator.local_EKF_pred(targetID, time)
                self.indeptEstimator.local_EKF_update(
                    [self], [measurement], targetID, time
                )
            else:
                # If no estimate exists, initialize
                self.indeptEstimator.local_EKF_initialize(target, time)

    def update_ci_estimator(
        self, measurement, target: target.Target, time: float
    ) -> None:
        """Update the DDF estimator for the satellite.

        The satellite will update its DDF estimator using the measurement provided.
        This will call the DDF functions to update the state and covariance estimates based on the measurement.

        Args:
            measurement (object): Measurement data obtained from the sensor.
            target (object): Target object containing targetID and other relevant information.
            time (float): Current time at which the measurement is taken.
        """
        assert self.ciEstimator is not None
        targetID = target.targetID

        if (
            len(self.ciEstimator.estHist[target.targetID]) < 1
        ):  # if this is an empty dictionary, initialize the filter
            self.ciEstimator.ci_EKF_initialize(target, time)
            return

        # Update the DDF estimator using the measurement
        self.ciEstimator.ci_EKF_pred(targetID, time)
        self.ciEstimator.ci_EKF_update([self], [measurement], targetID, time)

    def update_et_estimator(
        self, measurement, target: target.Target, time: float
    ) -> None:
        """Update the ET filters for the satellite.

        The satellite will update its ET filters using the measurement provided.
        This will call the ET functions to update the state and covariance estimates based on the measurement.

        Args:
            measurement (object): Measurement data obtained from the sensor.
            target (object): Target object containing targetID and other relevant information.
            time (float): Current time at which the measurement is taken.
        """
        targetID = target.targetID

        # Update the ET filters using the ET estimator
        local_et_estimator = self.etEstimators[
            0
        ]  # get the local et estimator for this satellite

        if len(local_et_estimator.estHist[targetID]) < 1:
            local_et_estimator.et_EKF_initialize(target, time)
            return

        local_et_estimator.et_EKF_pred(targetID, time)
        local_et_estimator.et_EKF_update([self], [measurement], targetID, time)
