from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from poliastro import bodies
from poliastro import twobody

if TYPE_CHECKING:
    from phase3 import estimator
    from phase3 import sensor

from phase3 import target

## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit


class Satellite:
    def __init__(
        self,
        a: float,
        ecc: float,
        inc: float,
        raan: float,
        argp: float,
        nu: float,
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

        # Set the estimators to None on initalization
        self.indeptEstimator: 'estimator.IndeptEstimator | None' = None
        self.ciEstimator: 'estimator.CiEstimator | None' = None
        self.etEstimators: list['estimator.EtEstimator'] | None = None

        self.targPriority: dict[int, int] = {}
        self.targetIDs: list[int] = []
        self.measurementHist: dict[int, defaultdict] = {}

        # Create the orbit
        # Check if already in units, if not convert
        if type(a) == int:
            a = a * u.km
        self.a = a
        if type(ecc) == int:
            ecc = ecc * u.dimensionless_unscaled
        self.ecc = ecc
        if type(inc) == int:
            inc = inc * u.deg
        self.inc = inc
        if type(raan) == int:
            raan = raan * u.deg
        self.raan = raan
        if type(argp) == int:
            argp = argp * u.deg
        self.argp = argp
        if type(nu) == int:
            nu = nu * u.deg
        self.nu = nu

        # Create the poliastro orbit
        self.orbit = twobody.Orbit.from_classical(
            bodies.Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu
        )
        self.orbitHist = defaultdict(dict)  # contains time and xyz of orbit history
        self.velHist = defaultdict(dict)  # contains time and xyz of velocity history
        self.time = 0

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
        targetID = target.targetID
        if len(self.indeptEstimator.estHist[targetID]) < 1:
            self.indeptEstimator.local_EKF_initialize(target, time)
            return

        # Update the independent estimator using the measurement
        self.indeptEstimator.local_EKF_pred(targetID, time)
        self.indeptEstimator.local_EKF_update([self], [measurement], targetID, time)

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
