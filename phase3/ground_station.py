from collections import defaultdict
from typing import Literal, Sequence, overload

import numpy as np
from astropy import units as u
from poliastro import bodies

from common import dataclassframe
from phase3 import collection
from phase3 import estimator
from phase3 import satellite
from phase3 import sim_config
from phase3 import target


class GroundStation:
    """
    Ground station class
    """

    def __init__(
        self,
        name: str,
        estimator: estimator.GsEstimator,
        config: sim_config.GroundStation,
    ):
        """Initialize a GroundStation object.

        Args:
            estimator (object): Estimator to use for the ground station.
            lat (float): Latitude of the ground station.
            long (float): Longitude of the ground station.
            fov (float): Spherical field of view of the groundStation
            commRange (float): Communication range of the ground station (km).
            name (str): Name of the ground station.
            color (str): Color of the ground station for visualization.
        """

        # Estimator object to use
        self.estimator = estimator

        # Location of the ground station in ECEF
        lat = np.deg2rad(config.lat)
        lon = np.deg2rad(config.lon)
        self.pos = np.array(
            [
                bodies.Earth.R.to(u.km).value * np.cos(lat) * np.cos(lon),
                bodies.Earth.R.to(u.km).value * np.cos(lat) * np.sin(lon),
                bodies.Earth.R.to(u.km).value * np.sin(lat),
            ]
        )

        # Track communication sent/recieved
        self.queued_ci_data = dataclassframe.DataClassFrame(
            clz=collection.GsEstimateTransmission
        )
        self.queued_meas_data = dataclassframe.DataClassFrame(
            clz=collection.GsMeasurementTransmission
        )
        self.comm_ci_data = dataclassframe.DataClassFrame(
            clz=collection.GsEstimateTransmission
        )
        self.comm_meas_data = dataclassframe.DataClassFrame(
            clz=collection.GsMeasurementTransmission
        )

        # Communication range of a satellite to the ground station
        self.fov = config.fov
        self.comms_range = config.comms_range

        # Other parameters
        self.name = name
        self.color = config.color
        self.time = 0

    @overload
    def queue_data(
        self,
        data: collection.GsMeasurementTransmission,
        dtype: Literal[collection.GsDataType.MEASUREMENT],
    ) -> None: ...

    @overload
    def queue_data(
        self,
        data: collection.GsEstimateTransmission,
        dtype: Literal[collection.GsDataType.COVARIANCE_INTERSECTION],
    ) -> None: ...

    def queue_data(
        self,
        data: collection.GsMeasurementTransmission | collection.GsEstimateTransmission,
        dtype: (
            Literal[collection.GsDataType.MEASUREMENT]
            | Literal[collection.GsDataType.COVARIANCE_INTERSECTION]
        ),
    ) -> None:
        """
        Adds the data to the queued data struct to be used later in processing, the mailbox system.

        Args:
            data, in order of [type][time][targetID][sat] = measurement
        """
        if dtype is collection.GsDataType.COVARIANCE_INTERSECTION:
            self.queued_ci_data.append(data)  # type: ignore
        elif dtype is collection.GsDataType.MEASUREMENT:
            self.queued_meas_data.append(data)  # type: ignore
        else:
            raise ValueError(f'Unexpected data type: {dtype}')

    def process_queued_data(
        self, sats: Sequence[satellite.Satellite], targs: list[target.Target]
    ) -> None:
        """
        Processes the data queued to be sent to the ground station.

        This function uses the estimator object to process the data.
        """
        if len(self.queued_meas_data):
            # Perform standard EKF with queued measurements

            # Map (target_id, time) -> transmissions
            measurements: dict[tuple[int, float], list[collection.GsTransmission]] = (
                defaultdict(list)
            )
            for transmission in self.queued_meas_data.to_dataclasses(
                self.queued_meas_data
            ):
                # Store the queued data into the commData, for post processing
                self.comm_meas_data.append(transmission)

                measurements[transmission.target_id, transmission.time].append(
                    transmission
                )

                # Now, with the lists of measurements and sats, send to the estimator
                targ = next(
                    filter(lambda t: t.target_id == transmission.target_id, targs)
                )
                if len(self.estimator.estHist[transmission.target_id]) < 1:
                    self.estimator.EKF_initialize(targ, transmission.time)
                    return

            for (target_id, meas_time), transmission in measurements.items():
                # Else, update the estimator
                self.estimator.EKF_pred(target_id, meas_time)
                self.estimator.EKF_update(sats, measurements, target_id, meas_time)

        if len(self.queued_ci_data):
            # Perform covariance intersection here
            for transmission in self.queued_ci_data.to_dataclasses(self.queued_ci_data):

                # Now do CI with the data
                self.estimator.CI(
                    transmission.target_id,
                    transmission.estimate,
                    transmission.covariance,
                    transmission.time,
                )

                # Store the queued data into the commData, for post processing
                self.comm_ci_data.append(transmission)

        # Clear the queued data
        self.queued_ci_data.drop(self.queued_ci_data.index, inplace=True)

    def can_communicate(self, x_sat: float, y_sat: float, z_sat: float) -> bool:
        """
        Returns True if the satellite can communicate with the ground station at the given time.

        Args:
            x_sat, y_sat, z_sat (float): ECEF position of the satellite

        Returns:
            bool: True if the satellite can communicate with the ground station at the given time.
        """

        # Create two lines, one from the center of earth to GS and one from GS to satellite

        # Get the ground station position
        x_gs, y_gs, z_gs = self.pos

        # Earth to GS vec
        e_to_gs_vec = [x_gs - 0, y_gs - 0, z_gs - 0]

        # GS to satellite vec
        gs_to_sat_vec = [x_sat - x_gs, y_sat - y_gs, z_sat - z_gs]

        # Get the angle between the two vectors
        angle = np.arccos(
            np.dot(e_to_gs_vec, gs_to_sat_vec)
            / (np.linalg.norm(e_to_gs_vec) * np.linalg.norm(gs_to_sat_vec))
        )

        # Now check, can the satellite talk with the ground station
        if angle < np.deg2rad(self.fov):
            if np.linalg.norm(gs_to_sat_vec) < self.comms_range:
                return True

        return False
