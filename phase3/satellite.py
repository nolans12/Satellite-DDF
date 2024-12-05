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

    def _get_neighbors(self, sat_type: str | None = None) -> list[str]:

        neighbors = self._network.get_neighbors(self.name)
        if sat_type is None:
            return neighbors

        # Filter neighbors based on type
        filtered = []
        for neighbor in neighbors:
            if sat_type == "fusion" and neighbor.startswith("FusionSat"):
                filtered.append(neighbor)
            elif sat_type == "sensing" and neighbor.startswith("SensingSat"):
                filtered.append(neighbor)
        return filtered

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
        self, target_id: str, target_ground_truth_pos: npt.NDArray, time: float
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
                    sat_name=self.name,
                    sat_state=sat_state,
                    R_mat=self._sensor.R,
                )
            )

    def update_bounties(self, time: float) -> None:
        """
        Update the bounty for each target.
        """
        # Can use the network to get the .bounties
        bounties = self.get_bounties(time)

        # Update the bounty for each target
        for target_id, source in bounties:
            self.bounty[target_id] = source

        # Get just the target IDs from the bounties tuples
        bounty_target_ids = [target_id for target_id, _ in bounties]

        # Remove bounties that are no longer active
        expired_targets = [tid for tid in self.bounty if tid not in bounty_target_ids]
        for target_id in expired_targets:
            del self.bounty[target_id]

    def get_bounties(self, time: float) -> list[tuple[str, str]]:
        """
        Get the bounties for the satellite.

        Args:
            time: Time to get bounties for.

        Returns:
            List of tuples containing (target_id, source_satellite) for each bounty.
        """
        bounties = self._network.bounties.loc[
            (self._network.bounties['time'] == time)
            & (self._network.bounties['destination'] == self.name)
        ]

        return list(zip(bounties['target_id'].tolist(), bounties['source'].tolist()))

    def send_meas_to_fusion(self, target_id: str, time: float) -> None:
        """
        Send measurements from the sensing satellites to the fusion satellites.
        """

        # Get all measurements for this target_id at this time
        measurements = self.get_measurements(target_id=target_id, time=time)

        if measurements:
            # Check, does this sat have a bounty on this target?
            if target_id in self.bounty:
                # If so, send the measurements to the fusion satellite
                sat_id = self.bounty[target_id]
                self._network.send_measurements_path(
                    measurements,
                    self.name,  # source
                    sat_id,  # destination
                    time=time,
                    size=50 * len(measurements),
                )
            else:
                # Just send to nearest fusion satellite

                neighbors = self._get_neighbors(sat_type="fusion")
                nearest_fusion_sat = min(
                    neighbors, key=lambda x: self._network.get_distance(self.name, x)
                )
                self._network.send_measurements_path(
                    measurements,
                    self.name,  # source
                    nearest_fusion_sat,  # destination
                    time=time,
                    size=50 * len(measurements),
                )

                print(
                    f"Sat {self.name} does not have a bounty for target {target_id}, sending measurements to {nearest_fusion_sat}"
                )

    def get_measurements(
        self, target_id: str, time: float | None = None
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
    def __init__(self, *args, local_estimator: estimator.Estimator | None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custody = {}  # targetID: Boolean, who this sat has custody of
        self.computation_capacity = (
            2  # how many EKFs (target custodys) can hold at a time!
        )
        self._estimator = local_estimator

    def process_measurements(self, time: float) -> None:
        """
        Process the measurements from the fusion satellite.
        """

        # Always check, am i over my computation capacity?
        if sum(self.custody.values()) > self.computation_capacity:
            print(f'Sat {self.name} has too many custody assignments!')
            print(f'{sum(self.custody.values())} > {self.computation_capacity}')
            quit()

        # Find the set of measurements that were sent to this fusion satellite
        # ONLY IF DESTINATION == self.name
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

            # Update estimators based on measurements
            if self._estimator is not None:
                self._estimator.EKF_predict(meas_for_target)
                self._estimator.EKF_update(meas_for_target)

    def send_bounties(
        self, target_id: str, targ_pos: npt.NDArray, time: float, nearest_sens: int
    ) -> None:
        """
        Send a bounty on the target_id from source to all avaliable sensing satellites.

        Inputs:
            target_id: The target ID to send a bounty on.
            time: The time at which to send the bounty.
            nearest_sens: The number of nearest sensing satellites to send the bounty to.
        """

        neighbors = self._network.get_nearest(
            position=targ_pos, sat_type="sensing", number=nearest_sens
        )
        size = 1  # bytes of a bounty send

        # Send a bounty update to all neighbors
        for neighbor in neighbors:
            self._network.send_bounty_path(
                self.name, neighbor, self.name, neighbor, target_id, size, time
            )
