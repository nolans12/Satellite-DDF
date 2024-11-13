# Import classes
import os
import pathlib
import random

import numpy as np
import pandas as pd
from astropy import units as u

from phase3 import collection
from phase3 import comms
from phase3 import estimator
from phase3 import ground_station
from phase3 import orbit
from phase3 import satellite
from phase3 import sensor
from phase3 import sim_config
from phase3 import target
from phase3.plotting import plotter


class Environment:
    def __init__(
        self,
        config: sim_config.SimConfig,
        sensing_sats: list[satellite.SensingSatellite],
        fusion_sats: list[satellite.FusionSatellite],
        ground_stations: list[ground_station.GroundStation],
        targs: list[target.Target],
        comms: comms.Comms,
    ):
        self._config = config
        self._sensing_sats = sensing_sats
        self._fusion_sats = fusion_sats
        self._ground_stations = ground_stations
        self._targs = targs
        self._comms = comms

        # Initialize time parameter to 0
        self.time = u.Quantity(0, u.minute)

        self._plotter = plotter.Plotter(config.plot)

        self._central_estimator = None
        if self._config.estimator is sim_config.Estimators.CENTRAL:
            self._central_estimator = estimator.Estimator()

    @classmethod
    def from_config(cls, cfg: sim_config.SimConfig) -> 'Environment':
        targs = [
            target.Target(
                name=name,
                target_id=t.target_id,
                coords=np.array(t.coords),
                heading=t.heading,
                speed=t.speed,
                uncertainty=np.array(t.uncertainty),
                color=t.color,
            )
            for name, t in cfg.targets.items()
        ]

        sensing_sats = [
            satellite.SensingSatellite(
                name=name,
                sensor=sensor.Sensor(
                    name=s.sensor or '',
                    fov=cfg.sensors[s.sensor or ''].fov,
                    bearingsError=np.array(cfg.sensors[s.sensor or ''].bearings_error),
                ),
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
            )
            for name, s in cfg.sensing_satellites.items()
        ]

        # All estimators are the same now
        estimator_clz = estimator.Estimator

        fusion_sats = [
            satellite.FusionSatellite(
                name=name,
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
                local_estimator=estimator_clz() if estimator_clz else None,
            )
            for name, s in cfg.fusion_satellites.items()
        ]

        ground_stations = [
            ground_station.GroundStation(
                name=name,
                estimator=estimator_clz(),
                config=gs,
            )
            for name, gs in cfg.ground_stations.items()
        ]

        # Define the communication network:
        comms_network = comms.Comms(
            sensing_sats,
            fusion_sats,
            ground_stations,
            config=cfg.comms,
        )

        for sat in sensing_sats + fusion_sats:
            sat.initialize(comms_network)

        # Create and return an environment instance:
        return cls(
            cfg, sensing_sats, fusion_sats, ground_stations, targs, comms_network
        )

    def simulate(self) -> None:
        """
        Simulate the environment over a time range.

        Returns:
        - Data collected during simulation.
        """

        print("Simulation Started")
        print(f"Running in {self._config.estimator.name} mode")

        time_steps = self._config.sim_duration_m / self._config.sim_time_step_m
        # round to the nearest int
        time_steps = round(time_steps)
        time_vec = u.Quantity(
            np.linspace(0, self._config.sim_duration_m, time_steps + 1), u.minute
        )
        # Initialize based on the current time
        time_vec = time_vec + self.time

        for t_net in time_vec:
            print(f'Time: {t_net:.2f}')
            time_step = self.time - t_net
            self.time = t_net

            # Propagate the environments positions
            self.propagate(time_step)

            # Get the measurements from the satellites
            self.collect_all_measurements()

            if self._config.estimator is sim_config.Estimators.FEDERATED:

                # Have the sensing satellites send their measurements to the fusion satellites
                self.transport_measurements_to_fusion()

                # Have the fusion satellites process their measurements
                self.process_fusion_measurements()

                # Only update custodies and bounties once plan_horizon
                # if self.time.value % self._config.plan_horizon_m == 0:

                # TODO: for now just always update custodies and bounties
                # The fusion layer figures out who should have custody on which targets/regions
                # End goal is to send out to sensing layers who they should report too
                self.update_custodies_and_fuse()

                # Update the bounties for the sensing satellites
                # Based on receiving targets from fusion layer
                self.update_bounties()

            if self._config.estimator is sim_config.Estimators.CENTRAL:
                self.central_fusion()

            # Update the plot environment
            self._plotter.plot(
                self.time.value,
                self._sensing_sats + self._fusion_sats,
                self._targs,
                self._ground_stations,
                self._comms,
            )

        print('Simulation Complete')

    def post_process(self):
        self.save_data_frames(save_name=self._config.plot.output_prefix)
        # Save gifs
        self._plotter.render_gifs()

    def propagate(self, time_step: u.Quantity[u.minute]) -> None:
        """
        Propagate the satellites and targets over the given time step.
        """
        # Get the float value
        time_val = self.time.value

        # Propagate the targets' positions
        for targ in self._targs:
            targ.propagate(
                time_step, time_val
            )  # Propagate and store the history of target time and xyz position and velocity

        # Propagate the satellites
        for sat in self._sensing_sats + self._fusion_sats:
            sat.propagate(
                time_step, time_val
            )  # Propagate and store the history of satellite time and xyz position and velocity

        # Update the communication network for the new sat positions
        self._comms.update_edges()

    def update_custodies_and_fuse(self) -> None:
        """
        The fusion layer figures out who should have custody on which targets/regions
        End goal is to send out to sensing layers who they should report too
        """

        # CONSISTENCY:
        # - Returns a dictionary mapping from targetID to a fused/consistent estimate
        target_estimates = self.get_consistent()

        # CUSTODY:
        # Once have consistent knowledge of the target states, figure out custody assignments
        # To start, just use satellite that is closest to the target
        custody_assignments = {}

        # Loop through all target estimates
        for target_id, estimate in target_estimates.items():
            # Get the closest satellite to the target
            targ_pos = estimate.estimate[np.array([0, 2, 4])]

            closest_sat = None
            closest_sat_dist = float('inf')
            for sat in self._fusion_sats:
                sat_pos = sat.pos
                dist = np.linalg.norm(targ_pos - sat_pos)
                if dist < closest_sat_dist:
                    closest_sat_dist = dist
                    closest_sat = sat

            custody_assignments[target_id] = closest_sat

        # TODO: TRACK TO TRACK HANDOFF

        # Use sat.send_bounties to send out the custody assignments
        for target_id, sat in custody_assignments.items():
            sat.custody[target_id] = True
            print(f'Sat {sat.name} has custody of target {target_id}')
            sat.send_bounties(target_id, self.time.value, nearest_sens=5)

        # Turn all other custody to false
        for sat in self._fusion_sats:
            for target_id in sat.custody:
                if (
                    target_id not in custody_assignments
                    or sat.name != custody_assignments[target_id].name
                ):
                    sat.custody[target_id] = False

    def get_consistent(self):
        """
        Ensures the fusion layer is consistent that uses CI to fuse common estimates.
        Assumes:
        - All estimates are known throughout the network.
        - Satellites can instantly do CI with each other.

        Returns:
        - Dictionary mapping from targetID to a fused/consistent estimate
        """

        # ASSUMPTION: All estimates are known throughout the network, perfect information

        # Collect all estimates in the network, at current time step
        target_estimates = {}
        for sat in self._fusion_sats:
            if sat._estimator is None or sat._estimator.estimation_data.empty:
                continue

            # Get latest estimates for each target this satellite tracks
            current_estimates = sat._estimator.estimation_data[
                sat._estimator.estimation_data.time == self.time.value
            ]

            for target_id in set(current_estimates.target_id):
                if target_id not in target_estimates:
                    target_estimates[target_id] = []

                # Get estimate for this target from this satellite at current time
                target_estimate = current_estimates[
                    current_estimates.target_id == target_id
                ].iloc[-1]

                # Store tuple of (estimate, satellite) for each target (need sat to use the CI call)
                target_estimates[target_id].append((target_estimate, sat))

        # CONSISTENCY:
        # - If there are multiple estimates for the same target, perform CI with all of them
        # - Assume the fusion layer just magically can do this at the moment, perfect knowledge/comms
        for target_id, estimate_sat_pairs in target_estimates.items():
            # If two satellites have estimates for the same target, perform CI with all of them
            if len(estimate_sat_pairs) > 1:
                for nan, curr_sat in estimate_sat_pairs:  # For each sat
                    other_estimates = [
                        est
                        for est, sat in estimate_sat_pairs
                        if sat.name != curr_sat.name
                    ]
                    print(
                        f"Satellite {curr_sat.name} fusing with {[sat.name for est, sat in estimate_sat_pairs if sat.name != curr_sat.name]}"
                    )
                    curr_sat._estimator.CI(
                        other_estimates
                    )  # Perform CI with list of all other estimates

                # Update the target estimates dictionary with the fused estimate (most recent)
                target_estimates[target_id] = estimate_sat_pairs[0][
                    1
                ]._estimator.estimation_data.iloc[
                    -1
                ]  # Take just the first satellites fused estimate
            else:
                # This removes the sat object from the dictionary, is just the estimate stored in target_estimates
                target_estimates[target_id] = estimate_sat_pairs[0][0]

        return target_estimates

    def update_bounties(self) -> None:
        """
        Update the bounties for the sensing satellites.
        """
        for sat in self._sensing_sats:
            sat.update_bounties(self.time.value)

    def collect_all_measurements(self) -> None:
        """
        Collect measurements from the satellites.
        """
        for sat in self._sensing_sats:
            for targ in self._targs:
                sat.collect_measurements(targ.target_id, targ.pos, self.time.value)

    def transport_measurements_to_fusion(self) -> None:
        """
        Transport measurements from the sensing satellites to the fusion satellites.
        """
        for sat in self._sensing_sats:
            for targ in self._targs:
                sat.send_meas_to_fusion(targ.target_id, self.time.value)

    def process_fusion_measurements(self) -> None:
        """
        Process the measurements from the fusion satellites.
        """
        for sat in self._fusion_sats:
            sat.process_measurements(self.time.value)

    def central_fusion(self):
        """
        Perform central fusion using collected measurements.
        """
        assert self._central_estimator is not None

        # Get all measurements taken by the satellites on each target
        for targ in self._targs:

            targetID = targ.target_id
            measurements: list[collection.Measurement] = []

            for sat in self._sensing_sats:
                measures = sat.get_measurements(targetID, self.time.value)
                measurements.extend(measures)

            if not measurements:
                continue

            self._central_estimator.EKF_predict(measurements)
            self._central_estimator.EKF_update(measurements)

    def send_to_ground_best_sat(self):
        """
        Planner for sending data from the satellite network to the ground station.

        DDF, BEST SAT:
            Choose the satellite from the network with the lowest track uncertainty, that can communicate with ground station.
            For that sat, send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self._ground_stations:

            # Loop through all targets (assume the GS cares about all for now))
            for targ in self._targs:

                # Figure out which satellite has the lowest track uncertainty matrix for this targetID
                bestSat = None
                bestTrackUncertainty = float('inf')
                bestData = None
                for sat in self._fusion_sats:

                    # Get the estimator data for this satellite
                    estimator_data = sat.estimator.estimation_data

                    # Check, does this satellite have data for this targetID?
                    if estimator_data.empty:
                        continue
                    if targ.target_id in estimator_data['targetID'].values:

                        # Get the data and track uncertainty
                        data = estimator_data[
                            estimator_data['targetID'] == targ.target_id
                        ].iloc[-1]
                        trackUncertainty = data['trackError']

                        # Update the best satellite if this one has lower track uncertainty
                        if trackUncertainty < bestTrackUncertainty:
                            bestTrackUncertainty = trackUncertainty
                            bestSat = sat
                            bestData = data

                # If a satellite was found
                if bestSat is not None:

                    # Get the data
                    est = bestData['est']
                    cov = bestData['cov']

                    data = collection.GsEstimateTransmission(
                        target_id=targ.target_id,
                        sender=bestSat.name,
                        receiver=gs.name,
                        time=self.time.value,
                        estimate=est,
                        covariance=cov,
                    )

                    # Add the data to the queued data onboard the ground station
                    gs.queue_data(
                        data, dtype=collection.GsDataType.COVARIANCE_INTERSECTION
                    )

        # Now that the data is queued, process the data in the filter
        for gs in self._ground_stations:
            gs.process_queued_data(self._fusion_sats, self._targs)

    def save_data_frames(
        self,
        save_name: str,
        save_path: pathlib.Path | None = None,
    ) -> None:
        """
        Save all data frames to csv files.

        Input is a folder name, "save_name"
        A folder will be made in the data folder with this name, and all data frames will be saved in there.
        """
        if save_path is None:
            save_path = pathlib.Path(__file__).parent / 'data'

        # Make sure the save path exists
        save_path.mkdir(exist_ok=True)

        # Then make a folder for the save_name in data
        save_path = save_path / save_name
        save_path.mkdir(exist_ok=True)

        # Now, save all data frames

        ## The data we have is:
        # - Target state data (done)

        # - Satellite state data (done)
        # - Satellite estimator data (est, cov, innovaiton, etc) (NOT DONE)

        # - Ground station estimate and covariance data (NOT DONE)

        # - Communications to and from satellites and ground stations (kinda done)

        ## Make a target state folder
        target_path = save_path / 'targets'
        target_path.mkdir(exist_ok=True)
        for targ in self._targs:
            if not targ._state_hist.empty:
                targ._state_hist.to_csv(target_path / f'{targ.name}.csv')

        ## Make a satellite state folder
        satellite_path = save_path / 'satellites'
        satellite_path.mkdir(exist_ok=True)
        for sat in self._fusion_sats + self._sensing_sats:
            if not sat._state_hist.empty:
                sat._state_hist.to_csv(satellite_path / f'{sat.name}_state.csv')

        for sat in self._sensing_sats:
            if not sat._measurement_hist.empty:
                sat._measurement_hist.to_csv(
                    satellite_path / f'{sat.name}_measurements.csv'
                )

        ## Make an estimator folder
        estimator_path = save_path / 'estimators'
        estimator_path.mkdir(exist_ok=True)
        # Now, save all estimator data
        for sat in self._fusion_sats:
            if sat._estimator is not None:
                if not sat._estimator.estimation_data.empty:
                    sat._estimator.estimation_data.to_csv(
                        estimator_path / f"{sat.name}_estimator.csv"
                    )
        if (
            self._central_estimator is not None
            and not self._central_estimator.estimation_data.empty
        ):
            self._central_estimator.estimation_data.to_csv(
                estimator_path / 'central_estimator.csv'
            )
