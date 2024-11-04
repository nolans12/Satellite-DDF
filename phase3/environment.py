# Import classes
import os
import pathlib
import random

import numpy as np
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
            self._central_estimator = estimator.CentralEstimator()

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

        if cfg.estimator is sim_config.Estimators.EVENT_TRIGGERED:
            # The central estimator is an independent estimator
            estimator_clz = estimator.IndependentEstimator
        elif cfg.estimator is sim_config.Estimators.COVARIANCE_INTERSECTION:
            estimator_clz = estimator.CiEstimator
        else:
            estimator_clz = None

        fusion_sats = [
            satellite.FusionSatellite(
                name=name,
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
                local_estimator=estimator_clz() if estimator_clz else None,
            )
            for name, s in cfg.sensing_satellites.items()
        ]

        ground_stations = [
            ground_station.GroundStation(
                name=name,
                estimator=estimator.GsEstimator(),
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
        # self._plotter.render_gifs()

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

    def collect_all_measurements(self) -> None:
        """
        Collect measurements from the satellites.
        """
        for sat in self._sensing_sats:
            for targ in self._targs:
                sat.collect_measurements(targ.target_id, targ.pos, self.time.value)

    def data_fusion(self) -> None:
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """
        if self._config.estimator is sim_config.Estimators.CENTRAL:
            self.central_fusion()
        elif self._config.estimator is sim_config.Estimators.COVARIANCE_INTERSECTION:
            self.send_estimates()
        elif self._config.estimator is sim_config.Estimators.EVENT_TRIGGERED:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        if self._config.estimator is sim_config.Estimators.COVARIANCE_INTERSECTION:
            for sat in self._fusion_sats:
                # Get just the data sent using CI, (match receiver to sat name and type to 'estimate')
                data_recieved = self._comms.comm_data[
                    (self._comms.comm_data['receiver'] == sat.name)
                    & (self._comms.comm_data['type'] == 'estimate')
                ]

                sat.filter_CI(data_recieved)

            if self._config.estimator is sim_config.Estimators.EVENT_TRIGGERED:
                etEKF = sat.etEstimators[0]
                etEKF.event_trigger_processing(sat, self.time.to_value(), self._comms)

        # ET estimator needs prediction to happen at everytime step, thus, even if measurement is none we need to predict
        elif self._config.estimator is sim_config.Estimators.EVENT_TRIGGERED:
            for sat in self._fusion_sats:
                etEKF = sat.etEstimators[0]
                etEKF.event_trigger_updating(sat, self.time.to_value(), self._comms)

    def send_estimates(self):
        """
        Send the most recent estimates from each satellite to its neighbors.

        Worst case CI, everybody sents to everybody
        """
        # Loop through all satellites
        random_sats = self._sensing_sats[:] + self._fusion_sats[:]
        random.shuffle(random_sats)
        for sat in random_sats:
            # For each targetID in the satellite estimate history

            # also shuffle the targetIDs (using new data frames)
            data_sat = sat.estimator.estimation_data

            # If no data, skip
            if data_sat.empty:
                continue

            # Else, get the targetIDs
            targetIDs = data_sat['targetID'].unique()

            # Shuffle the targetIDs
            random.shuffle(targetIDs)
            for targetID in targetIDs:

                # Now, get the most recent estimate for this targetID
                data_curr = data_sat[data_sat['targetID'] == targetID].iloc[-1]
                est = data_curr['est']
                cov = data_curr['cov']

                # Send the estimate to all neighbors
                for neighbor in self._comms.G.neighbors(sat):
                    self._comms.send_estimate(
                        sat, neighbor, est, cov, targetID, self.time.value
                    )

    def send_measurements(self):
        """
        Send the most recent measurements from each satellite to its neighbors.
        """
        # Loop through all satellites
        for sat in self._sensing_sats:
            # For each targetID in satellites measurement history
            for (
                target
            ) in self._targs:  # TODO: iniitalize with senders est and cov + noise?
                if target.targetID in sat.targetIDs:
                    targetID = target.targetID
                    envTime = self.time.value
                    # Skip if there are no measurements for this targetID
                    if isinstance(
                        sat.measurementHist[target.targetID][envTime], np.ndarray
                    ):
                        # This means satellite has a measurement for this target, now send it to neighbors
                        for neighbor in self._comms.G.neighbors(sat):
                            neighbor: satellite.Satellite
                            # If target is not in neighbors priority list, skip
                            if targetID not in neighbor.targPriority.keys():
                                continue

                            # Get the most recent measurement time
                            satTime = max(
                                sat.measurementHist[targetID].keys()
                            )  #  this should be irrelevant and equal to  self.time since a measurement is sent on same timestep

                            # Get the local EKF for this satellite
                            local_EKF = sat.etEstimators[0]

                            # Check for a new commonEKF between two satellites
                            commonEKF = None
                            for each_etEstimator in sat.etEstimators:
                                if each_etEstimator.shareWith == neighbor.name:
                                    commonEKF = each_etEstimator
                                    break

                            if (
                                commonEKF is None
                            ):  # or make a common filter if one doesn't exist
                                commonEKF = estimator.EtEstimator(
                                    local_EKF.targetPriorities, shareWith=neighbor.name
                                )
                                commonEKF.EFK_initialize(target, envTime)
                                sat.etEstimators.append(commonEKF)
                                # commonEKF.synchronizeFlag[targetID][envTime] = True

                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.EFK_initialize(target, envTime)

                            # Get the neighbors localEKF
                            neighbor_localEKF = neighbor.etEstimators[0]

                            # If the neighbor doesn't have a local EKF on this target, create one
                            if len(neighbor_localEKF.estHist[targetID]) == 0:
                                neighbor_localEKF.EKF_initialize(target, envTime)

                            # Check for a common EKF between the two satellites
                            commonEKF = None
                            for each_etEstimator in neighbor.etEstimators:
                                if each_etEstimator.shareWith == sat.name:
                                    commonEKF = each_etEstimator
                                    break

                            if (
                                commonEKF is None
                            ):  # if I don't, create one and add it to etEstimators list
                                commonEKF = estimator.EtEstimator(
                                    neighbor.targPriority, shareWith=sat.name
                                )
                                commonEKF.EFK_initialize(target, envTime)
                                neighbor.etEstimators.append(commonEKF)
                                # commonEKF.synchronizeFlag[targetID][envTime] = True

                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.EFK_initialize(target, envTime)

                            # Create implicit and explicit measurements vector for this neighbor
                            alpha, beta = local_EKF.event_trigger(
                                sat, neighbor, targetID, satTime
                            )

                            # Send that to neightbor
                            self._comms.send_measurements(
                                sat, neighbor, alpha, beta, targetID, satTime
                            )

                            if commonEKF.synchronize_flag[targetID][envTime]:
                                # Since this runs twice, we need to make sure we don't double count the data
                                self._comms.total_comm_et_data.append(
                                    collection.MeasurementTransmission(
                                        target_id=targetID,
                                        sender=sat.name,
                                        receiver=neighbor.name,
                                        time=envTime,
                                        size=50,
                                        alpha=alpha,
                                        beta=beta,
                                    )
                                )

    def central_fusion(self):
        """
        Perform central fusion using collected measurements.
        """
        assert self._central_estimator is not None

        # Get all measurements taken by the satellites on each target
        for targ in self._targs:

            targetID = targ.target_id
            measurements: list[collection.Measurement] = []

            sats_w_measurements = []

            for sat in self._sensing_sats:
                measures = sat.get_measurements(targetID, self.time.value)
                measurements.extend(measures)
                if measures:
                    sats_w_measurements.append(sat)

            if not measurements:
                continue

            self._central_estimator.EKF_pred(targ, self.time.value)
            self._central_estimator.EKF_update(
                sats_w_measurements, measurements, targetID, self.time.value
            )

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
            targ._state_hist.to_csv(target_path / f'{targ.name}.csv')

        ## Make a satellite state folder
        satellite_path = save_path / 'satellites'
        satellite_path.mkdir(exist_ok=True)
        for sat in self._fusion_sats:
            sat._state_hist.to_csv(satellite_path / f'{sat.name}_state.csv')

        for sat in self._sensing_sats:
            sat._measurement_hist.to_csv(
                satellite_path / f'{sat.name}_measurements.csv'
            )

        # ## Make a comms folder # TODO: fix with ryans new data class
        # comms_path = os.path.join(savePath, f"comms")
        # os.makedirs(comms_path, exist_ok=True)
        # if not self._comms.comm_data.empty:
        #     self._comms.comm_data.to_csv(os.path.join(comms_path, f"comm_data.csv"))

        # ## Make an estimator folder # TODO fix with estimator stuff
        # estimator_path = os.path.join(savePath, f"estimators")
        # os.makedirs(estimator_path, exist_ok=True)
        # # Now, save all estimator data
        # for sat in self.sats:
        #     if sat.estimator is not None:
        #         sat.estimator.estimation_data.to_csv(  # TODO: this should just be sat.estimator.estimation_data, need to change naming syntax
        #             os.path.join(estimator_path, f"{sat.name}_estimator.csv")
        #         )
        # for gs in self.groundStations:
        #     if gs.estimator is not None:
        #         gs.estimator.estimation_data.to_csv(
        #             os.path.join(estimator_path, f"{gs.name}_estimator.csv")
        #         )

        # if self._config.estimator is sim_config.Estimators.CENTRAL:
        #     if self.estimator is not None:
        #         self.estimator.estimation_data.to_csv(
        #             os.path.join(estimator_path, f"central_estimator.csv")
        #         )
