# Import classes
import copy
import pathlib
import time

import numpy as np
import pandas as pd
import pulp
from astropy import units as u

from common.predictors import state_transition
from phase3 import collection
from phase3 import comms
from phase3 import estimator
from phase3 import ground_station
from phase3 import orbit
from phase3 import raidRegion
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
        raid_regions: list[raidRegion.RaidRegion],
        targs: list[target.Target],
        comms: comms.Comms,
    ):
        self._config = config
        self._sensing_sats = sensing_sats
        self._fusion_sats = fusion_sats
        self._ground_stations = ground_stations
        self._raid_regions = raid_regions
        self._targs = targs
        self._comms = comms
        self._custody = {}

        # Initialize time parameter to 0
        self.time = u.Quantity(0, u.minute)

        self._plotter = plotter.Plotter(config.plot, config, raid_regions)

        self._central_estimator = None
        if self._config.estimator is sim_config.Estimators.CENTRAL:
            self._central_estimator = estimator.Estimator()

        self.timer = pd.DataFrame(columns=['time', 'federated_processing_time'])

    @classmethod
    def from_config(cls, cfg: sim_config.SimConfig) -> 'Environment':

        raid_regions = [
            raidRegion.RaidRegion(
                name=name,
                center=np.array(r.center),
                extent=np.array(r.extent),
                initial_targs=r.initial_targs,
                spawn_rate=r.spawn_rate,
                color=r.color,
                priority=r.priority,
            )
            for name, r in cfg.raids.items()
        ]

        targs = [t for raid in raid_regions for t in raid.targets]

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
            cfg,
            sensing_sats,
            fusion_sats,
            ground_stations,
            raid_regions,
            targs,
            comms_network,
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

                if self._config.do_ekfs:
                    # Have the fusion satellites process their measurements
                    self.process_fusion_measurements()
                else:
                    # Give the fusion satellites perfect data
                    self.perfect_target_data()

                # Only update custodies and bounties every plan time
                if self.time.value % self._config.plan_horizon_m == 0:
                    print("Planning...")

                    # Update custodies and fuse throughout fusion layer
                    self.update_custodies_and_fuse()

                    # Process the bounties queued on the network
                    self.update_bounties()

            if self._config.estimator is sim_config.Estimators.CENTRAL:
                self.central_fusion()

            # Update the plot environment
            self._plotter.plot(
                self.time.value,
                self._sensing_sats + self._fusion_sats,
                self._targs,
                self._raid_regions,
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

        # CUSTODY PLAN:
        # - Returns a dictionary mapping from targetID to the closest fusion satellite
        # custody_assignments = self.closest_fusion_custody(target_estimates)
        custody_assignments = self.short_horizon_custody(
            target_estimates, self._config.plan_horizon_m * u.minute, 5
        )
        # custody_assignments = self.short_horizon_a_star(
        #     target_estimates, self._config.plan_horizon_m * u.minute, 5
        # )

        # CUSTODY HANDOFF:
        # - Track to track handoff as needed
        # - Also resends out bounties to the sensing satellites
        self.custody_handoff(custody_assignments)

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
                if self._config.do_ekfs:
                    # If have real data, need to do ci
                    for nan, curr_sat in estimate_sat_pairs:  # For each sat
                        other_estimates = [
                            est
                            for est, sat in estimate_sat_pairs
                            if sat.name != curr_sat.name
                        ]
                    for est, sat in estimate_sat_pairs:
                        if sat.name != curr_sat.name:
                            print(f"Fusing with {sat.name} on target {target_id}")
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

    def closest_fusion_custody(self, target_estimates: dict) -> dict:
        """
        Determine custody assignments based on closest fusion satellite to each target.

        Args:
            target_estimates: Dictionary mapping target_id to its current estimate

        Returns:
            Dictionary mapping target_id to the closest fusion satellite
        """
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

        return custody_assignments

    def short_horizon_custody(
        self, target_estimates: dict, time_horizon: u.Quantity[u.minute], num_evals: int
    ) -> dict:
        """
        Determine custody assignments for the next u.minutes.
        Will solve the optimization problem to minimize mean difference between custody sat and target pos.

        Args:
            target_estimates: Dictionary mapping target_id to its current estimate
            time_horizon: Time horizon to plan for

        Returns:
            Dictionary mapping target_id to the closest fusion satellite
        """
        custody_assignments = {}

        ## Need to create an optimization problem, to minimize the cost of assigning targets
        prob = pulp.LpProblem("Custody_Optimization", pulp.LpMinimize)

        ## Create the decision variables, one for each fusion node -> targetID pairing
        x = {}
        for sat in self._fusion_sats:
            for target_id in target_estimates.keys():
                x[(sat, target_id)] = pulp.LpVariable(
                    f'{sat.name} → {target_id}', 0, 1, pulp.LpBinary
                )

        ## Define the cost of assigning a target to a fusion node
        def cost(sat, target_id, time_horizon, num_evals):

            # Make a copy of the satellite objects orbit
            sat_orbit = copy.deepcopy(sat.orbit)

            # Get the mean distance b/w fusion node and estimated targ pos over next time_horizon, evaluated num_evals times
            cost = 0
            targ_pos = target_estimates[target_id].estimate
            times = np.linspace(0, time_horizon.value, num_evals)
            dts = np.diff(times)
            for i, time in enumerate(times):
                pred_targ = state_transition(targ_pos, time)[np.array([0, 2, 4])]
                if i == 0:
                    pred_sat = sat_orbit.r.value
                else:
                    pred_sat = sat_orbit.propagate(dts[i - 1] * u.minute).r.value
                cost += np.linalg.norm(pred_sat - pred_targ)
            return cost / num_evals

        priority_table = {raid._name: raid._priority for raid in self._raid_regions}
        exchange_rate = self._config.exchange_rate

        ## Define the priority scalaing function
        def priority(target_id, exchange_rate, priority_table):

            prior = None
            for region, priority in priority_table.items():
                if region in target_id:
                    prior = priority

            if prior is None:
                print("ERROR: No priority found for target_id: ", target_id)
                exit()

            return exchange_rate ** (prior - 1)

        ## Define the objective function
        prob += pulp.lpSum(
            [
                cost(sat, target_id, time_horizon, num_evals)
                * priority(target_id, exchange_rate, priority_table)
                * x[(sat, target_id)]
                for (sat, target_id) in x.keys()
            ]
        )

        ## Define the constraints

        # Every target must be assigned to exactly one fusion node if capacity allows
        total_capacity = sum(sat.computation_capacity for sat in self._fusion_sats)
        num_targets = len(target_estimates)

        if num_targets <= total_capacity:
            # If we have enough capacity, assign each target to exactly one node
            for target_id in target_estimates.keys():
                prob += (
                    pulp.lpSum([x[(sat, target_id)] for sat in self._fusion_sats]) >= 1
                )
        else:
            # If we don't have enough capacity, maximize assignments up to capacity
            prob += (
                pulp.lpSum([x[(sat, target_id)] for sat, target_id in x.keys()])
                == total_capacity
            )
            # Also add the constraint that every target can only be assigned to one satellite
            for target_id in target_estimates.keys():
                prob += (
                    pulp.lpSum([x[(sat, target_id)] for sat in self._fusion_sats]) <= 1
                )
            print("OVERCAPACITY!")

        # The number of custodies assigned must be less than the capacity of the fusion node
        for sat in self._fusion_sats:
            prob += (
                pulp.lpSum(
                    [x[(sat, target_id)] for target_id in target_estimates.keys()]
                )
                <= sat.computation_capacity
            )

        # Solve the optimization problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        # prob.solve()

        # Get the custody assignments
        for sat, target_id in x.keys():
            if pulp.value(x[(sat, target_id)]) == 1:
                custody_assignments[target_id] = sat

        # If we are over capacity, print a table of how many assignments are for each region, USA, Germany, India, Australia, Hawaii
        if num_targets > total_capacity:
            print(
                "Over capacity, here is the table of assignments, for exchange rate: ",
                exchange_rate,
            )
            for region in priority_table.keys():
                print(
                    f"{region}: {sum(target_id.startswith(region) for target_id in custody_assignments.keys())}"
                )
            # exit()

        return custody_assignments

    def short_horizon_a_star(
        self, target_estimates: dict, time_horizon: u.Quantity[u.minute], num_evals: int
    ) -> dict:
        """
        Determine custody assignments for the next u.minutes.
        Cost function is the mean A* distance from target to closest fusion node to current sat.
        """
        custody_assignments = {}

        ## Need to create an optimization problem, to minimize the cost of assigning targets
        prob = pulp.LpProblem("Custody Optimization", pulp.LpMinimize)

        ## Create the decision variables, one for each fusion node -> targetID pairing
        x = {}
        for sat in self._fusion_sats:
            for target_id in target_estimates.keys():
                x[(sat, target_id)] = pulp.LpVariable(
                    f'{sat.name} → {target_id}', 0, 1, pulp.LpBinary
                )

        ## Define the cost of assigning a target to a fusion node
        # Based on mean projected A* distance in graph
        def cost(sat, target_id, time_horizon, num_evals):

            # Make a deep copy of the network each time (assume each satellite knows the network)
            comms_copy = copy.deepcopy(self._comms)

            cost = 0
            targ_state = target_estimates[target_id].estimate
            times = np.linspace(
                0, time_horizon.value, num_evals
            )  # from 0 to time_horizon, num_evals time steps
            dts = np.diff(times)
            # all relative to current time, so 0 at start
            for i, time in enumerate(times):

                if i != 0:
                    # Propagate all sats orbits and update the network
                    for node in comms_copy._nodes.values():
                        node.orbit = node.orbit.propagate(dts[i - 1] * u.minute)
                    comms_copy.update_edges()

                # Now, get the predicted target position
                pred_targ = state_transition(targ_state, time)[np.array([0, 2, 4])]

                # Now, find hte node in the graph that is closest to the predicted target position
                closest_node = comms_copy.get_nearest(pred_targ, 'fusion', 1)[0]

                # Now, get the path from the closest node to the satellite
                path = comms_copy.get_path(closest_node, sat.name, 0)

                # Now, get the mean distance of the path
                cost += np.sum(
                    [
                        comms_copy.get_distance(path[i], path[i + 1])
                        for i in range(len(path) - 1)
                    ]
                )

            return cost / num_evals

        # ex = list(x.keys())[99]
        # ex_sat, ex_target_id = ex
        # test = cost(ex_sat, ex_target_id, time_horizon, num_evals)

        print("generating costs!")
        ## Define the objective function
        prob += pulp.lpSum(
            [
                cost(sat, target_id, time_horizon, num_evals) * x[(sat, target_id)]
                for (sat, target_id) in x.keys()
            ]
        )

        print("costs generated!")

        ## Define the constraints

        # Every target must be assigned to exactly one fusion node
        for target_id in target_estimates.keys():
            prob += pulp.lpSum([x[(sat, target_id)] for sat in self._fusion_sats]) >= 1

        # The number of custodies assigned must be less than the capacity of the fusion node
        for sat in self._fusion_sats:
            prob += (
                pulp.lpSum(
                    [x[(sat, target_id)] for target_id in target_estimates.keys()]
                )
                <= sat.computation_capacity
            )

        # Solve the optimization problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Get the custody assignments
        for sat, target_id in x.keys():
            if pulp.value(x[(sat, target_id)]) == 1:
                custody_assignments[target_id] = sat

        return custody_assignments

    def custody_handoff(self, custody_assignments: dict) -> None:
        """
        Update the custody assignments for the satellites based on track to track handoff.

        The input is the new custody assignments. dict[target_id] = sat_object
        Need to loop through old custodies and update them to the new ones.
        """

        # Get current custody assignments
        current_custody = {
            target_id: sat for sat in self._fusion_sats for target_id in sat.custody
        }
        # Turn off all current custody assignments
        for sat in self._fusion_sats:
            for target_id in sat.custody:
                sat.custody[target_id] = False

        # Get all target ids in the new custody assignments
        new_target_ids = set(custody_assignments.keys())

        for target_id in new_target_ids:

            new_sat = custody_assignments[target_id]

            # Check for track to track handoff, custody has changed
            prior_est = None
            if (
                target_id in current_custody
                and current_custody[target_id].name
                != custody_assignments[target_id].name
            ):
                # If the custody has changed need to do track to track handoff
                # Turn off the old custody assignment
                old_sat = current_custody[target_id]

                # Only get prior estimate if old satellite has estimation data for this target
                if (
                    not old_sat._estimator.estimation_data.empty
                    and target_id in old_sat._estimator.estimation_data.target_id.values
                ):
                    prior_est = old_sat._estimator.estimation_data[
                        old_sat._estimator.estimation_data.target_id == target_id
                    ].iloc[-1]

            if prior_est is not None:
                # Initialize the new satellite's estimator with the prior estimate
                # ASSUMING INSTANT TRANSMISSION OF ESTIMATE
                new_sat._estimator.save_current_estimation_data(
                    target_id,
                    prior_est.time,
                    prior_est.estimate,
                    prior_est.covariance,
                    prior_est.innovation,
                    prior_est.innovation_covariance,
                )
                print(
                    f"Track to track handoff for target {target_id} from {old_sat.name} to {new_sat.name}"
                )

            # Get the current track, or if it doesnt exit, just use location as fusion sats
            if target_id not in new_sat._estimator.estimation_data.target_id.values:
                # Get the exact position of the target (yes cheating)
                # Will likely happen first plan, so at t = 0
                targ = next(
                    (targ for targ in self._targs if targ.target_id == target_id), None
                )
                if targ is None:
                    continue
                pos = targ.pos
            else:
                track = new_sat._estimator.estimation_data[
                    new_sat._estimator.estimation_data.target_id == target_id
                ].iloc[-1]
                pos = track.estimate[np.array([0, 2, 4])]

            new_sat.custody[target_id] = True
            print(f'Sat {new_sat.name} has custody of target {target_id}')
            new_sat.send_bounties(target_id, pos, self.time.value, nearest_sens=10)

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

    def perfect_target_data(self):
        """
        Give the fusion satellites perfect data for the targets.
        """
        for sat in self._fusion_sats:
            # Did it recieve any measurements?
            data_received = sat._network.receive_measurements(sat.name, self.time.value)
            if not data_received:
                continue

            # Get unique target IDs from received data
            target_ids = {meas.target_id for meas in data_received}

            for target_id in target_ids:
                # Find the target
                targ = next(
                    (targ for targ in self._targs if targ.target_id == target_id), None
                )
                if targ is None:
                    continue

                # Get the exact state of the target
                targ_state = targ._state_hist.iloc[-1]
                targ_time = targ_state.iloc[0]
                # Reorder from [x,y,z,vx,vy,vz] to [x,vx,y,vy,z,vz]
                est = np.array(
                    [
                        targ_state.iloc[1],  # x
                        targ_state.iloc[4],  # vx
                        targ_state.iloc[2],  # y
                        targ_state.iloc[5],  # vy
                        targ_state.iloc[3],  # z
                        targ_state.iloc[6],  # vz
                    ]
                )

                # Apply this to the estimator
                sat._estimator.save_current_estimation_data(
                    target_id,
                    targ_time,
                    est,
                    np.eye(6),
                    np.zeros(2),
                    np.eye(2),
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
                targ._state_hist.to_csv(target_path / f'{targ.target_id}.csv')

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
