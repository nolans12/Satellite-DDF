# Import classes
import csv
import io
import os
import pathlib
import random
from collections import defaultdict
from typing import cast

import imageio
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import axes3d

from common import path_utils
from phase3 import collection
from phase3 import comms
from phase3 import estimator
from phase3 import orbit
from phase3 import satellite
from phase3 import sensor
from phase3 import sim_config
from phase3 import target


class Environment:
    def __init__(self, *args, network: comms.Comms, **kwargs):
        self.comms = network

        # Initialize time parameter to 0
        self.time = u.Quantity(0, u.minute)
        self.delta_t = None

        # Init the earth plotting parameters
        self.init_plotting()

    def init_plotting(self):
        # Environment Plotting parameters
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = cast(axes3d.Axes3D, self.fig.add_subplot(111, projection='3d'))

        # If you want to do clustered case:
        self.ax.set_xlim([2000, 8000])
        self.ax.set_ylim([-6000, 6000])
        self.ax.set_zlim([2000, 8000])
        self.ax.view_init(elev=30, azim=0)

        # auto scale the axis to be equal
        self.ax.set_box_aspect([0.5, 1, 0.5])

        # Label the axes and set title
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.set_title('Satellite Orbit Visualization')

        # Earth parameters for plotting
        u_earth = np.linspace(0, 2 * np.pi, 100)
        v_earth = np.linspace(0, np.pi, 100)
        self.earth_r = 6378.0
        self.x_earth = self.earth_r * np.outer(np.cos(u_earth), np.sin(v_earth))
        self.y_earth = self.earth_r * np.outer(np.sin(u_earth), np.sin(v_earth))
        self.z_earth = self.earth_r * np.outer(
            np.ones(np.size(u_earth)), np.cos(v_earth)
        )

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

        sensing_sats = {
            name: satellite.SensingSatellite(
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
        }

        fusion_sats = {
            name: satellite.FusionSatellite(
                name=name,
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
                local_estimator=None,
            )
            for name, s in cfg.sensing_satellites.items()
        }

        # Define the communication network:
        comms_network = comms.Comms(
            list(sensing_sats.values()),
            list(fusion_sats.values()),
            [],
            config=cfg.comms,
        )

        # Create and return an environment instance:
        return cls(
            list(sensing_sats.values()),
            list(fusion_sats.values()),
            targs,
            cfg.estimator,
            network=comms_network,
        )

    def simulate(
        self,
        time_vec: u.Quantity,
        plot_config: sim_config.PlotConfig,
        pause_step: float = 0.0001,
    ):
        """
        Simulate the environment over a time range.

        Args:
        - time_vec: Array of time steps to simulate over.
        - pause_step: Time to pause between each step in the simulation.
        - plot_config: Plotting configuration

        Returns:
        - Data collected during simulation.
        """

        print("Simulation Started")

        # Initialize based on the current time
        time_vec = time_vec + self.time
        self.delta_t = (time_vec[1] - time_vec[0]).value
        for t_net in time_vec:

            print(f"Time: {t_net:.2f}")

            # Get the delta time to propagate
            t_d = t_net - self.time

            # Propagate the environments positions
            self.propagate(t_d)

            # Get the measurements from the satellites
            self.collect_all_measurements()

            if plot_config.show_live:
                # Update the plot environment
                self.plot()
                plt.pause(pause_step)
                plt.draw()

        print("Simulation Complete")

        # Now, save all data frames, will save to "data" folder
        self.save_data_frames(saveName=plot_config.output_prefix)

        return

    def propagate(self, time_step: u.Quantity[u.minute]) -> None:
        """
        Propagate the satellites and targets over the given time step.
        """

        # Update the current time (u.Minutes)
        self.time += time_step

        # Get the float value
        time_val = self.time.value

        # Propagate the targets' positions
        for targ in self.targs:
            targ.propagate(
                time_step, time_val
            )  # Propagate and store the history of target time and xyz position and velocity

        # Propagate the satellites
        for sat in self.sats:
            sat.propagate(
                time_step, time_val
            )  # Propagate and store the history of satellite time and xyz position and velocity

        # Update the communication network for the new sat positions
        self.comms.update_edges()

    def collect_all_measurements(self) -> None:
        """
        Collect measurements from the satellites.
        """
        for sat in self.sats:
            for targ in self.targs:
                sat.collect_measurements(targ.target_id, targ.pos, self.time.to_value())

    def data_fusion(self) -> None:
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """

        # If a central estimator is present, perform central fusion
        if self.estimator_config.central:
            self.central_fusion()

        # Now send estimates for future CI
        if self.estimator_config.ci:
            # self.send_estimates_optimize()
            self.send_estimates()

        # Now send measurements for future ET
        if self.estimator_config.et:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        if self.estimator_config.ci:
            for sat in self.sats:
                # Get just the data sent using CI, (match receiver to sat name and type to 'estimate')
                data_recieved = self.comms.comm_data[
                    (self.comms.comm_data['receiver'] == sat.name)
                    & (self.comms.comm_data['type'] == 'estimate')
                ]

                sat.filter_CI(data_recieved)

            if self.estimator_config.et:
                etEKF = sat.etEstimators[0]
                etEKF.event_trigger_processing(sat, self.time.to_value(), self.comms)

        # ET estimator needs prediction to happen at everytime step, thus, even if measurement is none we need to predict
        if self.estimator_config.et:
            for sat in self.sats:
                etEKF.event_trigger_updating(sat, self.time.to_value(), self.comms)

    def send_estimates_optimize(self):
        """
        Uses mixed interger linear programming to optimize the communication network for the CI estimator.
        Will loop at track uncertainty for each sat - targ pair (assume global information)
        Then use the communication contraints to optimize the network
        Choosing which satellites should communicaiton with whom.
        """

        # Get the track uncertainties from each of the satellites into a dictionary:
        trackUncertainty = defaultdict(lambda: defaultdict(dict))

        # Loop through all satellites
        for sat in self.sats:

            # Add the satname to the dictionary
            trackUncertainty[sat] = defaultdict(dict)

            # For each targetID in satellites object to track, add the Track Uncertainty to the dictionaryg
            for targetID in sat.targetIDs:

                # Check, is there a Track Uncertainty for this targetID?
                if (
                    not bool(sat.ciEstimator.trackErrorHist[targetID])
                    or len(sat.ciEstimator.trackErrorHist[targetID].keys()) == 0
                ):
                    # Make the track uncertainty 999
                    trackUncertainty[sat.name][targetID] = 999
                    continue  # Skip to next targetID

                # Otherwise, get the most recent trackUncertainty
                maxTime = max(sat.ciEstimator.trackErrorHist[targetID].keys())

                # Add the track uncertainty to the dictionary
                trackUncertainty[sat.name][targetID] = sat.ciEstimator.trackErrorHist[
                    targetID
                ][maxTime]

        # Now that we have the track uncertainties, we can optimize the communication network

        #### MIXED INTEGER LINEAR PROGRAMMING ####
        # Redefine goodness function to be based on a source and reciever node pair, not a path:
        def goodness(
            source: satellite.Satellite,
            reciever: satellite.Satellite,
            trackUncertainty,
            targetID,
        ):
            """A paths goodness is defined as the sum of the deltas in track uncertainty on a targetID, as far as that node hasnt already recieved data from that satellite"""

            # If either the source or reciever targetUncertainty doesnt contain that targetID, reward of 0
            if (
                targetID not in trackUncertainty[source.name].keys()
                or targetID not in trackUncertainty[reciever.name].keys()
            ):
                return 0

            # Get the track uncertainty of the source node
            sourceTrackUncertainty = trackUncertainty[source.name][targetID]

            # Get the track uncertainty of the target node
            recieverTrackUncertainty = trackUncertainty[reciever.name][targetID]

            # Get the desired targetID track uncertainty, from the reciever
            desired = reciever.targPriority[targetID]

            # Check, if the sats track uncertainty on that targetID needs help or not
            if recieverTrackUncertainty < desired:
                return 0

            # Else, calculate the goodness, + if the source is better, 0 if the sat is better
            if recieverTrackUncertainty - sourceTrackUncertainty < 0:
                return 0  # EX: If i have uncertainty of 200 and share it with a sat with 100, theres no benefit to sharing that

            # Else, return the goodness of the link, difference between the two track uncertainties
            return recieverTrackUncertainty - sourceTrackUncertainty

        # Now goal is to find the set of paths that maximize the total goodness, while also respecting the bandwidth constraints and not double counting, farying information is allowed

        # Generate all possible non cyclic paths up to a reasonable length (e.g., max 3 hops)
        def generate_all_paths(graph, max_hops) -> list[list[satellite.Satellite]]:
            paths = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        for path in nx.all_simple_paths(
                            graph, source=source, target=target, cutoff=max_hops
                        ):
                            paths.append(tuple(path))
            return paths

        # Generate all possible paths
        allPaths = generate_all_paths(self.comms.G, 4)

        # Define the fixed bandwidth consumption per CI
        fixed_bandwidth_consumption = 30

        # Define the optimization problem
        prob = pulp.LpProblem("Path_Optimization", pulp.LpMaximize)

        # Create binary decision variables for each path combination
        # 1 if the path is selected, 0 otherwise
        path_selection_vars = pulp.LpVariable.dicts(
            "path_selection",
            [
                (path, targetID)
                for path in allPaths
                for targetID in trackUncertainty[path[0].name].keys()
            ],
            0,
            1,
            pulp.LpBinary,
        )

        #### OBJECTIVE FUNCTION

        ## Define the objective function, total sum of goodness across all paths
        # Initalize a linear expression that will be used as the objective
        total_goodness_expression = pulp.LpAffineExpression()

        for path in allPaths:  # Loop through all paths possible

            for targetID in trackUncertainty[
                path[0].name
            ].keys():  # Loop through all targetIDs that a path could be talking about

                # Initalize a linear expression that will define the goodness of a path in talking about a targetID
                path_goodness = pulp.LpAffineExpression()

                # Loop through the links of the path
                for i in range(len(path) - 1):

                    # Get the goodness of a link in the path on the specified targetID
                    edge_goodness = goodness(
                        path[0], path[i + 1], trackUncertainty, targetID
                    )

                    # Add the edge goodness to the path goodness
                    path_goodness += edge_goodness

                # Thus we are left with a value for the goodness of the path in talking about targetID
                # But, we dont know if we are going to take that path, thats up to the optimizer
                # So make it a binary expression, so that if the path is selected,
                # the path_goodness will be added to the total_goodness_expression.
                # Otherwsie if the path isn't selected, the path_goodness will be 0
                total_goodness_expression += (
                    path_goodness * path_selection_vars[(path, targetID)]
                )

        # Add the total goodness expression to the linear programming problem as the objective function
        prob += total_goodness_expression, "Total_Goodness_Objective"

        #### CONSTRAINTS

        ## Ensure the total bandwidth consumption across a link does not exceed the bandwidth constraints
        for edge in self.comms.G.edges():  # Loop through all edges possible
            u, v = edge  # Unpack the edge

            # Create a list to accumulate the terms for the total bandwidth usage on this edge
            bandwidth_usage_terms = []

            # Iterate over all possible paths
            for path, targetID in path_selection_vars:

                # Check if the current path includes the edge in question
                if any((path[i], path[i + 1]) == edge for i in range(len(path) - 1)):

                    # Now path_selection_vars is a binary expression/condition will either be 0 or 1
                    # Thus, the following term essentially accounts for the bandwidth usage on this edge, if its used
                    bandwidth_usage = (
                        path_selection_vars[(path, targetID)]
                        * fixed_bandwidth_consumption
                    )

                    # Add the term to the list
                    bandwidth_usage_terms.append(bandwidth_usage)

            # Sum all the expressions in the list to get the total bandwidth usage on this edge
            total_bandwidth_usage = pulp.lpSum(bandwidth_usage_terms)

            # Add the constraint to the linear programming problem
            # The constraint indicates that the total bandwidth usage on this edge should not exceed the bandwidth constraint
            # This constraint will be added for all edges in the graph after the loop
            prob += (
                total_bandwidth_usage <= self.comms.G[u][v]['maxBandwidth'],
                f"Bandwidth_constraint_{edge}",
            )

        ## Ensure the reward for sharing information about a targetID from source node to another node is not double counted
        for source in self.comms.G.nodes():  # Loop over all source nodes possible

            for (
                receiver
            ) in self.comms.G.nodes():  # Loop over all receiver nodes possible

                # If the receiver is not the source node, we can add a constraint
                if receiver != source:

                    # Loop over all targetIDs source and reciever could be talking about
                    for targetID in trackUncertainty[source.name].keys():

                        # Initalize a linear expression that will be used as a constraint
                        # This expression is exclusivly for source -> reciever about targetID, gets reinitalized every time
                        path_count = pulp.LpAffineExpression()

                        # Now we want to add the constraint that no more than 1
                        # source -> reciever about targetID is selected

                        # So we will count the number of paths that could be selected that are source -> reciever about targetID
                        for path in allPaths:

                            # Check if the path starts at the source and contains the receiver
                            if path[0] == source and receiver in path:

                                # Add the path selection variable to the path sum if its selected and talking about the targetID
                                path_count += path_selection_vars[(path, targetID)]

                        # Add a constraint to ensure the path sum is at most 1
                        # Thus, there will be a constraint for every source -> reciever about targetID combo,
                        # ensuring the total number of paths selected that contain that isn't greater than 1
                        prob += (
                            path_count <= 1,
                            f"Single_path_for_target_{source}_{receiver}_{targetID}",
                        )

        # Solve the optimization problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Output the results, paths selected
        selected_paths = [
            (path, targetID)
            for (path, targetID) in path_selection_vars
            if path_selection_vars[(path, targetID)].value() == 1
        ]

        ### SEND THE ESTIMATES

        # Now, we have the selected paths, we can send the estimates
        for path, targetID in selected_paths:

            # Get the est, cov, and time of the most recent estimate
            sourceSat = path[0]

            # Get the most recent estimate time
            sourceTime = max(sourceSat.ciEstimator.estHist[targetID].keys())
            est = sourceSat.ciEstimator.estHist[targetID][sourceTime]
            cov = sourceSat.ciEstimator.covarianceHist[targetID][sourceTime]

            # Get the most recent
            self.comms.send_estimate_path(path, est, cov, targetID, sourceTime)

    def send_estimates(self):
        """
        Send the most recent estimates from each satellite to its neighbors.

        Worst case CI, everybody sents to everybody
        """
        # Loop through all satellites
        random_sats = self.sats[:]
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
                for neighbor in self.comms.G.neighbors(sat):
                    self.comms.send_estimate(
                        sat, neighbor, est, cov, targetID, self.time.value
                    )

    def send_measurements(self):
        """
        Send the most recent measurements from each satellite to its neighbors.
        """
        # Loop through all satellites
        for sat in self.sats:
            # For each targetID in satellites measurement history
            for (
                target
            ) in self.targs:  # TODO: iniitalize with senders est and cov + noise?
                if target.targetID in sat.targetIDs:
                    targetID = target.targetID
                    envTime = self.time.value
                    # Skip if there are no measurements for this targetID
                    if isinstance(
                        sat.measurementHist[target.targetID][envTime], np.ndarray
                    ):
                        # This means satellite has a measurement for this target, now send it to neighbors
                        for neighbor in self.comms.G.neighbors(sat):
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
                                commonEKF.et_EKF_initialize(target, envTime)
                                sat.etEstimators.append(commonEKF)
                                # commonEKF.synchronizeFlag[targetID][envTime] = True

                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.et_EKF_initialize(target, envTime)

                            # Get the neighbors localEKF
                            neighbor_localEKF = neighbor.etEstimators[0]

                            # If the neighbor doesn't have a local EKF on this target, create one
                            if len(neighbor_localEKF.estHist[targetID]) == 0:
                                neighbor_localEKF.et_EKF_initialize(target, envTime)

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
                                commonEKF.et_EKF_initialize(target, envTime)
                                neighbor.etEstimators.append(commonEKF)
                                # commonEKF.synchronizeFlag[targetID][envTime] = True

                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.et_EKF_initialize(target, envTime)

                            # Create implicit and explicit measurements vector for this neighbor
                            alpha, beta = local_EKF.event_trigger(
                                sat, neighbor, targetID, satTime
                            )

                            # Send that to neightbor
                            self.comms.send_measurements(
                                sat, neighbor, alpha, beta, targetID, satTime
                            )

                            if commonEKF.synchronizeFlag[targetID][envTime]:
                                # Since this runs twice, we need to make sure we don't double count the data
                                self.comms.total_comm_et_data.append(
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

        # Get all measurements taken by the satellites on each target
        for targ in self.targs:

            targetID = targ.targetID
            measurements = (
                []
            )  # needs to look like [array([44.21412194,  2.32977878]), array([41.48069992, 29.14226864])] for [alpha, beta] for each satellite

            sats_w_measurements = []

            for sat in self.sats:
                if targetID in sat.measurementHist['targetID'].values:

                    # Get the measurement at the current time
                    meas = sat.measurementHist[
                        (sat.measurementHist['targetID'] == targetID)
                        & (sat.measurementHist['time'] == self.time.value)
                    ]['measurement'].values[0]

                    if meas is not None:
                        measurements.append(meas)
                        sats_w_measurements.append(sat)
            # Now, perform the central fusion, using the estimator, decide to do initization or pred then update
            if len(measurements) > 0:

                if self.estimator.estimation_data.empty:
                    # Initialize
                    self.estimator.EKF_initialize(targ, self.time.value)
                else:
                    # Check, do we have an estimate for this targetID?
                    if targetID in self.estimator.estimation_data['targetID'].values:
                        # Predict
                        self.estimator.EKF_pred(targetID, self.time.value)
                        # Update
                        self.estimator.EKF_update(
                            sats_w_measurements, measurements, targetID, self.time.value
                        )
                    else:
                        # Initialize
                        self.estimator.EKF_initialize(targ, self.time.value)

    def send_to_ground_best_sat(self):
        """
        Planner for sending data from the satellite network to the ground station.

        DDF, BEST SAT:
            Choose the satellite from the network with the lowest track uncertainty, that can communicate with ground station.
            For that sat, send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:

            # Loop through all targets (assume the GS cares about all for now))
            for targ in self.targs:

                # Figure out which satellite has the lowest track uncertainty matrix for this targetID
                bestSat = None
                bestTrackUncertainty = float('inf')
                bestData = None
                for sat in self.sats:

                    # Get the estimator data for this satellite
                    estimator_data = sat.estimator.estimation_data

                    # Check, does this satellite have data for this targetID?
                    if estimator_data.empty:
                        continue
                    if targ.targetID in estimator_data['targetID'].values:

                        # Get the data and track uncertainty
                        data = estimator_data[
                            estimator_data['targetID'] == targ.targetID
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
                        target_id=targ.targetID,
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
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def save_data_frames(
        self,
        saveName: str,
        savePath: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data'
        ),
    ) -> None:
        """
        Save all data frames to csv files.

        Input is a folder name, "saveName"
        A folder will be made in the data folder with this name, and all data frames will be saved in there.
        """

        # Make a folder for "data" if it doesn't exist
        os.makedirs(savePath, exist_ok=True)

        # Then make a folder for the saveName in data
        savePath = os.path.join(savePath, saveName)
        os.makedirs(savePath, exist_ok=True)

        # Now, save all data frames

        ## The data we have is:
        # - Target state data (done)

        # - Satellite state data (done)
        # - Satellite estimator data (est, cov, innovaiton, etc) (NOT DONE)

        # - Ground station estimate and covariance data (NOT DONE)

        # - Communications to and from satellites and ground stations (kinda done)

        ## Make a target state folder
        target_path = os.path.join(savePath, f"targets")
        os.makedirs(target_path, exist_ok=True)
        for targ in self.targs:
            targ._state_hist.to_csv(os.path.join(target_path, f"{targ.name}.csv"))

        ## Make a satellite state folder
        satellite_path = os.path.join(savePath, f"satellites")
        os.makedirs(satellite_path, exist_ok=True)
        for sat in self.sats:
            sat._state_hist.to_csv(
                os.path.join(satellite_path, f"{sat.name}_state.csv")
            )
            sat._measurement_hist.to_csv(
                os.path.join(satellite_path, f"{sat.name}_measurements.csv")
            )

        # ## Make a comms folder # TODO: fix with ryans new data class
        # comms_path = os.path.join(savePath, f"comms")
        # os.makedirs(comms_path, exist_ok=True)
        # if not self.comms.comm_data.empty:
        #     self.comms.comm_data.to_csv(os.path.join(comms_path, f"comm_data.csv"))

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

        # if self.estimator_config.central:
        #     if self.estimator is not None:
        #         self.estimator.estimation_data.to_csv(
        #             os.path.join(estimator_path, f"central_estimator.csv")
        #         )

    ### 3D Dynamic Environment Plot ###
    def plot(self) -> None:
        """
        Plot the current state of the environment.
        """
        self.resetPlot()
        self.plotEarth()
        self.plotSatellites()
        self.plotTargets()
        self.plotCommunication()
        self.plotGroundStations()
        self.plotLegend_Time()
        self.save_envPlot_to_imgs()

    def resetPlot(self) -> None:
        """
        Reset the plot by removing all lines, collections, and texts.
        """
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()
        for text in self.ax.texts:
            text.remove()

    def plotSatellites(self) -> None:
        """
        Plot the current state of each satellite, including their positions and sensor projections.
        """
        for sat in self.sats:
            # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value
            # Cut the label of a satellite off before the first underscore
            satName = sat.name.split('.')[0]
            self.ax.scatter(x, y, z, s=40, color=sat.color, label=satName)

            # Plot the visible projection of the satellite sensor
            points = sat.sensor.projBox
            self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2], color=sat.color, marker='x'
            )

            box = np.array([points[0], points[3], points[1], points[2], points[0]])
            self.ax.add_collection3d(
                art3d.Poly3DCollection(
                    [box],
                    facecolors=sat.color,
                    linewidths=1,
                    edgecolors=sat.color,
                    alpha=0.1,
                )
            )

    def plotTargets(self) -> None:
        """
        Plot the current state of each target, including their positions and velocities.
        """
        for targ in self.targs:
            # Plot the current xyz location of the target
            x, y, z = targ.pos
            vx, vy, vz = targ.vel
            mag = np.linalg.norm([vx, vy, vz])
            if mag > 0:
                vx, vy, vz = vx / mag, vy / mag, vz / mag

            # do a standard scatter plot for the target
            self.ax.scatter(
                x, y, z, s=40, marker='x', color=targ.color, label=targ.name
            )

    def plotEarth(self) -> None:
        """
        Plot the Earth's surface.
        """
        self.ax.plot_surface(
            self.x_earth, self.y_earth, self.z_earth, color='k', alpha=0.1
        )
        # ### ALSO USE IF YOU WANT EARTH TO NOT BE SEE THROUGH
        # self.ax.plot_surface(self.x_earth*0.9, self.y_earth*0.9, self.z_earth*0.9, color = 'white', alpha=1)

    def plotCommunication(self) -> None:
        """
        Plot the communication structure between satellites.
        """
        if self.comms.displayStruct:
            for edge in self.comms.G.edges:
                sat1 = edge[0]
                sat2 = edge[1]
                x1, y1, z1 = sat1.orbit.r.value
                x2, y2, z2 = sat2.orbit.r.value
                if self.comms.G.edges[edge]['active']:
                    self.ax.plot(
                        [x1, x2],
                        [y1, y2],
                        [z1, z2],
                        color=(0.3, 1.0, 0.3),
                        linestyle='dashed',
                        linewidth=2,
                    )
                else:
                    self.ax.plot(
                        [x1, x2],
                        [y1, y2],
                        [z1, z2],
                        color='k',
                        linestyle='dashed',
                        linewidth=1,
                    )

    def plotGroundStations(self) -> None:
        """
        Plot the ground stations.
        """
        for gs in self.groundStations:
            x, y, z = gs.loc
            self.ax.scatter(x, y, z, s=40, marker='s', color=gs.color, label=gs.name)

    def plotLegend_Time(self) -> None:
        """
        Plot the legend and the current simulation time.
        """
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.ax.text2D(
            0.05, 0.95, f"Time: {self.time:.2f}", transform=self.ax.transAxes
        )

    def save_envPlot_to_imgs(self) -> None:
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        self.imgs.append(img)

    def render_gifs(
        self,
        plot_config: sim_config.PlotConfig,
        save_name: str,
        out_dir: pathlib.Path = path_utils.PHASE_3 / 'gifs',
        fps: int = 5,
    ):
        """
        Renders and saves GIFs based on the specified file type.

        Parameters:
        - plot_config: The type of GIF to render. Options are 'satellite_simulation' or 'uncertainty_ellipse'.
        - save_name: The base name for the saved GIF files.
        - out_dir: The directory path where the GIF files will be saved. Defaults to the directory of the script.
        - fps: Frames per second for the GIF. Defaults to 10.

        Returns:
        None
        """
        frame_duration = 1000 / fps  # in ms

        if sim_config.GifType.SATELLITE_SIMULATION in plot_config.gifs:
            file = out_dir / f'{save_name}_satellite_sim.gif'
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self.imgs:
                    writer.append_data(img)

        if sim_config.GifType.UNCERTAINTY_ELLIPSE in plot_config.gifs:
            for targ in self.targs:
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        for sat2 in self.sats:
                            if targ.targetID in sat2.targetIDs:
                                if sat != sat2:
                                    file = (
                                        out_dir
                                        / f"{save_name}_{targ.name}_{sat.name}_{sat2.name}_stereo_GE.gif"
                                    )
                                    with imageio.get_writer(
                                        file, mode='I', duration=frame_duration
                                    ) as writer:
                                        for img in self.imgs_stereo_GE[targ.targetID][
                                            sat
                                        ][sat2]:
                                            writer.append_data(img)

        if sim_config.GifType.DYNAMIC_COMMS in plot_config.gifs:
            file = out_dir / f"{save_name}_dynamic_comms.gif"
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self.imgs_dyn_comms:
                    writer.append_data(img)
