# Import classes
import csv
import io
import os
import pathlib
import random
from collections import defaultdict
from typing import cast

import imageio
import networkx as nx
import numpy as np
import pandas as pd
import pulp
from astropy import units as u
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import axes3d
from numpy import typing as npt

from common import path_utils
from phase3 import collection
from phase3 import comms
from phase3 import estimator
from phase3 import groundStation
from phase3 import orbit
from phase3 import satellite
from phase3 import sensor
from phase3 import sim_config
from phase3 import target
from phase3 import util

## Creates the environment class, which contains a vector of satellites all other parameters


class Environment:
    def __init__(
        self,
        sats: list[satellite.Satellite],
        targs: list[target.Target],
        comms: comms.Comms,
        groundStations: list[groundStation.GroundStation],
        commandersIntent: util.CommandersIndent,
        estimator_config: sim_config.EstimatorConfig,
    ):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """
        self.estimator_config = estimator_config

        # For each satellite, define its initial goal and initialize the estimation algorithms
        for sat in sats:
            targPriorityInitial = commandersIntent[0][
                sat.name
            ]  # grab the initial target priorities for the satellite

            # Update the Satellite with the initial target priorities
            sat.targPriority = (
                targPriorityInitial  # dictionary of [targetID: priority] pairs
            )
            sat.targetIDs = list(targPriorityInitial.keys())  # targetID to track
            sat.measurementHist = {
                targetID: defaultdict(dict) for targetID in targPriorityInitial.keys()
            }  # dictionary of [targetID: {time: measurement}] pairs

            # Create estimation algorithms for each satellite
            if self.estimator_config.local:
                sat.indeptEstimator = estimator.IndeptEstimator()  # initialize the independent estimator for these targets

            if self.estimator_config.central:
                self.centralEstimator = estimator.CentralEstimator()

            if self.estimator_config.ci:
                sat.ciEstimator = estimator.CiEstimator()

            if self.estimator_config.et:
                sat.etEstimators = [
                    estimator.EtEstimator(commandersIntent[0][sat.name], shareWith=None)
                ]

        ## Populate the environment variables
        self.sats = sats  # define the satellites

        self.targs = targs  # define the targets

        self.commandersIntent = commandersIntent  # define the commanders intent

        self.comms = comms  # define the communication network

        self.groundStations = groundStations  # define the ground stations

        # Initialize time parameter to 0
        self.time = u.Quantity(0, u.minute)
        self.delta_t = None

        # Environemnt Plotting parameters
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

        # Empty lists and dictionaries for simulation images
        self.imgs = []  # dictionary for 3D gif images
        self.imgs_dyn_comms = []  # dictionary for dynamic 2D comms gif images

        # Nested Dictionary for storing stereo estimation plots
        self.imgs_stereo_GE = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.tempCount = 0

    def simulate(
        self,
        time_vec: u.Quantity,
        plot_config: sim_config.PlotConfig,
        pause_step: float = 0.0001,
        save_estimation_data: bool = False,
        save_communication_data: bool = False,
    ):
        """
        Simulate the environment over a time range.

        Args:
        - time_vec: Array of time steps to simulate over.
        - pause_step: Time to pause between each step in the simulation.
        - plot_config: Plotting configuration
        - save_estimation_data: Flag to save the estimation data.
        - save_communication_data: Flag to save the communication data.

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

            # Does the commanders intent change for the satellites at this time?
            if t_net.to_value(t_net.unit) in self.commandersIntent.keys():
                for sat in self.sats:
                    sat.targPriority = self.commandersIntent[
                        t_net.to_value(t_net.unit)
                    ][sat.name]
                    sat.targetIDs = list(sat.targPriority.keys())

            # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion()

            # Have the network send information to the ground station
            # self.send_to_ground_god_mode()
            # self.send_to_ground_centralized()
            # self.send_to_ground_avaliable_sats()
            ## MUST CHOOSE THIS TO MATCH THE ACTUAL DDF YOUR DOING
            # self.send_to_ground_best_sat_local()
            # self.send_to_ground_best_sat_ci()
            # self.send_to_ground_best_sat_et()

            if plot_config.show_env:
                # Update the plot environment
                self.plot()
                plt.pause(pause_step)
                plt.draw()

            if plot_config.plot_et_network:
                # Update the dynamic comms plot
                self.plot_dynamic_comms()

        print("Simulation Complete")

        # Now, save all data frames, will save to "data" folder
        self.save_data_frames(saveName=plot_config.output_prefix)

        ###### WE ARE TRANSITIONING TO MOVING ALL DATA TO BE SAVED IN A DATA FRAME, THEN PLOTTING FROM CSVS ######

        # if plot_config.plot_groundStation_results:
        #     self.plot_gs_results(
        #         time_vec, saveName=plot_config.output_prefix
        #     )  # Plot the ground station results

        # # Plot the filter results
        # if plot_config.plot_estimation:
        #     self.plot_estimator_results(
        #         time_vec, saveName=plot_config.output_prefix
        #     )  # marginal error, innovation, and NIS/NEES plots

        # # Plot the commm results
        # if plot_config.plot_communication:

        #     # Make plots for total data sent and used throughout time
        #     self.plot_global_comms(saveName=plot_config.output_prefix)
        #     # self.plot_used_comms(saveName=plot_config.output_prefix)

        #     # For the CI estimators, plot time hist of comms
        #     if self.estimator_config.ci:
        #         self.plot_timeHist_comms_ci(saveName=plot_config.output_prefix)

        # # Save the uncertainty ellipse plots
        # if plot_config.plot_uncertainty_ellipses:
        #     self.plot_all_uncertainty_ellipses(time_vec)  # Uncertainty Ellipse Plots

        # # Log the Data
        # if save_estimation_data:
        #     self.log_data(time_vec, saveName=plot_config.output_prefix)

        # if save_communication_data:
        #     self.log_comms_data(time_vec, saveName=plot_config.output_prefix)

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

    def data_fusion(self) -> None:
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """
        # Collect all measurements for every satellite in the environement
        collectedFlag, measurements = self.collect_all_measurements()

        # If a central estimator is present, perform central fusion
        if self.estimator_config.central:
            self.central_fusion(collectedFlag, measurements)

        # Now send estimates for future CI
        if self.estimator_config.ci:
            self.send_estimates_optimize()
            # self.send_estimates()

        # Now send measurements for future ET
        if self.estimator_config.et:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            if self.estimator_config.ci:
                sat.ciEstimator.CI(sat, self.comms)

            if self.estimator_config.et:
                etEKF = sat.etEstimators[0]
                etEKF.event_trigger_processing(sat, self.time.to_value(), self.comms)

        # ET estimator needs prediction to happen at everytime step, thus, even if measurement is none we need to predict
        for sat in self.sats:
            if self.estimator_config.et:
                etEKF.event_trigger_updating(sat, self.time.to_value(), self.comms)

    def collect_all_measurements(self) -> tuple[defaultdict, defaultdict]:
        """
        Collect measurements from satellites for all available targets.

        Returns:
        - collectedFlag: dictionary tracking which satellites collected measurements for each target.
        - measurements: dictionary storing measurements collected for each target by each satellite.
        """
        collectedFlag = defaultdict(lambda: defaultdict(dict))
        measurements = defaultdict(lambda: defaultdict(dict))

        # Collect measurements on any available targets
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Collect the bearing measurement, if available, and run an EKF to update the estimate on the target
                    collectedFlag[targ][sat] = sat.collect_measurements_and_filter(targ)

                    if collectedFlag[targ][sat]:  # If a measurement was collected
                        measurements[targ][sat] = sat.measurementHist[targ.targetID][
                            self.time.to_value()
                        ]  # Store the measurement in the dictionary

        return (
            collectedFlag,
            measurements,
        )  # Return the collected flags and measurements

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
        """
        # Loop through all satellites
        random_sats = self.sats[:]
        random.shuffle(random_sats)
        for sat in random_sats:
            # For each targetID in the satellite estimate history

            # also shuffle the targetIDs
            shuffle_targetIDs = list(sat.ciEstimator.estHist.keys())
            random.shuffle(shuffle_targetIDs)
            for targetID in shuffle_targetIDs:

                # Skip if there are no estimates for this targetID
                if isinstance(
                    sat.measurementHist[targetID][self.time.to_value()], np.ndarray
                ):  # len(sat.ciEstimator.estHist[targetID].keys()) == 0:

                    # This means satellite has an estimate for this target, now send it to neighbors
                    neighbors = list(self.comms.G.neighbors(sat))
                    random.shuffle(neighbors)
                    for neighbor in neighbors:

                        # Check, does that neighbor care about that target?
                        if targetID not in neighbor.targetIDs:
                            continue

                        # Get the most recent estimate time
                        satTime = max(sat.ciEstimator.estHist[targetID].keys())

                        est = sat.ciEstimator.estHist[targetID][satTime]
                        cov = sat.ciEstimator.covarianceHist[targetID][satTime]

                        # Send most recent estimate to neighbor
                        self.comms.send_estimate(
                            sat, neighbor, est, cov, targetID, satTime
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

    def central_fusion(self, collectedFlag, measurements):
        """
        Perform central fusion using collected measurements.

        Args:
        - collectedFlag: dictionary tracking which satellites collected measurements for each target.
        - measurements: dictionary storing measurements collected for each target by each satellite.
        """
        # Now do central fusion
        for targ in self.targs:
            # Extract satellites that took a measurement
            satsWithMeasurements = [
                sat for sat in self.sats if collectedFlag[targ][sat]
            ]
            newMeasurements = [measurements[targ][sat] for sat in satsWithMeasurements]

            targetID = targ.targetID
            # If any satellite took a measurement on this target
            if satsWithMeasurements:
                # Run EKF with all satellites that took a measurement on the target
                if len(self.centralEstimator.estHist[targ.targetID]) < 1:
                    self.centralEstimator.central_EKF_initialize(
                        targ, self.time.to_value()
                    )
                    return
                self.centralEstimator.central_EKF_pred(targetID, self.time.to_value())
                self.centralEstimator.central_EKF_update(
                    satsWithMeasurements,
                    newMeasurements,
                    targetID,
                    self.time.to_value(),
                )

    ## GROUND STATION PROTOCOLS, NEED TO CHOOSE JUST ONE MAYBE
    def send_to_ground_god_mode(self):
        """
        Planner for sending data from the satellite network to the ground station.

        GOD MODE:
            Every satellite that took a measurement on a target, send the measurement to the ground station.
            Regardless of if the sat can communicate with the ground station.
        """

        # TODO: this function is kinda hard because the EKF estimator needs the true position of the targets to initalize.
        # But most other things just use targetID, thus, have to get target in the data queue even though only need targetID for everything besides the initalization.

        # For each satellite, check, did it get a new measurement?
        # Compare the environment time with most recent measurement history time

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue
                # Check to see if a satellite has a measurement for that targetID at the given time
                for sat in self.sats:
                    if isinstance(
                        sat.measurementHist[targ.targetID][self.time.to_value()],
                        np.ndarray,
                    ):

                        # Get the measurement
                        meas = sat.measurementHist[targ.targetID][self.time.to_value()]

                        data = collection.GsMeasurementTransmission(
                            target_id=targ.targetID,
                            sender=sat.name,
                            receiver=gs.name,
                            time=self.time.value,
                            measurement=meas,
                        )

                        # Add the data to the queued data onboard the ground station
                        gs.queue_data(data, dtype=collection.GsDataType.MEAS)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def send_to_ground_centralized(self):
        """
        Planner for sending data from the satellite network to the ground station.

        CENTRALIZED:
            For every satellite that can communicate with the ground station and took a measurement on a target,
            send the measurement for that target to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue
                # Check to see if a satellite has a measurement for that targetID at the given time
                for sat in self.sats:

                    if isinstance(
                        sat.measurementHist[targ.targetID][self.time.to_value()],
                        np.ndarray,
                    ):

                        # Check, can the satellite communicate with the ground station?
                        x_sat, y_sat, z_sat = sat.orbit.r.value

                        if gs.can_communicate(x_sat, y_sat, z_sat):

                            # print(f"Satellite {sat.name} can communicate with ground station {gs.name}")

                            # Get the measurement
                            meas = sat.measurementHist[targ.targetID][
                                self.time.to_value()
                            ]

                            data = collection.GsMeasurementTransmission(
                                target_id=targ.targetID,
                                sender=sat.name,
                                receiver=gs.name,
                                time=self.time.value,
                                measurement=meas,
                            )

                            # Add the data to the queued data onboard the ground station
                            gs.queue_data(data, dtype=collection.GsDataType.MEAS)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def send_to_ground_avaliable_sats(self):
        """
        Planner for sending data from the satellite network to the ground station.

        DDF, ALL SATS:
            All satellites in the network that can communicate with ground station send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue
                # Check to see if a satellite has a estimate for that targetID at the given time
                for sat in self.sats:

                    # Is time in the estimate history? # CANT DO IS INSTANCE OTHERWISE CREATES AN EMPTY DICT
                    if len(sat.ciEstimator.estHist[targ.targetID].keys()) > 0:

                        # Check, can the satellite communicate with the ground station?
                        x_sat, y_sat, z_sat = sat.orbit.r.value

                        if gs.can_communicate(x_sat, y_sat, z_sat):

                            # Get the most recent estimate and covariance, and just send that. even if the estimator doesnt fuse with it cause its stale
                            timePrior = max(
                                sat.ciEstimator.estHist[targ.targetID].keys()
                            )

                            # Get the estimate and covariance
                            est = sat.ciEstimator.estHist[targ.targetID][timePrior]
                            cov = sat.ciEstimator.covarianceHist[targ.targetID][
                                timePrior
                            ]

                            data = collection.GsEstimateTransmission(
                                target_id=targ.targetID,
                                sender=sat.name,
                                receiver=gs.name,
                                time=self.time.value,
                                estimate=est,
                                covariance=cov,
                            )

                            # Add the data to the queued data onboard the ground station
                            gs.queue_data(data, dtype=collection.GsDataType.CI)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def send_to_ground_best_sat_local(self):
        """
        Planner for sending data from the satellite network to the ground station.

        DDF, BEST SAT:
            Choose the satellite from the network with the lowest track uncertainty, that can communicate with ground station.
            For that sat, send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue

                # Figre out which satellite has the lowest track uncertainty matrix for this targetID
                bestSat = None
                bestTrackUncertainty = 999
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        if (
                            len(
                                sat.indeptEstimator.trackErrorHist[targ.targetID].keys()
                            )
                            > 0
                        ):
                            x_sat_cur, y_sat_cur, z_sat_cur = sat.orbit.r.value
                            if gs.can_communicate(x_sat_cur, y_sat_cur, z_sat_cur):
                                timePrior = max(
                                    sat.indeptEstimator.trackErrorHist[
                                        targ.targetID
                                    ].keys()
                                )
                                trackUncertainty = sat.indeptEstimator.trackErrorHist[
                                    targ.targetID
                                ][timePrior]
                                if trackUncertainty < bestTrackUncertainty:
                                    bestTrackUncertainty = trackUncertainty
                                    bestSat = sat

                # If a satellite was found
                if bestSat is not None:

                    # Get the most recent estimate and covariance, and just send that. even if the estimator doesnt fuse with it cause its stale
                    timePrior = max(
                        bestSat.indeptEstimator.estHist[targ.targetID].keys()
                    )

                    # Get the estimate and covariance
                    est = bestSat.indeptEstimator.estHist[targ.targetID][timePrior]
                    cov = bestSat.indeptEstimator.covarianceHist[targ.targetID][
                        timePrior
                    ]

                    data = collection.GsEstimateTransmission(
                        target_id=targ.targetID,
                        sender=bestSat.name,
                        receiver=gs.name,
                        time=self.time.value,
                        estimate=est,
                        covariance=cov,
                    )

                    # Add the data to the queued data onboard the ground station
                    gs.queue_data(data, dtype=collection.GsDataType.CI)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def send_to_ground_best_sat_ci(self):
        """
        Planner for sending data from the satellite network to the ground station.

        DDF, BEST SAT:
            Choose the satellite from the network with the lowest track uncertainty, that can communicate with ground station.
            For that sat, send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue

                # Figre out which satellite has the lowest track uncertainty matrix for this targetID
                bestSat = None
                bestTrackUncertainty = 999
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        if (
                            len(sat.ciEstimator.trackErrorHist[targ.targetID].keys())
                            > 0
                        ):
                            x_sat_cur, y_sat_cur, z_sat_cur = sat.orbit.r.value
                            if gs.can_communicate(x_sat_cur, y_sat_cur, z_sat_cur):
                                timePrior = max(
                                    sat.ciEstimator.trackErrorHist[targ.targetID].keys()
                                )
                                trackUncertainty = sat.ciEstimator.trackErrorHist[
                                    targ.targetID
                                ][timePrior]
                                if trackUncertainty < bestTrackUncertainty:
                                    bestTrackUncertainty = trackUncertainty
                                    bestSat = sat

                # If a satellite was found
                if bestSat is not None:

                    # Get the most recent estimate and covariance, and just send that. even if the estimator doesnt fuse with it cause its stale
                    timePrior = max(bestSat.ciEstimator.estHist[targ.targetID].keys())

                    # Get the estimate and covariance
                    est = bestSat.ciEstimator.estHist[targ.targetID][timePrior]
                    cov = bestSat.ciEstimator.covarianceHist[targ.targetID][timePrior]

                    data = collection.GsEstimateTransmission(
                        target_id=targ.targetID,
                        sender=bestSat.name,
                        receiver=gs.name,
                        time=self.time.value,
                        estimate=est,
                        covariance=cov,
                    )

                    # Add the data to the queued data onboard the ground station
                    gs.queue_data(data, dtype=collection.GsDataType.CI)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

    def send_to_ground_best_sat_et(self):
        # TODO: MERGE SUCH THAT DONT NEED SEPERATE ET CALL?

        """
        Planner for sending data from the satellite network to the ground station.

        DDF, BEST SAT:
            Choose the satellite from the network with the lowest track uncertainty, that can communicate with ground station.
            For that sat, send a CI fusion estimate to the ground station.
        """

        # For each ground station
        for gs in self.groundStations:
            # For each targetID the ground station cares about
            for targ in self.targs:
                if targ.targetID not in gs.estimator.targs:
                    continue

                # Figre out which satellite has the lowest track uncertainty matrix for this targetID
                bestSat = None
                bestTrackUncertainty = 999
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        if (
                            len(
                                sat.etEstimators[0].trackErrorHist[targ.targetID].keys()
                            )
                            > 0
                        ):
                            x_sat_cur, y_sat_cur, z_sat_cur = sat.orbit.r.value
                            if gs.can_communicate(x_sat_cur, y_sat_cur, z_sat_cur):
                                timePrior = max(
                                    sat.etEstimators[0]
                                    .trackErrorHist[targ.targetID]
                                    .keys()
                                )
                                trackUncertainty = sat.etEstimators[0].trackErrorHist[
                                    targ.targetID
                                ][timePrior]
                                if trackUncertainty < bestTrackUncertainty:
                                    bestTrackUncertainty = trackUncertainty
                                    bestSat = sat

                # If a satellite was found
                if bestSat is not None:

                    # Get the most recent estimate and covariance, and just send that. even if the estimator doesnt fuse with it cause its stale
                    timePrior = max(
                        bestSat.etEstimators[0].trackErrorHist[targ.targetID].keys()
                    )

                    # Get the estimate and covariance
                    est = bestSat.etEstimators[0].estHist[targ.targetID][timePrior]
                    cov = bestSat.etEstimators[0].covarianceHist[targ.targetID][
                        timePrior
                    ]

                    data = collection.GsEstimateTransmission(
                        target_id=targ.targetID,
                        sender=bestSat.name,
                        receiver=gs.name,
                        time=self.time.value,
                        estimate=est,
                        covariance=cov,
                    )

                    # Add the data to the queued data onboard the ground station
                    gs.queue_data(data, dtype=collection.GsDataType.CI)

        # Now that the data is queued, process the data in the filter
        for gs in self.groundStations:
            gs.process_queued_data(self.sats, self.targs)

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
            targ.stateHist.to_csv(os.path.join(target_path, f"{targ.name}.csv"))

        ## Make a satellite state folder
        satellite_path = os.path.join(savePath, f"satellites")
        os.makedirs(satellite_path, exist_ok=True)
        for sat in self.sats:
            sat.stateHist.to_csv(os.path.join(satellite_path, f"{sat.name}.csv"))

        ## Make a comms folder
        comms_path = os.path.join(savePath, f"comms")
        os.makedirs(comms_path, exist_ok=True)
        if len(self.comms.total_comm_data) > 0:
            self.comms.total_comm_data.to_csv(
                os.path.join(comms_path, f"comm_data.csv")
            )

        ## Make an estimator folder
        estimator_path = os.path.join(savePath, f"estimators")
        os.makedirs(estimator_path, exist_ok=True)
        # Now, save all estimator data
        for sat in self.sats:
            sat.indeptEstimator.estimation_data.to_csv(  # TODO: this should just be sat.estimator.estimation_data, need to change naming syntax
                os.path.join(estimator_path, f"{sat.name}_estimator.csv")
            )

    ### Estimation Errors and Track Uncertainty Plots ###
    def plot_gs_results(self, time_vec: u.Quantity[u.minute], saveName: str) -> None:
        """
        Makes one plot for each ground station target pairing.

        The plot is a subplot where the top 3 plots are the x, y, z position errors and covariances for the estimator
        The bottom 2 plots contain the track uncertainty and then the comm sent/recieved plot for the ground station.
        """

        plt.close('all')
        state_labels = ['X [km]', 'Y [km]', 'Z [km]']

        for gs in self.groundStations:
            for targ in self.targs:
                if targ.targetID in gs.estimator.targs:

                    # Create a 2x3 grid of subplots, where the second row will have 2 plots
                    fig = plt.figure(figsize=(15, 8))
                    fig.suptitle(f"{saveName}: {gs.name}, {targ.name}", fontsize=14)
                    grid = gridspec.GridSpec(2, 6)

                    # Make the subplots:
                    axX = fig.add_subplot(grid[0, 0:2])
                    axX.set_xlabel('Time [min]')
                    axX.set_ylabel('X Position Error [km]')
                    axY = fig.add_subplot(grid[0, 2:4])
                    axY.set_xlabel('Time [min]')
                    axY.set_ylabel('Y Position Error [km]')
                    axZ = fig.add_subplot(grid[0, 4:6])
                    axZ.set_xlabel('Time [min]')
                    axZ.set_ylabel('Z Position Error [km]')
                    axTU = fig.add_subplot(grid[1, 0:3])
                    axTU.set_xlabel('Time [min]')
                    axTU.set_ylabel('Track Uncertainty [km]')
                    axComm = fig.add_subplot(grid[1, 3:6])
                    axComm.set_xlabel('Time [min]')
                    axComm.set_ylabel('Comm Sent/Recieved')

                    # Now actually plot the data
                    targetID = targ.targetID
                    trueHist = targ.hist

                    # Plot the x, y, z position errors and covariances
                    (
                        times,
                        estHist,
                        covHist,
                        innovationHist,
                        innovationCovHist,
                        trackErrorHist,
                    ) = self.getEstimationHistory(
                        targetID, time_vec, filter=gs.estimator
                    )
                    # Plot the errors
                    self.plot_errors(
                        [axX],
                        times,
                        estHist,
                        trueHist,
                        covHist,
                        label_color=gs.color,
                        linewidth=2.5,
                    )
                    self.plot_errors(
                        [axY],
                        times,
                        estHist,
                        trueHist,
                        covHist,
                        label_color=gs.color,
                        linewidth=2.5,
                    )
                    self.plot_errors(
                        [axZ],
                        times,
                        estHist,
                        trueHist,
                        covHist,
                        label_color=gs.color,
                        linewidth=2.5,
                    )

                    # Now plot the track uncertainty
                    axTU.plot(
                        trackErrorHist.keys(),
                        trackErrorHist.values(),
                        label='Track Uncertainty',
                        color=gs.color,
                        linewidth=2.5,
                    )
                    axTU.set_ylim(bottom=0)

                    # Now, we want to plot the communication sent/recieved
                    # The goal of this is a bar plot vs time.
                    # Where there is a bar for each satellite in the network at everytime step,
                    # But, the bars are only filled of the satellite communicated with the ground station at that time:
                    # Thus, the bar plot will show the communication structure of the network
                    target_estimates = gs.comm_ci_data.loc[
                        gs.comm_ci_data['targetID'] == targ.targetID
                    ]
                    target_measures = gs.comm_meas_data.loc[
                        gs.comm_meas_data['targetID'] == targ.targetID
                    ]
                    times = sorted(
                        set(target_estimates['time']).union(
                            set(target_measures['time'])
                        )
                    )

                    prevData = 0
                    for time in times:
                        target_estimates_t = target_estimates.loc[
                            target_estimates['time'] == time
                        ]
                        target_measures_t = target_measures.loc[
                            target_measures['time'] == time
                        ]

                        transmissions = gs.comm_ci_data.to_dataclasses(
                            target_estimates_t
                        ) + gs.comm_meas_data.to_dataclasses(target_measures_t)
                        sats_uncommunicated = set(self.sats)
                        for transmission in transmissions:
                            # Find the target that has that targetID:
                            targ = next(
                                filter(
                                    lambda t: t.targetID == transmission.target_id,
                                    self.targs,
                                )
                            )

                            sat = next(
                                filter(
                                    lambda s: transmission.sender == s.name, self.sats
                                )
                            )

                            # If the satellite did communicate, plot a solid box with satellite color
                            axComm.bar(
                                time,
                                1,
                                bottom=prevData,
                                color=sat.color,
                                edgecolor='k',
                                width=self.delta_t,
                            )
                            prevData += 1
                            sats_uncommunicated.remove(sat)

                        # If the satellite didint communcate with the ground station at that time
                        for sat in sats_uncommunicated:
                            # Get the position of the satellite at that time, use sat.orbitHist
                            x_sat, y_sat, z_sat = sat.orbitHist[time]
                            if gs.can_communicate(x_sat, y_sat, z_sat):
                                # If the satellite could have communicated but didn't, plot a hatched box with the satellite's color
                                axComm.bar(
                                    time,
                                    1,
                                    bottom=prevData,
                                    color='w',
                                    edgecolor=sat.color,
                                    hatch='//',
                                    linewidth=0,
                                    width=self.delta_t,
                                )
                                prevData += 1
                            else:
                                # If the satellite couldn't have communicated, plot black hatch
                                axComm.bar(
                                    time,
                                    1,
                                    bottom=prevData,
                                    color='w',
                                    edgecolor='k',
                                    hatch='//',
                                    linewidth=0,
                                    width=self.delta_t,
                                )
                                prevData += 1
                        if len(sats_uncommunicated) == len(self.sats):
                            # Case where no sats communicated to gs, plot black hatches for all
                            for _ in range(len(self.sats)):
                                axComm.bar(
                                    time,
                                    1,
                                    bottom=prevData,
                                    color='w',
                                    edgecolor='k',
                                    hatch='//',
                                    linewidth=0,
                                    width=self.delta_t,
                                )
                                prevData += 1

                    # Make a patch for legend for the satellite colors
                    handles = [
                        patches.Patch(color=gs.color, label=f'{gs.name} Estimator')
                    ]
                    for sat in self.sats:
                        handles.append(patches.Patch(color=sat.color, label=sat.name))
                    axComm.legend(
                        handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left'
                    )

                    # Adjust layout for better spacing
                    plt.tight_layout()

                    filePath = os.path.dirname(os.path.realpath(__file__))
                    plotPath = os.path.join(filePath, 'plots')
                    os.makedirs(plotPath, exist_ok=True)
                    plt.savefig(
                        os.path.join(plotPath, f"{saveName}_{targ.name}_{gs.name}.png"),
                        dpi=300,
                    )

                    plt.close()

    def plot_estimator_results(
        self, time_vec: u.Quantity[u.minute], saveName: str
    ) -> None:
        """
        Makes one plot for each satellite target pairing that shows a comparison between the different estimation algorithms

        Will show local, central, ci, and et estimators for each target, depending on what is available

        Make a subplot showing the state errors, the innovation errors, and the track uncertainty for each estimator
        """

        plt.close('all')
        state_labels = [
            'X [km]',
            'Vx [km/min]',
            'Y [km]',
            'Vy [km/min]',
            'Z [km]',
            'Vz [km/min]',
        ]
        meas_labels = ['In Track [deg]', 'Cross Track [deg]', 'Track Uncertainity [km]']

        # For Each Target
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Set up colors
                    satColor = sat.color
                    centralColor = '#070400'
                    ciColor = '#DC143C'
                    etColor = 'slategray'

                    targetID = targ.targetID
                    trueHist = targ.hist
                    if self.estimator_config.local:
                        localEKF = sat.indeptEstimator
                    if self.estimator_config.central:
                        centralEKF = self.centralEstimator
                    if self.estimator_config.ci:
                        ciEKF = sat.ciEstimator
                    if self.estimator_config.et:
                        etEKF = sat.etEstimators[0]

                    fig = plt.figure(figsize=(15, 8))
                    fig.suptitle(f"{targ.name}, {sat.name}", fontsize=14)
                    axes = self.setup_axes(fig, state_labels, meas_labels)
                    handles = []

                    # Check, do we have local estimates?
                    if self.estimator_config.local:
                        (
                            times,
                            estHist,
                            covHist,
                            innovationHist,
                            innovationCovHist,
                            trackErrorHist,
                        ) = self.getEstimationHistory(
                            targetID, time_vec, filter=localEKF
                        )

                        times = [time for time in time_vec.value if time in estHist]
                        innovation_times = [
                            time for time in time_vec.value if time in innovationHist
                        ]
                        trackError_times = [
                            time for time in time_vec.value if time in trackErrorHist
                        ]

                        self.plot_errors(
                            axes,
                            times,
                            estHist,
                            trueHist,
                            covHist,
                            label_color=satColor,
                            linewidth=2.5,
                        )
                        self.plot_innovations(
                            axes,
                            innovation_times,
                            innovationHist,
                            innovationCovHist,
                            label_color=satColor,
                            linewidth=2.5,
                        )
                        self.plot_track_uncertainty(
                            axes,
                            trackError_times,
                            trackErrorHist,
                            targ.tqReq,
                            label_color=satColor,
                            linewidth=2.5,
                        )
                        handles.append(
                            patches.Patch(
                                color=satColor, label=f'{sat.name} Indept. Estimator'
                            )
                        )

                    # Check, do we have central estimates?
                    if self.estimator_config.central:
                        (
                            central_times,
                            central_estHist,
                            central_covHist,
                            central_innovationHist,
                            central_innovationCovHist,
                            central_trackErrorHist,
                        ) = self.getEstimationHistory(
                            targetID, time_vec, filter=centralEKF
                        )

                        central_times = [
                            time for time in time_vec.value if time in central_estHist
                        ]
                        central_trackError_times = [
                            time
                            for time in time_vec.value
                            if time in central_trackErrorHist
                        ]

                        self.plot_errors(
                            axes,
                            central_times,
                            central_estHist,
                            trueHist,
                            central_covHist,
                            label_color=centralColor,
                            linewidth=1.5,
                        )
                        self.plot_track_uncertainty(
                            axes,
                            central_trackError_times,
                            central_trackErrorHist,
                            targ.tqReq,
                            label_color=centralColor,
                            linewidth=1.5,
                        )
                        handles.append(
                            patches.Patch(
                                color=centralColor, label=f'Central Estimator'
                            )
                        )

                    # Check, do we have CI estimates?
                    if self.estimator_config.ci:
                        (
                            ci_times,
                            ci_estHist,
                            ci_covHist,
                            ci_innovationHist,
                            ci_innovationCovHist,
                            ci_trackErrorHist,
                        ) = self.getEstimationHistory(targetID, time_vec, filter=ciEKF)

                        ci_times = [
                            time for time in time_vec.value if time in ci_estHist
                        ]
                        ci_innovation_times = [
                            time for time in time_vec.value if time in ci_innovationHist
                        ]
                        ci_trackError_times = [
                            time for time in time_vec.value if time in ci_trackErrorHist
                        ]

                        self.plot_errors(
                            axes,
                            ci_times,
                            ci_estHist,
                            trueHist,
                            ci_covHist,
                            label_color=ciColor,
                            linewidth=2.5,
                        )
                        self.plot_innovations(
                            axes,
                            ci_innovation_times,
                            ci_innovationHist,
                            ci_innovationCovHist,
                            label_color=ciColor,
                            linewidth=2.5,
                        )
                        self.plot_track_uncertainty(
                            axes,
                            ci_trackError_times,
                            ci_trackErrorHist,
                            targ.tqReq,
                            label_color=ciColor,
                            linewidth=2.5,
                        )
                        handles.append(
                            patches.Patch(color=ciColor, label=f'CI Estimator')
                        )

                    # Check, do we have ET estimates?
                    if self.estimator_config.et:
                        (
                            et_times,
                            et_estHist,
                            et_covHist,
                            et_innovationHist,
                            et_innovationCovHist,
                            et_trackErrorHist,
                        ) = self.getEstimationHistory(targetID, time_vec, filter=etEKF)

                        et_times = [
                            time for time in time_vec.value if time in et_estHist
                        ]
                        # et_innovation_times = [time for time in time_vec.value if time in et_innovationHist]
                        et_trackError_times = [
                            time for time in time_vec.value if time in et_trackErrorHist
                        ]

                        self.plot_errors(
                            axes,
                            et_times,
                            et_estHist,
                            trueHist,
                            et_covHist,
                            label_color=etColor,
                            linewidth=2.5,
                        )
                        # self.plot_innovations(axes, et_innovation_times, et_innovationHist, et_innovationCovHist, label_color=ciColor, linewidth=2.5)
                        self.plot_track_uncertainty(
                            axes,
                            et_trackError_times,
                            et_trackErrorHist,
                            targ.tqReq,
                            label_color=etColor,
                            linewidth=2.5,
                        )
                        handles.append(
                            patches.Patch(color=etColor, label=f'ET Estimator')
                        )

                    # Add the legend
                    fig.legend(handles=handles, loc='lower right')
                    plt.tight_layout()

                    # Save the Plot with respective suffix
                    self.save_plot(fig, saveName, targ, sat)

                    # Close the figure
                    plt.close(fig)

    def getEstimationHistory(self, targetID, time_vec, filter=None):
        """
        Get the estimation history for a given target and estimator.
        """
        times, estHist, covHist, innovationHist, innovationCovHist, trackErrorHist = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        times = [time for time in time_vec.value if time in filter.estHist[targetID]]
        estHist = filter.estHist[targetID]
        covHist = filter.covarianceHist[targetID]
        innovationHist = filter.innovationHist[targetID]
        innovationCovHist = filter.innovationCovHist[targetID]
        trackErrorHist = filter.trackErrorHist[targetID]

        return (
            times,
            estHist,
            covHist,
            innovationHist,
            innovationCovHist,
            trackErrorHist,
        )

    def plot_errors(
        self, ax, times, estHist, trueHist, covHist, label_color, linewidth
    ):
        """
        Plot error vector and two sigma bounds for covariance.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot.
            times (list): List of times for the estimates.
            estHist (dict): Dictionary of estimated histories.
            trueHist (dict): Dictionary of true histories.
            covHist (dict): Dictionary of covariance histories.
            label_color (str): Color for the plot.
            linewidth (float): Width of the plot lines.
        """
        i = 0
        for axis in ax:
            if times:  # If there is an estimate on target
                segments = self.segment_data(times, max_gap=self.delta_t * 2)
                for segment in segments:

                    if i == 6:
                        return
                    est_errors = [
                        estHist[time][i] - trueHist[time][i] for time in segment
                    ]
                    upper_bound = [2 * np.sqrt(covHist[time][i][i]) for time in segment]
                    lower_bound = [
                        -2 * np.sqrt(covHist[time][i][i]) for time in segment
                    ]

                    axis.plot(
                        segment, est_errors, color=label_color, linewidth=linewidth
                    )
                    axis.plot(
                        segment,
                        upper_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )
                    axis.plot(
                        segment,
                        lower_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )

                    i += 1

    def plot_innovations(
        self, ax, times, innovationHist, innovationCovHist, label_color, linewidth
    ):
        """
        Plot the innovation in bearings angles.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot.
            times (list): List of times for the innovations.
            innovationHist (dict): Dictionary of innovation histories.
            innovationCovHist (dict): Dictionary of innovation covariance histories.
            label_color (str): Color for the plot.
            linewidth (float): Width of the plot lines.
        """
        if times:  # If there is an estimate on target
            segments = self.segment_data(times, max_gap=self.delta_t * 2)

            for i in range(2):  # For each measurement [in track, cross track]
                for segment in segments:

                    innovation = [innovationHist[time][i] for time in segment]
                    upper_bound = [
                        2 * np.sqrt(innovationCovHist[time][i][i]) for time in segment
                    ]
                    lower_bound = [
                        -2 * np.sqrt(innovationCovHist[time][i][i]) for time in segment
                    ]

                    # Check, if the innovation is 0, prune it
                    for idx, value in enumerate(innovation):
                        if value == 0:
                            # Pop the value from the innovation
                            segment.pop(idx)
                            innovation.pop(idx)
                            upper_bound.pop(idx)
                            lower_bound.pop(idx)

                    ax[6 + i].plot(
                        segment, innovation, color=label_color, linewidth=linewidth
                    )
                    ax[6 + i].plot(
                        segment,
                        upper_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )
                    ax[6 + i].plot(
                        segment,
                        lower_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )

    def plot_et_messages(
        self,
        ax,
        sat: satellite.Satellite,
        sat2: satellite.Satellite,
        targetID: int,
        timeVec,
    ):
        # TODO: EITHER USE THIS OR DONT USE THIS!, SHOWS THE ET MESSAGING

        # Find common EKF
        sat12_commonEKF = None
        for et_estimator in sat.etEstimators:
            if et_estimator.shareWith == sat2.name:
                sat12_commonEKF = et_estimator
                break

        for time in timeVec:
            if sat12_commonEKF is not None:
                if sat12_commonEKF.synchronizeFlag[targetID][time] == True:
                    ax.scatter(time, 0.5, color='g', marker='D', s=70)
                    continue

            related_comms = self.comms.used_comm_et_data.loc[
                (self.comms.used_comm_et_data['sender'] == sat.name)
                & (self.comms.used_comm_et_data['receiver'] == sat2.name)
                & (self.comms.used_comm_et_data['target_id'] == targetID)
                & (self.comms.used_comm_et_data['time'] == time)
            ]
            related_transmissions = self.comms.used_comm_et_data.to_dataclasses(
                related_comms
            )
            if (
                len(related_transmissions) > 0
                and related_transmissions[0].has_alpha_beta is not None
            ):
                alpha = related_transmissions[0].alpha
                beta = related_transmissions[0].beta
                if not np.isnan(alpha):
                    ax.scatter(time, 0.9, color='r', marker=r'$\alpha$', s=80)
                else:
                    ax.scatter(time, 0.2, color='b', marker=r'$\alpha$', s=80)

                if not np.isnan(beta):
                    ax.scatter(time, 0.8, color='r', marker=r'$\beta$', s=120)
                else:
                    ax.scatter(time, 0.1, color='b', marker=r'$\beta$', s=120)

        ax.set_yticks([0, 0.5, 1])
        # set the axis limits to be the whole time vector
        ax.set_xlim([timeVec[0], timeVec[-1]])
        ax.set_yticklabels(['Implicit', 'CI', 'Explict'])
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Message Type')
        ax.set_title(f'{sat2.name} -> {sat.name} Messages')

    def plot_track_uncertainty(
        self, ax, times, trackErrorHist, targQuality, label_color, linewidth, ci=False
    ):
        """
        Plot the track quality.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot.
            times (list): List of times for the track quality.
            trackErrorHist (dict): Dictionary of track quality histories.
            label_color (str): Color for the plot.
            linewidth (float): Width of the plot lines.
        """
        if times:
            nonEmptyTime = []
            segments = self.segment_data(times, max_gap=self.delta_t * 2)

            for segment in segments:
                # Figure out, does this segment have a real data point in every time step
                new_time = []
                for time in segment:
                    if not not trackErrorHist[time]:
                        new_time.append(time)
                        nonEmptyTime.append(time)

                track_quality = [trackErrorHist[time] for time in new_time]
                ax[8].plot(
                    new_time, track_quality, color=label_color, linewidth=linewidth
                )

            if ci:
                # Finally plot a dashed line for the targetPriority
                ax[8].axhline(
                    y=targQuality, color='k', linestyle='dashed', linewidth=1.5
                )
                # Add a text label on the above right side of the dashed line
                ax[8].text(
                    min(nonEmptyTime),
                    targQuality + 5,
                    f"Target Quality: {targQuality}",
                    fontsize=8,
                    color='k',
                )

    def segment_data(self, times, max_gap=1):
        """
        Splits a list of times into segments where the time difference between consecutive points
        is less than or equal to a specified maximum gap.

        Args:
            times (list): List of times to be segmented.
            max_gap (float): Maximum allowed gap between consecutive times to be considered
                            in the same segment. Defaults to 30.

        Returns:
            list: A list of lists, where each sublist contains a segment of times.
        """

        # Initialize the list to store segments
        segments = []

        if not times:
            return segments

        # Start the first segment with the first time point
        current_segment = [times[0]]

        # Iterate over the remaining time points
        for i in range(1, len(times)):
            # Check if the difference between the current time and the previous time is within the max_gap
            if times[i] - times[i - 1] <= max_gap:
                # If within the max_gap, add the current time to the current segment
                current_segment.append(times[i])
            else:
                # If not within the max_gap, finalize the current segment and start a new segment
                segments.append(current_segment)
                current_segment = [times[i]]

        # Add the last segment to the list of segments
        segments.append(current_segment)

        return segments

    def shifted_colors(self, hex_colors1, hex_colors2, shift=50):
        # TODO: DELETE?
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB tuple."""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            """Convert RGB tuple to hex color."""
            return '#{:02x}{:02x}{:02x}'.format(*rgb)

        def find_middle_colors(color1, color2, shift=10):
            """Find the middle color between two hex colors."""
            # Convert hex colors to RGB
            rgb1 = hex_to_rgb(color1)
            rgb2 = hex_to_rgb(color2)

            # Calculate the middle RGB values
            middle_rgb = tuple((c1 + c2) // 2 for c1, c2 in zip(rgb1, rgb2))

            # Shift the middle color to get two new colors
            color1 = [max(comp - shift, 0) for comp in middle_rgb]
            color2 = [min(comp + shift, 255) for comp in middle_rgb]

            # Convert the middle RGB value back to hex
            return rgb_to_hex(color1), rgb_to_hex(color2)

        sat1commonColor, sat2commonColor = find_middle_colors(
            hex_colors1, hex_colors2, shift
        )
        return sat1commonColor, sat2commonColor

    def setup_axes(self, fig, state_labels, meas_labels):
        """
        Set up the axes.

        Args:
            fig (matplotlib.figure.Figure): The figure to which the axes are added.
            state_labels (list): List of labels for the state plots.
            meas_labels (list): List of labels for the measurement plots.

        Returns:
            list: List of axes.
        """
        gs = gridspec.GridSpec(3, 6)  # 3x3 Grid
        axes = []
        for i in range(3):  # Stack the Position and Velocity Components
            ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
            ax.grid(True)
            axes.append(ax)
            ax = fig.add_subplot(gs[1, 2 * i : 2 * i + 2])
            ax.grid(True)
            axes.append(ax)

        for i in range(2):  # Create the innovation plots
            ax = fig.add_subplot(gs[2, 2 * i : 2 * i + 2])
            ax.grid(True)
            axes.append(ax)

        # Create the track quality plot
        ax = fig.add_subplot(gs[2, 4:6])
        ax.grid(True)
        axes.append(ax)

        # Label plots with respective labels
        for i in range(6):  # TODO: Should we fix all x-axis to be time vector
            axes[i].set_xlabel("Time [min]")
            axes[i].set_ylabel(f"Error in {state_labels[i]}")
        for i in range(2):
            axes[6 + i].set_xlabel("Time [min]")
            axes[6 + i].set_ylabel(f"Innovation in {meas_labels[i]}")
        axes[8].set_xlabel("Time [min]")
        axes[8].set_ylabel("Track Uncertainity [km]")

        return axes

    def save_plot(self, fig, saveName, targ, sat, suffix=None):
        """
        Save each plot into the "plots" folder with the given suffix.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            plotEstimators (bool): Flag indicating whether to save the plot.
            saveName (str): Name for the saved plot file.
            targ (Target): Target object.
            sat (Satellite): Satellite object.
            suffix (str): Suffix for the saved plot file name.
        """
        filePath = os.path.dirname(os.path.realpath(__file__))
        plotPath = os.path.join(filePath, 'plots')
        os.makedirs(plotPath, exist_ok=True)
        if saveName is None:
            if suffix is None:
                plt.savefig(
                    os.path.join(plotPath, f"{targ.name}_{sat.name}.png"), dpi=300
                )
            else:
                plt.savefig(
                    os.path.join(plotPath, f"{targ.name}_{sat.name}_{suffix}.png"),
                    dpi=300,
                )
        else:
            if suffix is None:
                plt.savefig(
                    os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}.png"),
                    dpi=300,
                )
            else:
                plt.savefig(
                    os.path.join(
                        plotPath, f"{saveName}_{targ.name}_{sat.name}_{suffix}.png"
                    ),
                    dpi=300,
                )
        plt.close()

    ### Plot communications sent/recieved
    # Plot the total data sent and received by satellites
    def plot_global_comms(self, saveName: str | None):
        """Plots the total data sent and received by satellites in DDF algorithms"""
        # Get the names of satellites:
        sat_names = [sat.name for sat in self.sats]

        ## Plot comm data sent for CI Algo
        if self.estimator_config.ci:
            comms_plot.create_comms_plot(
                self.comms.total_comm_data,
                self.targs,
                sat_names,
                super_title='TOTAL Data Sent and Received by Satellites',
                y_label='Total Data Sent/Recieved (# of numbers)',
                filename='total_ci_comms',
                prefix=saveName,
                save=True,
            )

        ## Plot comm data sent for ET Algo
        if self.estimator_config.et:
            comms_plot.create_comms_plot(
                self.comms.total_comm_et_data,
                self.targs,
                sat_names,
                super_title='TOTAL ET Data Sent and Received by Satellites',
                y_label='Total ET Data Sent/Recieved (# of numbers)',
                filename='total_et_comms',
                prefix=saveName,
                save=True,
            )

    # Plots the actual data amount used by the satellites
    # TODO: Remove? It honestly doesn't matter
    def plot_used_comms(self, saveName: str | None):
        """
        Plots the used data sent and received by satellites in DDF algorithms.

        'Used' means information used for a satellite to meet TQ requirements.
        """
        sat_names = [sat.name for sat in self.sats]

        ## Plot comm data sent for CI Algo
        if self.estimator_config.ci:
            comms_plot.create_comms_plot(
                self.comms.used_comm_data,
                self.targs,
                sat_names,
                super_title='USED Data Sent and Received by Satellites',
                y_label='Used Data Sent/Recieved (# of numbers)',
                filename='used_ci_comms',
                prefix=saveName,
                save=True,
            )

        ## Plot comm data sent for ET Algo
        if self.estimator_config.et:
            comms_plot.create_comms_plot(
                self.comms.used_comm_et_data,
                self.targs,
                sat_names,
                super_title='USED ET Data Sent and Received by Satellites',
                y_label='Used ET Data Sent/Recieved (# of numbers)',
                filename='used_et_comms',
                prefix=saveName,
                save=True,
            )

    # Sub plots for each satellite showing the track uncertainty for each target and then the comms sent/recieved about each target vs time
    def plot_timeHist_comms_ci(self, saveName: str | None):
        """Plots a time history of the CI communications received for each satellite on every target."""

        # For each satellite make a plot:
        for sat in self.sats:

            # Create the figure and subplot:
            fig = plt.figure(figsize=(15, 8))

            fig.suptitle(
                f"CI DDF, Track Uncertainty and Data Used by {sat.name}", fontsize=14
            )
            gs = gridspec.GridSpec(2, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            # Add a subplot title
            ax1.set_title(f"Track Uncertainty for {sat.name}")
            ax1.set_ylabel('Track Uncertainty [km]')

            ax2 = fig.add_subplot(gs[1, 0])
            # Add a subplot title
            ax2.set_title(f"Data Used by {sat.name}")
            ax2.set_ylabel('Data Sent/Recieved (# of numbers)')
            ax2.set_xlabel('Time [min]')

            # Now, at the bottom of the plot, add the legends
            handles = []
            for targ in self.targs:
                if (
                    targ.targetID in sat.targetIDs
                ):  # Change such that it is only the targets that the satellite is tracking
                    handles.append(
                        patches.Patch(color=targ.color, label=f"{targ.name}")
                    )
            for tempSat in self.sats:
                handles.append(
                    patches.Patch(color=tempSat.color, label=f"{tempSat.name}")
                )

            # Create a legend
            fig.legend(
                handles=handles, loc='lower right', ncol=2, bbox_to_anchor=(1, 0)
            )

            nonEmptyTime = []

            # Now do plots for the first subplot, we will be plotting track uncertainty for each target
            for targ in self.targs:
                if targ.targetID not in sat.targetIDs:
                    continue

                # Get the uncertainty data
                trackUncertainty = sat.ciEstimator.trackErrorHist[targ.targetID]

                # Get the times for the track_uncertainty
                times = [time for time in trackUncertainty.keys()]
                segments = self.segment_data(times, max_gap=self.delta_t * 2)

                for segment in segments:
                    # Does the semgnet have a real data point in eveyr time step?
                    newTime = []
                    for time in segment:
                        if not not trackUncertainty[time]:
                            newTime.append(time)
                            nonEmptyTime.append(time)

                    trackVec = [trackUncertainty[time] for time in newTime]
                    ax1.plot(newTime, trackVec, color=targ.color, linewidth=1.5)

            # Now, do the dashed lines for the target quality, including varying with commanders intent

            count = 0
            for time in self.commandersIntent.keys():

                count += 1
                xMin = time

                # Does there exist another key in commandersIntent?
                if len(self.commandersIntent.keys()) > count:
                    xMax = list(self.commandersIntent.keys())[count]
                else:
                    xMax = max(nonEmptyTime)

                # Now, make a dashed line from xMin to xMax for the target qualities in commandersIntent

                for targ in self.targs:

                    # Should the satellite even be tracking this target?
                    if targ.targetID not in sat.targetIDs:
                        continue

                    # Now plot a dashed line for the targetPriority
                    ax1.hlines(
                        y=self.commandersIntent[time][sat.name][targ.targetID],
                        xmin=xMin,
                        xmax=xMax,
                        color=targ.color,
                        linestyle='dashed',
                        linewidth=1.5,
                    )

                    # Now plot a dashed line for the targetPriority
                    # ax1.axhline(y=self.commandersIntent[time][sat.name][targ.targetID], color=targ.color, linestyle='dashed', linewidth=1.5)
                    # Add a text label on the above right side of the dashed line
                    # ax1.text(xMin, self.commandersIntent[time][sat.name][targ.targetID] + 5, f"Target Quality: {targ.tqReq}", fontsize=8, color=targ.color)

            # # Now for each target make the dashed lines for the target quality
            # for targ in self.targs:
            #     if targ.targetID not in sat.targetIDs:
            #         continue

            #     # Now plot a dashed line for the targetPriority
            #     ax1.axhline(y=sat.targPriority[targ.targetID], color=targ.color, linestyle='dashed', linewidth=1.5)
            #     # Add a text label on the above right side of the dashed line
            #     # ax1.text(min(nonEmptyTime), sat.targPriority[targ.targetID] + 5, f"Target Quality: {targ.tqReq}", fontsize=8, color=targ.color)

            # Now do the 2nd subplot, bar plot showing the data sent/recieved by each satellite about each target

            # Save previous data, to stack the bars
            prevData = defaultdict(dict)

            nonEmptyTime = list(set(nonEmptyTime))  # also make it assending
            nonEmptyTime.sort()

            # also now use a specified order of sats
            sat_names = [sat.name for sat in self.sats]

            # Use the nonEmptyTime to get the minimum difference between time steps
            differences = [j - i for i, j in zip(nonEmptyTime[:-1], nonEmptyTime[1:])]
            min_diff = min(differences)

            # Make nonEmptyTimes the keys of prevData
            for time in nonEmptyTime:
                prevData[time] = 0

            for targ in self.targs:
                if targ.targetID not in sat.targetIDs:
                    continue

                # Check if the target has any communication data
                related_comms = self.comms.used_comm_data.loc[
                    self.comms.used_comm_data['target_id'] == targ.targetID
                ]
                if not len(related_comms):
                    continue

                for sender in sat_names:
                    if sender == sat:
                        continue

                    # So now, we have a satellite, the reciever, recieving information about targetID from sender
                    # We want to count, how much information did the reciever recieve from the sender in a time history and plot that on a bar chart
                    related_comms = self.comms.used_comm_data.loc[
                        (self.comms.used_comm_data['sender'] == sender)
                        & (self.comms.used_comm_data['receiver'] == sat.name)
                        & (self.comms.used_comm_data['target_id'] == targ.targetID)
                    ]
                    related_transmissions = self.comms.used_comm_data.to_dataclasses(
                        related_comms
                    )
                    data = {
                        comm.time: comm.size
                        for comm in related_transmissions
                        if comm.time in nonEmptyTime
                    }

                    # Check, does any of the data contain [] or None? If so, make it a 0
                    for key in data:
                        if not data[key]:
                            data[key] = 0

                    # Now make sure any data that exists in prevData, exists in data
                    for key in prevData.keys():
                        if key not in data:
                            data[key] = 0

                    # Get the values from the data:
                    values = [data[time] for time in nonEmptyTime]

                    # Get the color of the sender
                    sender = [s for s in self.sats if s.name == sender][0]

                    # Now do a bar plot with the data
                    ax2.bar(
                        nonEmptyTime,
                        values,
                        bottom=list(prevData.values()),
                        color=targ.color,
                        hatch='//',
                        edgecolor=sender.color,
                        linewidth=0,
                        width=min_diff,
                    )

                    # Add the values to the prevData
                    for key in data.keys():
                        prevData[key] += data[key]

            # Save the plot
            if saveName is not None:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                os.makedirs(plotPath, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        plotPath, f"{saveName}_{sat.name}_track_uncertainty.png"
                    ),
                    dpi=300,
                )

            # plt.close(fig)

    # TODO: Remove? online DDF graph plotting?
    def plot_dynamic_comms(self):
        comms = self.comms
        envTime = self.time.to_value()
        once = True

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        diComms = nx.MultiDiGraph()

        edge_styles = []
        node_colors = []

        for targ in self.targs:
            for sat in self.sats:
                if sat not in diComms.nodes():
                    diComms.add_node(sat)
                    node_colors.append(sat.color)

                for sat2 in self.sats:
                    if sat == sat2:
                        continue

                    # Add the second satellite node
                    if sat2 not in diComms.nodes():
                        diComms.add_node(sat2)
                        node_colors.append(sat2.color)

                    targetID = targ.targetID
                    targ_color = targ.color

                    # find the common EKFs
                    commonEKF = None
                    for each_etEstimator in sat.etEstimators:
                        if each_etEstimator.shareWith == sat2.name:
                            commonEKF = each_etEstimator
                            break

                    if commonEKF is not None:
                        # If the satellites synchronize, add an edge
                        if commonEKF.synchronizeFlag[targetID][envTime] == True:
                            diComms.add_edge(sat2, sat)
                            style = self.get_edge_style(
                                comms, targetID, sat, sat2, envTime, CI=True
                            )
                            edge_styles.append((sat2, sat, style, targ_color))

                        # If there is a communication between the two satellites, add an edge
                        related_comms = self.comms.total_comm_et_data.loc[
                            (self.comms.total_comm_et_data['sender'] == sat.name)
                            & (self.comms.total_comm_et_data['receiver'] == sat2.name)
                            & (
                                self.comms.total_comm_et_data['target_id']
                                == targ.targetID
                            )
                            & (self.comms.total_comm_et_data['time'] == envTime)
                        ]
                        related_transmissions = (
                            self.comms.total_comm_et_data.to_dataclasses(related_comms)
                        )
                        if (
                            len(related_transmissions)
                            and related_transmissions[0].has_alpha_beta
                        ):
                            diComms.add_edge(sat2, sat)
                            style = self.get_edge_style(
                                comms, targetID, sat, sat2, envTime
                            )
                            edge_styles.append((sat2, sat, style, targ_color))

            # Draw the graph with the nodes and edges

        if once:
            pos = nx.circular_layout(diComms)
            nx.draw_networkx_nodes(
                diComms, pos, ax=ax, node_size=1000, node_color=node_colors
            )
            once = False
        # Draw edges with appropriate styles
        for i, edge in enumerate(edge_styles):
            # Adjust the curvature for each edge
            connectionstyle = (
                f'arc3,rad={(i - len(edge_styles) / 2) / len(edge_styles)}'
            )
            nx.draw_networkx_edges(
                diComms,
                pos,
                edgelist=[(edge[0], edge[1])],
                ax=ax,
                style=edge[2],
                edge_color=edge[3],
                arrows=True,
                arrowsize=10,
                connectionstyle=connectionstyle,
                min_source_margin=0,
                min_target_margin=40,
                width=2,
            )

        # Add labels
        labels = {node: node.name for node in diComms.nodes()}
        nx.draw_networkx_labels(diComms, pos, ax=ax, labels=labels)
        # Add Title
        ax.set_title(f"Dynamic Communications at Time {envTime} min")
        handles = [
            patches.Patch(color=targ.color, label=targ.name) for targ in self.targs
        ]
        fig.legend(
            handles=handles, loc='lower right', ncol=1, bbox_to_anchor=(0.9, 0.1)
        )

        # Display and close the figure
        img = self.save_comm_plot_to_image(fig)
        self.imgs_dyn_comms.append(img)

        ax.cla()
        # clear graph
        diComms.clear()
        plt.close(fig)

    # TODO: IF REMOVE plot_dynamic_comms, REMOVE THIS
    def get_edge_style(
        self,
        comms: comms.Comms,
        targetID: int,
        sat1: satellite.Satellite,
        sat2: satellite.Satellite,
        envTime: float,
        CI: bool = False,
    ):
        """
        Helper function to determine the edge style based on communication data.
        Returns 'solid' if both alpha and beta are present, 'dashed' if only one is present,
        and None if neither is present (meaning no line).
        """

        if CI:
            return (0, ())

        related_comms = self.comms.total_comm_et_data.loc[
            (self.comms.total_comm_et_data['sender'] == sat1.name)
            & (self.comms.total_comm_et_data['receiver'] == sat2.name)
            & (self.comms.total_comm_et_data['target_id'] == targetID)
            & (self.comms.total_comm_et_data['time'] == envTime)
        ]
        assert len(related_comms)
        related_transmissions = self.comms.total_comm_et_data.to_dataclasses(
            related_comms
        )
        alpha = related_transmissions[0].alpha
        beta = related_transmissions[0].beta
        if np.isnan(alpha) and np.isnan(beta):
            return (0, (1, 10))
        elif np.isnan(alpha) or np.isnan(beta):
            return (0, (3, 10, 1, 10))
        else:
            return (0, (5, 10))

    # TODO: IF REMOVE plot_dynamic_comms, REMOVE THIS
    def save_comm_plot_to_image(self, fig):
        """
        Saves the plot to an image.

        Parameters:
        - fig: The matplotlib figure to save.

        Returns:
        numpy.ndarray: The image array.
        """
        ios = io.BytesIO()
        fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        return img

    # TODO: helpful for visualizing/debugging
    ### Plot 3D Gaussian Uncertainity Ellispoids ###
    def plot_all_uncertainty_ellipses(self, time_vec):
        """
        Plots Local Uncertainty Ellipsoids, DDF Uncertainty Ellipsoids, Central Uncertainty Ellipsoids,
        and Overlapping Uncertainty Ellipsoids for each target and satellite.

        ## TODO: Modify such stereo ellipse plotter is dynamic if it finds stereo estimation

        Returns:
        None
        """
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    for sat2 in self.sats:
                        if targ.targetID in sat2.targetIDs:
                            if sat != sat2:
                                # Create a 2x2 subplot to save as a gif
                                # Rows; Stereo CI, Stereo ET, CI vs Central, ET vs Central
                                fig = plt.figure(figsize=(12, 12))
                                fig.suptitle(
                                    f"{targ.name}, {sat.name}, {sat2.name} Stereo Gaussian Uncertainty Ellipsoids"
                                )

                                sat1Color = sat.color
                                sat2Color = sat2.color
                                ciColor = '#DC143C'  # Crimson
                                etColor = '#DC143C'  # Crimson
                                centralColor = '#070400'
                                alpha = 0.2

                                # Create 2x2 Grid
                                ax1 = fig.add_subplot(221, projection='3d')
                                ax2 = fig.add_subplot(222, projection='3d')
                                ax3 = fig.add_subplot(223, projection='3d')
                                ax4 = fig.add_subplot(224, projection='3d')
                                plt.subplots_adjust(
                                    left=0.05,
                                    right=0.95,
                                    top=0.9,
                                    bottom=0.05,
                                    wspace=0.15,
                                    hspace=0.15,
                                )

                                for bigTime in time_vec.value:
                                    time = bigTime
                                    sat1_times = sat.indeptEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    sat2_times = sat2.indeptEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    stereo_times = [
                                        time
                                        for time in sat1_times
                                        if time in sat2_times
                                    ]

                                    ci_times = sat.ciEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    times = [
                                        time
                                        for time in stereo_times
                                        if time in ci_times
                                    ]

                                    if bigTime in times:
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos1 = np.array(
                                            [
                                                sat.indeptEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        est_pos2 = np.array(
                                            [
                                                sat2.indeptEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        ci_pos = np.array(
                                            [
                                                sat.ciEstimator.estHist[targ.targetID][
                                                    time
                                                ][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )

                                        cov_matrix1 = (
                                            sat.indeptEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )
                                        cov_matrix2 = (
                                            sat2.indeptEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )
                                        ci_cov = sat.ciEstimator.covarianceHist[
                                            targ.targetID
                                        ][time][[0, 2, 4]][:, [0, 2, 4]]

                                        eigenvalues1, eigenvectors1 = np.linalg.eigh(
                                            cov_matrix1
                                        )
                                        eigenvalues2, eigenvectors2 = np.linalg.eigh(
                                            cov_matrix2
                                        )
                                        ci_eigenvalues, ci_eigenvectors = (
                                            np.linalg.eigh(ci_cov)
                                        )

                                        error1 = np.linalg.norm(true_pos - est_pos1)
                                        error2 = np.linalg.norm(true_pos - est_pos2)
                                        ci_error = np.linalg.norm(true_pos - ci_pos)

                                        LOS_vec1 = -sat.orbitHist[
                                            time
                                        ] / np.linalg.norm(sat.orbitHist[time])
                                        LOS_vec2 = -sat2.orbitHist[
                                            time
                                        ] / np.linalg.norm(sat2.orbitHist[time])

                                        self.plot_ellipsoid(
                                            ax1,
                                            est_pos1,
                                            cov_matrix1,
                                            color=sat1Color,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax1,
                                            est_pos2,
                                            cov_matrix2,
                                            color=sat2Color,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax1,
                                            ci_pos,
                                            ci_cov,
                                            color=ciColor,
                                            alpha=alpha + 0.1,
                                        )

                                        self.plot_estimate(
                                            ax1, est_pos1, true_pos, sat1Color
                                        )
                                        self.plot_estimate(
                                            ax1, est_pos2, true_pos, sat2Color
                                        )
                                        self.plot_estimate(
                                            ax1, ci_pos, true_pos, ciColor
                                        )

                                        self.plot_LOS(ax1, est_pos1, LOS_vec1)
                                        self.plot_LOS(ax1, est_pos2, LOS_vec2)

                                        self.set_axis_limits(
                                            ax1,
                                            ci_pos,
                                            np.sqrt(ci_eigenvalues),
                                            margin=50.0,
                                        )
                                        self.plot_labels(ax1, time)
                                        self.make_legend1(
                                            ax1,
                                            sat,
                                            sat1Color,
                                            sat2,
                                            sat2Color,
                                            ciColor,
                                            error1,
                                            error2,
                                            ci_error,
                                            'CI',
                                        )

                                    et_times = sat.etEstimator.estHist[targ.targetID][
                                        sat
                                    ][sat].keys()
                                    times = [
                                        time
                                        for time in stereo_times
                                        if time in et_times
                                    ]

                                    if bigTime in times:
                                        # Plot Et
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos1 = np.array(
                                            [
                                                sat.indeptEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        est_pos2 = np.array(
                                            [
                                                sat2.indeptEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        et_pos = np.array(
                                            [
                                                sat.etEstimator.estHist[targ.targetID][
                                                    sat
                                                ][sat][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )

                                        cov_matrix1 = (
                                            sat.indeptEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )
                                        cov_matrix2 = (
                                            sat2.indeptEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )
                                        et_cov = sat.etEstimator.covarianceHist[
                                            targ.targetID
                                        ][sat][sat][time][np.array([0, 2, 4])][
                                            :, np.array([0, 2, 4])
                                        ]

                                        eigenvalues1, eigenvectors1 = np.linalg.eigh(
                                            cov_matrix1
                                        )
                                        eigenvalues2, eigenvectors2 = np.linalg.eigh(
                                            cov_matrix2
                                        )
                                        et_eigenvalues, et_eigenvectors = (
                                            np.linalg.eigh(et_cov)
                                        )

                                        error1 = np.linalg.norm(true_pos - est_pos1)
                                        error2 = np.linalg.norm(true_pos - est_pos2)
                                        et_error = np.linalg.norm(true_pos - et_pos)

                                        LOS_vec1 = -sat.orbitHist[
                                            time
                                        ] / np.linalg.norm(sat.orbitHist[time])
                                        LOS_vec2 = -sat2.orbitHist[
                                            time
                                        ] / np.linalg.norm(sat2.orbitHist[time])

                                        self.plot_ellipsoid(
                                            ax2,
                                            est_pos1,
                                            cov_matrix1,
                                            color=sat1Color,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax2,
                                            est_pos2,
                                            cov_matrix2,
                                            color=sat2Color,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax2,
                                            et_pos,
                                            et_cov,
                                            color=etColor,
                                            alpha=alpha,
                                        )

                                        self.plot_estimate(
                                            ax2, est_pos1, true_pos, sat1Color
                                        )
                                        self.plot_estimate(
                                            ax2, est_pos2, true_pos, sat2Color
                                        )
                                        self.plot_estimate(
                                            ax2, et_pos, true_pos, etColor
                                        )

                                        self.plot_LOS(ax2, est_pos1, LOS_vec1)
                                        self.plot_LOS(ax2, est_pos2, LOS_vec2)

                                        self.make_legend1(
                                            ax2,
                                            sat,
                                            sat1Color,
                                            sat2,
                                            sat2Color,
                                            etColor,
                                            error1,
                                            error2,
                                            et_error,
                                            'ET',
                                        )
                                        self.set_axis_limits(
                                            ax2,
                                            et_pos,
                                            np.sqrt(et_eigenvalues),
                                            margin=50.0,
                                        )
                                        self.plot_labels(ax2, time)

                                    ci_times = sat.ciEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    central_times = self.centralEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    times = [
                                        time
                                        for time in ci_times
                                        if time in central_times
                                    ]

                                    if bigTime in times:
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos = np.array(
                                            [
                                                sat.ciEstimator.estHist[targ.targetID][
                                                    time
                                                ][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        central_pos = np.array(
                                            [
                                                self.centralEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )

                                        cov_matrix = sat.ciEstimator.covarianceHist[
                                            targ.targetID
                                        ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        central_cov = (
                                            self.centralEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )

                                        eigenvalues, eigenvectors = np.linalg.eigh(
                                            cov_matrix
                                        )
                                        central_eigenvalues, central_eigenvectors = (
                                            np.linalg.eigh(central_cov)
                                        )

                                        error = np.linalg.norm(true_pos - est_pos)
                                        central_error = np.linalg.norm(
                                            true_pos - central_pos
                                        )

                                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(
                                            sat.orbitHist[time]
                                        )

                                        self.plot_ellipsoid(
                                            ax3,
                                            est_pos,
                                            cov_matrix,
                                            color=ciColor,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax3,
                                            central_pos,
                                            central_cov,
                                            color=centralColor,
                                            alpha=alpha,
                                        )

                                        self.plot_estimate(
                                            ax3, est_pos, true_pos, sat1Color
                                        )
                                        self.plot_estimate(
                                            ax3, central_pos, true_pos, centralColor
                                        )

                                        self.plot_LOS(ax3, est_pos, LOS_vec)
                                        self.set_axis_limits(
                                            ax3,
                                            est_pos,
                                            np.sqrt(eigenvalues),
                                            margin=50.0,
                                        )
                                        self.plot_labels(ax3, time)
                                        self.make_legend2(
                                            ax3,
                                            ciColor,
                                            centralColor,
                                            error,
                                            central_error,
                                            'CI',
                                        )

                                    et_times = sat.etEstimator.estHist[targ.targetID][
                                        sat
                                    ][sat].keys()
                                    central_times = self.centralEstimator.estHist[
                                        targ.targetID
                                    ].keys()
                                    times = [
                                        time
                                        for time in et_times
                                        if time in central_times
                                    ]

                                    if bigTime in times:
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos = np.array(
                                            [
                                                sat.etEstimator.estHist[targ.targetID][
                                                    sat
                                                ][sat][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )
                                        central_pos = np.array(
                                            [
                                                self.centralEstimator.estHist[
                                                    targ.targetID
                                                ][time][i]
                                                for i in [0, 2, 4]
                                            ]
                                        )

                                        cov_matrix = sat.etEstimator.covarianceHist[
                                            targ.targetID
                                        ][sat][sat][time][np.array([0, 2, 4])][
                                            :, np.array([0, 2, 4])
                                        ]
                                        central_cov = (
                                            self.centralEstimator.covarianceHist[
                                                targ.targetID
                                            ][time][[0, 2, 4]][:, [0, 2, 4]]
                                        )

                                        eigenvalues, eigenvectors = np.linalg.eigh(
                                            cov_matrix
                                        )
                                        central_eigenvalues, central_eigenvectors = (
                                            np.linalg.eigh(central_cov)
                                        )

                                        error = np.linalg.norm(true_pos - est_pos)
                                        central_error = np.linalg.norm(
                                            true_pos - central_pos
                                        )

                                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(
                                            sat.orbitHist[time]
                                        )

                                        self.plot_ellipsoid(
                                            ax4,
                                            est_pos,
                                            cov_matrix,
                                            color=etColor,
                                            alpha=alpha,
                                        )
                                        self.plot_ellipsoid(
                                            ax4,
                                            central_pos,
                                            central_cov,
                                            color=centralColor,
                                            alpha=alpha,
                                        )

                                        self.plot_estimate(
                                            ax4, est_pos, true_pos, sat1Color
                                        )
                                        self.plot_estimate(
                                            ax4, central_pos, true_pos, centralColor
                                        )

                                        self.plot_LOS(ax4, est_pos, LOS_vec)
                                        self.set_axis_limits(
                                            ax4,
                                            est_pos,
                                            np.sqrt(eigenvalues),
                                            margin=50.0,
                                        )
                                        self.plot_labels(ax4, time)
                                        self.make_legend2(
                                            ax4,
                                            etColor,
                                            centralColor,
                                            error,
                                            central_error,
                                            'ET',
                                        )

                                    handles = [
                                        patches.Patch(
                                            color=sat1Color,
                                            label=f'{sat.name} Local Estimator',
                                        ),
                                        patches.Patch(
                                            color=sat2Color,
                                            label=f'{sat2.name} Local Estimator',
                                        ),
                                        patches.Patch(
                                            color=ciColor, label=f'CI Estimator'
                                        ),
                                        patches.Patch(
                                            color=etColor, label=f'ET Estimator'
                                        ),
                                        patches.Patch(
                                            color=centralColor,
                                            label=f'Central Estimator',
                                        ),
                                    ]

                                    fig.legend(
                                        handles=handles,
                                        loc='lower right',
                                        ncol=5,
                                        bbox_to_anchor=(1, 0),
                                    )
                                    ax1.set_title(f"Covariance Intersection")
                                    ax2.set_title(f"Event Triggered Fusion")
                                    img = self.save_GEplot_to_image(fig)
                                    self.imgs_stereo_GE[targ.targetID][sat][
                                        sat2
                                    ].append(img)

                                    ax1.cla()
                                    ax2.cla()
                                    ax3.cla()
                                    ax4.cla()

                                plt.close(fig)

    # TODO: used for uncertainty ellipses
    def plot_LOS(self, ax, est_pos, LOS_vec):
        """
        Plots the line of sight vector on the given axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        - est_pos: The estimated position.
        - LOS_vec: The line of sight vector.

        Returns:
        None
        """
        # Calculate the end point of the LOS vector relative to est_pos
        arrow_length = 30  # Length of the LOS vector
        LOS_vec_unit = LOS_vec / np.linalg.norm(LOS_vec)  # Normalize the LOS vector

        # Adjusted starting point of the arrow
        arrow_start = est_pos - (LOS_vec_unit * arrow_length)

        # Use quiver to plot the arrow starting from arrow_start to est_pos
        ax.quiver(
            arrow_start[0],
            arrow_start[1],
            arrow_start[2],
            LOS_vec_unit[0],
            LOS_vec_unit[1],
            LOS_vec_unit[2],
            color='k',
            length=arrow_length,
            normalize=True,
        )
        # ax.quiver(est_pos[0], est_pos[1], est_pos[2], LOS_vec[0], LOS_vec[1], LOS_vec[2], color='k', length=10, normalize=True)

    # TODO: used for uncertainty ellipses
    def plot_labels(self, ax, time):
        """
        Plots labels on the given axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        - targ: The target object.
        - sat: The satellite object.
        - time: The current time.
        - err: The error value.

        Returns:
        None
        """
        ax.text2D(0.05, 0.95, f"Time: {time:.2f}", transform=ax.transAxes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=30)

    # TODO: used for uncertainty ellipses
    def make_legend1(
        self,
        ax,
        sat1,
        sat1color,
        sat2,
        sat2color,
        ciColor,
        error1,
        error2,
        ci_error,
        ci_type=None,
    ):
        if ci_type == 'CI':
            labels = [
                f'{sat1.name} Error: {error1:.2f} km',
                f'{sat2.name} Error: {error2:.2f} km',
                f'CI Error: {ci_error:.2f} km',
            ]
            handles = [
                patches.Patch(color=sat1color, label=labels[0]),
                patches.Patch(color=sat2color, label=labels[1]),
                patches.Patch(color=ciColor, label=labels[2]),
            ]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))

        elif ci_type == 'ET':
            labels = [
                f'{sat1.name} Error: {error1:.2f} km',
                f'{sat2.name} Error: {error2:.2f} km',
                f'ET Error: {ci_error:.2f} km',
            ]
            handles = [
                patches.Patch(color=sat1color, label=labels[0]),
                patches.Patch(color=sat2color, label=labels[1]),
                patches.Patch(color=ciColor, label=labels[2]),
            ]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))

    # TODO: used for uncertainty ellipses
    def make_legend2(self, ax, ciColor, centralColor, error1, error2, ci_type=None):
        if ci_type == 'CI':
            labels = [f'CI Error: {error1:.2f} km', f'Central Error: {error2:.2f} km']
            handles = [
                patches.Patch(color=ciColor, label=labels[0]),
                patches.Patch(color=centralColor, label=labels[1]),
            ]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))

        elif ci_type == 'ET':
            labels = [f'ET Error: {error1:.2f} km', f'Central Error: {error2:.2f} km']
            handles = [
                patches.Patch(color=ciColor, label=labels[0]),
                patches.Patch(color=centralColor, label=labels[1]),
            ]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))

    # TODO: used for uncertainty ellipses
    def set_axis_limits(self, ax, est_pos, radii, margin=50.0):
        """
        Sets the limits of the axes.

        Parameters:
        - ax: The matplotlib axes to set limits on.
        - est_pos: The estimated position.
        - radii: The radii for the limits.
        - margin: The margin to add to the limits.

        Returns:
        None
        """
        min_vals = est_pos - radii - margin
        max_vals = est_pos + radii + margin
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        ax.set_box_aspect([1, 1, 1])

    # TODO: used for uncertainty ellipses
    def save_GEplot_to_image(self, fig):
        """
        Saves the plot to an image.

        Parameters:
        - fig: The matplotlib figure to save.

        Returns:
        numpy.ndarray: The image array.
        """
        ios = io.BytesIO()
        fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        return img

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

    ### Data Dump File ###
    # TODO: look back over what data dumps we actually want
    def log_data(
        self, time_vec, saveName, filePath=os.path.dirname(os.path.realpath(__file__))
    ):
        """
        Logs data for all satellites and their corresponding targets to CSV files.

        Parameters:
        - time_vec (array-like): A vector of time points.
        - saveName (str): The base name for the saved CSV files.
        - filePath (str): The directory path where the CSV files will be saved. Defaults to the directory of the script.

        Returns:
        None
        """
        # Loop through all satellites
        for sat in self.sats:
            # Loop through all targets for each satellite
            for targ in self.targs:
                if targ.targetID in sat.targetIDs:
                    # Collect all data to be logged
                    times = time_vec.value
                    sat_hist = sat.orbitHist
                    trueHist = targ.hist

                    sat_measHistTimes = sat.measurementHist[targ.targetID].keys()
                    sat_measHist = sat.measurementHist[targ.targetID]

                    # Collect All Estimation Data
                    estTimes = sat.indeptEstimator.estHist[targ.targetID].keys()
                    estHist = sat.indeptEstimator.estHist[targ.targetID]
                    covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                    trackError = sat.indeptEstimator.trackErrorHist[targ.targetID]
                    innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.indeptEstimator.innovationCovHist[
                        targ.targetID
                    ]

                    ci_times = sat.ciEstimator.estHist[targ.targetID].keys()
                    ci_estHist = sat.ciEstimator.estHist[targ.targetID]
                    ci_covHist = sat.ciEstimator.covarianceHist[targ.targetID]
                    ci_trackError = sat.ciEstimator.trackErrorHist[targ.targetID]

                    ci_innovation_times = sat.ciEstimator.innovationHist[
                        targ.targetID
                    ].keys()
                    ci_innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
                    ci_innovationCovHist = sat.ciEstimator.innovationCovHist[
                        targ.targetID
                    ]

                    et_times = sat.etEstimator.estHist[targ.targetID][sat][sat].keys()
                    et_estHist = sat.etEstimator.estHist[targ.targetID][sat][sat]
                    et_covHist = sat.etEstimator.covarianceHist[targ.targetID][sat][sat]
                    et_trackError = sat.etEstimator.trackErrorHist[targ.targetID][sat][
                        sat
                    ]

                    # File Name
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}.csv"

                    # Format the data and write it to the file
                    self.format_data(
                        filename,
                        targ.targetID,
                        times,
                        sat_hist,
                        trueHist,
                        sat_measHistTimes,
                        sat_measHist,
                        estTimes,
                        estHist,
                        covHist,
                        trackError,
                        innovationHist,
                        innovationCovHist,
                        ci_times,
                        ci_estHist,
                        ci_covHist,
                        ci_trackError,
                        ci_innovation_times,
                        ci_innovationHist,
                        ci_innovationCovHist,
                        et_times,
                        et_estHist,
                        et_covHist,
                        et_trackError,
                    )

    def format_data(
        self,
        filename,
        targetID,
        times,
        sat_hist,
        trueHist,
        sat_measHistTimes,
        sat_measHist,
        estTimes,
        estHist,
        covHist,
        trackError,
        innovationHist,
        innovationCovHist,
        ci_times,
        ci_estHist,
        ci_covHist,
        ci_trackError,
        ci_innovation_times,
        ci_innovationHist,
        ci_innovationCovHist,
        et_times,
        et_estHist,
        et_covHist,
        et_trackError,
    ) -> None:
        """
        Formats and writes data to a CSV file.

        Parameters:
        - filename (str): The name of the CSV file to be created.
        - targetID (int): The ID of the target.
        - times (array-like): A vector of time points.
        - sat_hist (dict): Satellite history data.
        - trueHist (dict): True history data of the target.
        - sat_measHistTimes (dict_keys): Measurement history times of the satellite.
        - sat_measHist (dict): Measurement history of the satellite.
        - estTimes (dict_keys): Estimation times.
        - estHist (dict): Estimation history.
        - covHist (dict): Covariance history.
        - trackError (dict): Track quality history.
        - innovationHist (dict): Innovation history.
        - innovationCovHist (dict): Innovation covariance history.
        - ci_times (dict_keys): DDF estimation times.
        - ci_estHist (dict): DDF estimation history.
        - ci_covHist (dict): DDF covariance history.
        - ci_trackError (dict): DDF track quality history.
        - ci_innovation_times (dict_keys): DDF innovation times.
        - ci_innovationHist (dict): DDF innovation history.
        - ci_innovationCovHist (dict): DDF innovation covariance history.
        - et_times (dict_keys): ET estimation times.
        - et_estHist (dict): ET estimation history.
        - et_covHist (dict): ET covariance history.
        - et_measHist (dict): ET measurement history.
        - et_trackError (dict): ET track quality history.

        Returns:
        None
        """
        precision = 3  # Set the desired precision

        def format_list(lst):
            if isinstance(lst, np.ndarray):
                return [
                    f"{x:.{precision}f}" if not np.isnan(x) else "nan"
                    for x in lst.flatten()
                ]
            elif isinstance(lst, int) or isinstance(lst, float):
                return [f"{float(lst):.{precision}f}" if not np.isnan(lst) else "nan"]
            else:
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst]

        # Create a single CSV file for the target
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            # Writing headers
            writer.writerow(
                [
                    'Time',
                    'x_sat',
                    'y_sat',
                    'z_sat',
                    'True_x',
                    'True_vx',
                    'True_y',
                    'True_vy',
                    'True_z',
                    'True_vz',
                    'InTrackAngle',
                    'CrossTrackAngle',
                    'Est_x',
                    'Est_vx',
                    'Est_y',
                    'Est_vy',
                    'Est_z',
                    'Est_vz',
                    'Cov_xx',
                    'Cov_vxvx',
                    'Cov_yy',
                    'Cov_vyvy',
                    'Cov_zz',
                    'Cov_vzvz',
                    'Track Uncertainty',
                    'Innovation_ITA',
                    'Innovation_CTA',
                    'InnovationCov_ITA',
                    'InnovationCov_CTA',
                    'ci_Est_x',
                    'ci_Est_vx',
                    'ci_Est_y',
                    'ci_Est_vy',
                    'ci_Est_z',
                    'ci_Est_vz',
                    'ci_Cov_xx',
                    'ci_Cov_vxvx',
                    'ci_Cov_yy',
                    'ci_Cov_vyvy',
                    'ci_Cov_zz',
                    'ci_Cov_vzvz',
                    'DDF Track Uncertainty',
                    'ci_Innovation_ITA',
                    'ci_Innovation_CTA',
                    'ci_InnovationCov_ITA',
                    'ci_InnovationCov_CTA',
                    'ET_Est_x',
                    'ET_Est_vx',
                    'ET_Est_y',
                    'ET_Est_vy',
                    'ET_Est_z',
                    'ET_Est_vz',
                    'ET_Cov_xx',
                    'ET_Cov_vxvx',
                    'ET_Cov_yy',
                    'ET_Cov_vyvy',
                    'ET_Cov_zz',
                    'ET_Cov_vzvz',
                    'ET_Track Error',
                ]
            )

            # Writing data rows
            for time in times:
                row = [f"{time:.{precision}f}"]
                row += format_list(sat_hist[time])
                row += format_list(trueHist[time])

                if time in sat_measHistTimes:
                    row += format_list(sat_measHist[time])

                if time in estTimes:
                    row += format_list(estHist[time])
                    row += format_list(np.diag(covHist[time]))
                    row += format_list(trackError[time])
                    row += format_list(innovationHist[time])
                    row += format_list(np.diag(innovationCovHist[time]))

                if time in ci_times:
                    row += format_list(ci_estHist[time])
                    row += format_list(np.diag(ci_covHist[time]))
                    row += format_list(ci_trackError[time])

                if time in ci_innovation_times:
                    row += format_list(ci_innovationHist[time])
                    row += format_list(np.diag(ci_innovationCovHist[time]))

                if time in et_times:
                    row += format_list(et_estHist[time])
                    row += format_list(np.diag(et_covHist[time]))
                    row += format_list(et_trackError[time])

                writer.writerow(row)

    def log_comms_data(
        self,
        time_vec: u.Quantity[u.minute],
        saveName: str,
        filePath: str = os.path.dirname(os.path.realpath(__file__)),
    ) -> None:
        for sat in self.sats:
            for targ in self.targs:
                if targ.targetID in sat.targetIDs:
                    commNode = self.comms.G.nodes[sat]
                    filename = (
                        f"{filePath}/data/{saveName}_{targ.name}_{sat.name}_comm.csv"
                    )
                    self.format_comms_data(
                        filename, time_vec.value, sat, commNode, targ.targetID
                    )

    def format_comms_data(
        self,
        filename: str,
        time_vec: u.Quantity[u.minute],
        sat: satellite.Satellite,
        commNode: comms.Comms,
        targetID: int,
    ) -> None:
        precision = 3

        def format_list(lst):
            if isinstance(lst, np.ndarray):
                return [
                    f"{x:.{precision}f}" if not np.isnan(x) else "nan"
                    for x in lst.flatten()
                ]
            elif isinstance(lst, int) or isinstance(lst, float):
                return [f"{float(lst):.{precision}f}" if not np.isnan(lst) else "nan"]
            else:
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst]

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    'Time',
                    'Satellite',
                    'Target',
                    'Sender',
                    "Received Alpha",
                    "Received Beta",
                    "Receiver",
                    "Sent Alpha",
                    "Sent Beta",
                ]
            )

            times = [time for time in time_vec]
            timeReceived = [time for time in commNode['received_measurements'].keys()]
            timesSent = [time for time in commNode['sent_measurements'].keys()]

            for time in times:
                row = [f"{time:.{precision}f}"]
                row += [sat.name]
                row += [f"Targ{targetID}"]

                if time in timeReceived:
                    for i in range(len(commNode['received_measurements'][time])):
                        row += [
                            commNode['received_measurements'][time][targetID]['sender'][
                                i
                            ].name
                        ]  # format_list(commNode['received_measurements'][time][targetID]['sender'][i])
                        row += format_list(
                            commNode['received_measurements'][time][targetID]['meas'][i]
                        )
                else:
                    row += ['', '', '']

                if time in timesSent:
                    for i in range(len(commNode['sent_measurements'][time])):
                        row += [
                            commNode['sent_measurements'][time][targetID]['receiver'][
                                i
                            ].name
                        ]  # format_list(commNode['sent_measurements'][time][targetID]['receiver'][i])
                        row += format_list(
                            commNode['sent_measurements'][time][targetID]['meas'][i]
                        )
                else:
                    row += ['', '', '']
                writer.writerow(row)

    @classmethod
    def from_config(cls, cfg: sim_config.SimConfig) -> 'Environment':
        targs = [
            target.Target(
                name=name,
                tqReq=t.tq_req,
                targetID=t.target_id,
                coords=np.array(t.coords),
                heading=t.heading,
                speed=t.speed,
                uncertainty=np.array(t.uncertainty),
                color=t.color,
            )
            for name, t in cfg.targets.items()
        ]

        sats = {
            name: satellite.Satellite(
                name=name,
                sensor=sensor.Sensor(
                    name=s.sensor,
                    fov=cfg.sensors[s.sensor].fov,
                    bearingsError=np.array(cfg.sensors[s.sensor].bearings_error),
                ),
                orbit=orbit.Orbit.from_sim_config(s.orbit),
                color=s.color,
            )
            for name, s in cfg.satellites.items()
        }

        # Define the goal of the system:
        commandersIntent: util.CommandersIndent = {
            time: {sat: intent for sat, intent in sat_intents.items()}
            for time, sat_intents in cfg.commanders_intent.items()
        }

        first_intent = next(iter(next(iter(commandersIntent.values())).values()))

        # commandersIntent[4] = {
        #     sat1a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        #     sat1b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        #     sat2a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        #     sat2b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        # }

        # Define the ground stations:
        groundStations = [
            groundStation.GroundStation(
                estimator=estimator.GsEstimator(),
                lat=gs.lat,
                lon=gs.lon,
                fov=gs.fov,
                commRange=gs.comms_range,
                name=name,
                color=gs.color,
            )
            for name, gs in cfg.ground_stations.items()
        ]

        # Define the communication network:
        comms_network = comms.Comms(
            list(sats.values()),
            maxBandwidth=cfg.comms.max_bandwidth,
            maxNeighbors=cfg.comms.max_neighbors,
            maxRange=u.Quantity(cfg.comms.max_range, u.km),
            minRange=u.Quantity(cfg.comms.min_range, u.km),
            displayStruct=cfg.comms.display_struct,
        )

        # Create and return an environment instance:
        return cls(
            list(sats.values()),
            targs,
            comms_network,
            groundStations,
            commandersIntent,
            cfg.estimators,
        )
