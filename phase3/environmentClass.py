# Import classes
import csv
import io
import os
import random
from collections import defaultdict

import imageio
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pulp
from astropy import units as u
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from numpy import typing as npt

from phase3 import commClass
from phase3 import estimatorClass
from phase3 import satelliteClass
from phase3 import targetClass
from phase3 import util

## Creates the environment class, which contains a vector of satellites all other parameters


class Environment:
    def __init__(
        self,
        sats: list[satelliteClass.Satellite],
        targs: list[targetClass.Target],
        comms: commClass.Comms,
        commandersIntent: util.CommandersIndent,
        localEstimatorBool: bool,
        centralEstimatorBool: bool,
        ciEstimatorBool: bool,
        etEstimatorBool: bool,
    ):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """

        # For each satellite, define its initial goal and initialize the estimation algorithms
        for sat in sats:
            targPriorityInitial = commandersIntent[0][
                sat
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
            self.localEstimatorBool = localEstimatorBool
            if localEstimatorBool:
                sat.indeptEstimator = estimatorClass.IndeptEstimator(
                    commandersIntent[0][sat]
                )  # initialize the independent estimator for these targets

            self.centralEstimatorBool = centralEstimatorBool
            if centralEstimatorBool:
                self.centralEstimator = estimatorClass.CentralEstimator(
                    commandersIntent[0][sat]
                )  # initialize the central estimator for these targets

            self.ciEstimatorBool = ciEstimatorBool
            if ciEstimatorBool:
                sat.ciEstimator = estimatorClass.CiEstimator(commandersIntent[0][sat])

            self.etEstimatorBool = etEstimatorBool
            if etEstimatorBool:
                sat.etEstimators = [
                    estimatorClass.EtEstimator(commandersIntent[0][sat], shareWith=None)
                ]

        ## Populate the environment variables
        self.sats = sats  # define the satellites

        self.targs = targs  # define the targets

        self.commandersIntent = commandersIntent  # define the commanders intent

        self.comms = comms  # define the communication network

        # Define variables to track the comms
        self.comms.total_comm_data = util.NestedDict()
        self.used_comm_data = util.NestedDict()

        # Initialize time parameter to 0
        self.time: u.Quantity[u.minute] = 0 * u.minute
        self.delta_t = None

        # Environemnt Plotting parameters
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

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
        self.z_earth = self.earth_r * np.outer(np.ones(np.size(u_earth)), np.cos(v_earth))

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
        time_vec: npt.NDArray,
        pause_step: float = 0.0001,
        saveName: str | None = None,
        show_env: bool = False,
        plot_estimation_results: bool = False,
        plot_communication_results: bool = False,
        plot_et_network: bool = False,
        plot_uncertainty_ellipses: bool = False,
        save_estimation_data: bool = False,
        save_communication_data: bool = False,
    ):
        """
        Simulate the environment over a time range.

        Args:
        - time_vec: Array of time steps to simulate over.
        - pause_step: Time to pause between each step in the simulation.
        - saveName: Name to save the simulation data to.
        - show_env: Flag to show the environment plot at each step.
        - plot_estimation_results: Flag to plot the estimation results.
        - plot_communication_results: Flag to plot the communication results.
        - plot_et_network: Flag to plot the dynamic communication network in ET.
        - plot_uncertainty_ellipses: Flag to plot the uncertainty ellipses.
        - save_estimation_data: Flag to save the estimation data.
        - save_communication_data: Flag to save the communication data.

        Returns:
        - Data collected during simulation.
        """

        print("Simulation Started")

        # Initialize based on the current time
        time_vec = time_vec + self.time
        self.delta_t = (time_vec[1] - time_vec[0]).to_value(time_vec.unit)
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
                    ][sat]
                    sat.targetIDs = sat.targPriority.keys()

            # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion()

            if show_env:
                # Update the plot environment
                self.plot()
                plt.pause(pause_step)
                plt.draw()

            if plot_et_network:
                # Update the dynamic comms plot
                self.plot_dynamic_comms()
                plt.pause(pause_step)
                plt.draw()

        print("Simulation Complete")

        # Plot the filter results
        if plot_estimation_results:
            self.plot_estimator_results(
                time_vec, saveName=saveName
            )  # marginal error, innovation, and NIS/NEES plots

        # Plot the commm results
        if plot_communication_results:

            # Make plots for total data sent and used throughout time
            self.plot_global_comms(saveName=saveName)
            self.plot_used_comms(saveName=saveName)

            # For the CI estimators, plot time hist of comms
            if self.ciEstimatorBool:
                self.plot_timeHist_comms_ci(saveName=saveName)

        # Save the uncertainty ellipse plots
        if plot_uncertainty_ellipses:
            self.plot_all_uncertainty_ellipses(time_vec)  # Uncertainty Ellipse Plots

        # Log the Data
        if save_estimation_data:
            self.log_data(time_vec, saveName=saveName)

        if save_communication_data:
            self.log_comms_data(time_vec, saveName=saveName)

        return

    def propagate(self, time_step: u.Quantity[u.minute]) -> None:
        """
        Propagate the satellites and targets over the given time step.
        """
        # Update the current time
        self.time += time_step

        time_val = self.time.to_value(
            self.time.unit
        )  # extract the numerical value of time

        # Update the time for all targets and satellites
        for targ in self.targs:
            targ.time = time_val
        for sat in self.sats:
            sat.time = time_val

        # Propagate the targets' positions
        for targ in self.targs:
            targ.propagate(
                time_step, self.time
            )  # Stores the history of target time and xyz position and velocity

        # Propagate the satellites
        for sat in self.sats:
            sat.orbit = sat.orbit.propagate(time_step)
            sat.orbitHist[sat.time] = (
                sat.orbit.r.value
            )  # Store the history of sat time and xyz position
            sat.velHist[sat.time] = (
                sat.orbit.v.value
            )  # Store the history of sat time and xyz velocity

        # Update the communication network for the new sat positions
        self.comms.make_edges(self.sats)

    def data_fusion(self) -> None:
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """
        # Collect all measurements for every satellite in the environement
        collectedFlag, measurements = self.collect_all_measurements()

        # If a central estimator is present, perform central fusion
        if self.centralEstimatorBool:
            self.central_fusion(collectedFlag, measurements)

        # Now send estimates for future CI
        if self.ciEstimatorBool:
            self.send_estimates_optimize()
            # self.send_estimates()

        # Now send measurements for future ET
        if self.etEstimatorBool:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            if self.ciEstimatorBool:
                sat.ciEstimator.CI(sat, self.comms)

            if self.etEstimatorBool:
                etEKF = sat.etEstimators[0]
                etEKF.event_trigger_processing(sat, self.time.to_value(), self.comms)

        # ET estimator needs prediction to happen at everytime step, thus, even if measurement is none we need to predict
        for sat in self.sats:
            if self.etEstimatorBool:
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
        def goodness(source, reciever, trackUncertainty, targetID):
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
        def generate_all_paths(graph, max_hops):
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

        ### DEBUGGING PRINTS

        # # Print the selected paths
        # print("Selected paths:")
        # for (path, targetID) in selected_paths:
        #     # also print the total goodness of the selected paths
        #     total_goodness = sum(
        #         goodness(path[0], path[i+1], trackUncertainty, targetID)
        #         for i in range(len(path) - 1)
        #     )
        #     # now loop through the satellites in the path, and print the satellite names, then print the total goodness
        #     print(
        #         [sat.name for sat in path],
        #         f"TargetID: {targetID}",
        #         f"Goodness: {total_goodness}",
        #     )

        # # Print the total bandwidht usage vs avaliable across the graph
        # total_bandwidth_usage = sum(
        #     fixed_bandwidth_consumption
        #     for (path, targetID) in selected_paths
        #     for i in range(len(path) - 1)
        # )
        # print(f"Total bandwidth usage: {total_bandwidth_usage}")
        # print(f"Total available bandwidth: {sum(self.comms.G[u][v]['maxBandwidth'] for u, v in self.comms.G.edges())}")

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
                    envTime = self.time.to_value()
                    # Skip if there are no measurements for this targetID
                    if isinstance(
                        sat.measurementHist[target.targetID][envTime], np.ndarray
                    ):
                        # This means satellite has a measurement for this target, now send it to neighbors
                        for neighbor in self.comms.G.neighbors(sat):
                            neighbor: satelliteClass.Satellite
                            # If target is not in neighbors priority list, skip
                            if targetID not in neighbor.targPriority.keys():
                                continue

                            # Get the most recent measurement time
                            satTime = max(
                                sat.measurementHist[targetID].keys()
                            )  #  this should be irrelevant and equal to  self.time since a measurement is sent on same timestep

                            local_EKF = sat.etEstimators[0]

                            # Create a new commonEKF between two satellites
                            commonEKF = None
                            for each_etEstimator in sat.etEstimators:
                                if each_etEstimator.shareWith == neighbor.name:
                                    commonEKF = each_etEstimator
                                    break

                            if (
                                commonEKF is None
                            ):  # or make a common filter if one doesn't exist
                                commonEKF = EtEstimator(
                                    local_EKF.targetPriorities, shareWith=neighbor.name
                                )
                                commonEKF.et_EKF_initialize(target, envTime)
                                sat.etEstimators.append(commonEKF)
                                commonEKF.synchronizeFlag[targetID][envTime] = True
                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.et_EKF_initialize(target, envTime)

                            # Create a local and common EKF for neighbor if it doesnt exist
                            neighbor_localEKF = neighbor.etEstimators[0]

                            if (
                                len(neighbor_localEKF.estHist[targetID]) == 1
                            ):  # if I don't have a local EKF, create one
                                neighbor_localEKF.et_EKF_initialize(target, envTime)
                                neighbor_localEKF.synchronizeFlag[targetID][
                                    envTime
                                ] = True

                            commonEKF = None
                            for each_etEstimator in neighbor.etEstimators:
                                if each_etEstimator.shareWith == sat.name:
                                    commonEKF = each_etEstimator
                                    break

                            if (
                                commonEKF is None
                            ):  # if I don't, create one and add it to etEstimators list
                                commonEKF = EtEstimator(
                                    neighbor.targPriority, shareWith=sat.name
                                )
                                commonEKF.et_EKF_initialize(target, envTime)
                                neighbor.etEstimators.append(commonEKF)
                                commonEKF.synchronizeFlag[targetID][envTime] = True

                            if len(commonEKF.estHist[targetID]) == 0:
                                commonEKF.et_EKF_initialize(target, envTime)

                                # Create implicit and explicit measurements vector for this neighbor
                            meas = local_EKF.event_trigger(
                                sat, neighbor, targetID, satTime
                            )

                            # Send that to neightbor
                            self.comms.send_measurements(
                                sat, neighbor, meas, targetID, satTime
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

    ### 3D Dynamic Environment Plot ###
    def plot(self):
        """
        Plot the current state of the environment.
        """
        self.resetPlot()
        self.plotEarth()
        self.plotSatellites()
        self.plotTargets()
        self.plotCommunication()
        self.plotLegend_Time()
        self.save_envPlot_to_imgs()

    def resetPlot(self):
        """
        Reset the plot by removing all lines, collections, and texts.
        """
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()
        for text in self.ax.texts:
            text.remove()

    def plotSatellites(self):
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

    def plotTargets(self):
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

            # self.ax.quiver(x, y, z, vx * 1000, vy * 1000, vz * 1000, color=targ.color, arrow_length_ratio=0.75, label=targ.name)

            # do a standard scatter plot for the target
            self.ax.scatter(x, y, z, s=40, color=targ.color, label=targ.name)

    def plotEarth(self):
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

        # Uncomment this for a POV sat1 viewing angle for mono-track case
        # self.calcViewingAngle()

    def calcViewingAngle(self) -> None:
        '''
        Calculate the viewing angle for the 3D plot in MonoTrack Case
        '''
        monoTarg = self.targs[0]
        x, y, z = monoTarg.pos
        range = jnp.sqrt(x**2 + y**2 + z**2)

        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x) * 180 / jnp.pi

        self.ax.view_init(elev=30, azim=azimuth)

    def save_envPlot_to_imgs(self) -> None:
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(
            np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4)
        )[:, :, 0:4]
        self.imgs.append(img)

    ### Estimation Errors and Track Uncertainty Plots ###
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
                    if self.localEstimatorBool:
                        localEKF = sat.indeptEstimator
                    if self.centralEstimatorBool:
                        centralEKF = self.centralEstimator
                    if self.ciEstimatorBool:
                        ciEKF = sat.ciEstimator
                    if self.etEstimatorBool:
                        etEKF = sat.etEstimators[0]

                    fig = plt.figure(figsize=(15, 8))
                    fig.suptitle(f"{targ.name}, {sat.name}", fontsize=14)
                    axes = self.setup_axes(fig, state_labels, meas_labels)
                    handles = []

                    # Check, do we have local estimates?
                    if self.localEstimatorBool:
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
                    if self.centralEstimatorBool:
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
                    if self.ciEstimatorBool:
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
                    if self.etEstimatorBool:
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

    # TODO: THIS IS ORIGINAL VERSION USING DIFFERENT PLOTS DEPENDING ON ESTIMATOR COMBO
    def plot_estimator_results_2(
        self, time_vec: u.Quantity[u.minute], saveName: str
    ) -> None:
        """
        Create three types of plots: Local vs Central, CI vs Central, and Local vs CI vs Central.

        Args:
            time_vec: List of time values.
            saveName: Name for the saved plot file.
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
        suffix_vec = ['local', 'ci', 'et', 'et_vs_ci', 'et_pairwise']
        title_vec = ['Local vs Central', 'CI vs Central', 'ET vs Central', 'ET vs CI']
        title_vec = [title + " Estimator Results" for title in title_vec]

        # For Each Target
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Set up colors
                    satColor = sat.color
                    ciColor = '#DC143C'  # Crimson
                    centralColor = '#070400'  #'#228B22' # Forest Green

                    targetID = targ.targetID
                    trueHist = targ.hist
                    if self.localEstimatorBool:
                        localEKF = sat.indeptEstimator
                    if self.centralEstimatorBool:
                        centralEKF = self.centralEstimator
                    if self.ciEstimatorBool:
                        ciEKF = sat.ciEstimator
                    if self.etEstimatorBool:
                        etEKF = sat.etEstimators[0]

                    for plotNum in range(4):

                        fig = plt.figure(figsize=(15, 8))
                        fig.suptitle(
                            f"{targ.name}, {sat.name} {title_vec[plotNum]}", fontsize=14
                        )
                        axes = self.setup_axes(fig, state_labels, meas_labels)

                        if plotNum == 0:
                            """ " First Plot: Local vs Central"""

                            # First, check if the central estimator was used
                            if not self.centralEstimatorBool:
                                continue

                            (
                                times,
                                estHist,
                                covHist,
                                innovationHist,
                                innovationCovHist,
                                trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=localEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec,'local')
                            (
                                central_times,
                                central_estHist,
                                central_covHist,
                                central_innovationHist,
                                central_innovationCovHist,
                                central_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=centralEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'central')

                            # Local
                            self.plot_errors(
                                axes,
                                times,
                                estHist,
                                trueHist,
                                covHist,
                                label_color=satColor,
                                linewidth=2.5,
                            )
                            # self.plot_innovations(axes, times, innovationHist, innovationCovHist, label_color=satColor, linewidth=2.5)
                            self.plot_track_uncertainty(
                                axes,
                                times,
                                trackErrorHist,
                                targ.tqReq,
                                label_color=satColor,
                                linewidth=2.5,
                            )

                            # Central
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
                                central_times,
                                central_trackErrorHist,
                                targ.tqReq,
                                label_color=centralColor,
                                linewidth=1.5,
                            )

                            handles = [
                                patches.Patch(
                                    color=satColor,
                                    label=f'{sat.name} Indept. Estimator',
                                ),
                                patches.Patch(
                                    color=centralColor, label=f'Central Estimator'
                                ),
                            ]

                        elif plotNum == 1:
                            """ " Second Plot: CI vs Central"""

                            # First check if the CI estimator was used
                            if not self.ciEstimatorBool:
                                continue

                            (
                                ci_times,
                                ci_estHist,
                                ci_covHist,
                                ci_innovationHist,
                                ci_innovationCovHist,
                                ci_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=ciEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'ci')
                            (
                                central_times,
                                central_estHist,
                                central_covHist,
                                central_innovationHist,
                                central_innovationCovHist,
                                central_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=centralEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'central')

                            # CI
                            self.plot_errors(
                                axes,
                                ci_times,
                                ci_estHist,
                                trueHist,
                                ci_covHist,
                                label_color=ciColor,
                                linewidth=2.5,
                            )
                            # self.plot_innovations(axes, ci_times, ci_innovationHist, ci_innovationCovHist, label_color=ciColor, linewidth=2.5)
                            self.plot_track_uncertainty(
                                axes,
                                ci_times,
                                ci_trackErrorHist,
                                targ.tqReq,
                                label_color=ciColor,
                                linewidth=2.5,
                            )

                            # Central
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
                                central_times,
                                central_trackErrorHist,
                                targ.tqReq,
                                label_color=centralColor,
                                linewidth=1.5,
                            )

                            handles = [
                                patches.Patch(
                                    color=ciColor, label=f'{sat.name} CI Estimator'
                                ),
                                patches.Patch(
                                    color=centralColor, label=f'Central Estimator'
                                ),
                            ]

                        elif plotNum == 2:
                            """ " Third Plot is ET vs Central"""

                            # First check, was the ET estimator used?
                            if not self.etEstimatorBool:
                                continue

                            # Third Plot is ET vs Central
                            (
                                et_times,
                                et_estHist,
                                et_covHist,
                                et_innovationHist,
                                et_innovationCovHist,
                                et_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=etEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            (
                                central_times,
                                central_estHist,
                                central_covHist,
                                central_innovationHist,
                                central_innovationCovHist,
                                central_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=centralEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'central')

                            # ET
                            self.plot_errors(
                                axes,
                                et_times,
                                et_estHist,
                                trueHist,
                                et_covHist,
                                label_color=satColor,
                                linewidth=2.5,
                            )
                            self.plot_track_uncertainty(
                                axes,
                                et_times,
                                et_trackErrorHist,
                                targ.tqReq,
                                label_color=satColor,
                                linewidth=2.5,
                            )

                            # Central
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
                                central_times,
                                central_trackErrorHist,
                                targ.tqReq,
                                label_color=centralColor,
                                linewidth=1.5,
                            )

                            handles = [
                                patches.Patch(
                                    color=satColor, label=f'{sat.name} ET Estimator'
                                ),
                                patches.Patch(
                                    color=centralColor, label=f'Central Estimator'
                                ),
                            ]

                        elif plotNum == 3:
                            # ET vs CI

                            # First check, was the ET estimator used?
                            if not self.etEstimatorBool or not self.ciEstimatorBool:
                                continue

                            (
                                et_times,
                                et_estHist,
                                et_covHist,
                                et_innovationHist,
                                et_innovationCovHist,
                                et_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=etEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            (
                                ci_times,
                                ci_estHist,
                                ci_covHist,
                                ci_innovationHist,
                                ci_innovationCovHist,
                                ci_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=ciEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'ci')

                            # ET
                            self.plot_errors(
                                axes,
                                et_times,
                                et_estHist,
                                trueHist,
                                et_covHist,
                                label_color=satColor,
                                linewidth=2.5,
                            )
                            self.plot_track_uncertainty(
                                axes,
                                et_times,
                                et_trackErrorHist,
                                targ.tqReq,
                                label_color=satColor,
                                linewidth=2.5,
                            )

                            # CI
                            self.plot_errors(
                                axes,
                                ci_times,
                                ci_estHist,
                                trueHist,
                                ci_covHist,
                                label_color=ciColor,
                                linewidth=1.5,
                            )
                            self.plot_track_uncertainty(
                                axes,
                                ci_times,
                                ci_trackErrorHist,
                                targ.tqReq,
                                label_color=ciColor,
                                linewidth=1.5,
                            )

                            handles = [
                                patches.Patch(
                                    color=satColor, label=f'{sat.name} ET Estimator'
                                ),
                                patches.Patch(
                                    color=ciColor, label=f'{sat.name} CI Estimator'
                                ),
                            ]

                        # Add the legend and tighten the layout
                        fig.legend(
                            handles=handles,
                            loc='lower right',
                            ncol=3,
                            bbox_to_anchor=(1, 0),
                        )
                        plt.tight_layout()

                        # Save the Plot with respective suffix
                        self.save_plot(fig, saveName, targ, sat, suffix_vec[plotNum])

                        # Close the figure to save memory
                        plt.close(fig)

                    ## PLOTTING PAIRWISE ET THINGS ##
                    if not self.etEstimatorBool:
                        continue

                    for sat2 in self.sats:
                        if sat != sat2 and sat2.targetIDs == sat.targetIDs:

                            fig = plt.figure(figsize=(15, 8))
                            fig.suptitle(
                                f"{targ.name}, {sat.name}, {sat2.name} ET Filters",
                                fontsize=14,
                            )
                            axes = self.setup_axes(fig, state_labels, meas_labels)
                            etEKF = sat.etEstimators[0]
                            sat12_etEKF = None
                            for each_etestimator in sat.etEstimators:
                                if each_etestimator.shareWith == sat2.name:
                                    sat12_etEKF = each_etestimator
                                    break

                            sat2_etEKF = sat2.etEstimators[0]
                            sat21_etEKF = None
                            for each_etestimator in sat2.etEstimators:
                                if each_etestimator.shareWith == sat.name:
                                    sat21_etEKF = each_etestimator
                                    break

                            sat2Color = sat2.color
                            sat1commonColor, sat2commonColor = self.shifted_colors(
                                satColor, sat2Color, shift=50
                            )

                            (
                                et_times,
                                et_estHist,
                                et_covHist,
                                et_innovationHist,
                                et_innovationCovHist,
                                et_trackErrorHist,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=etEKF
                            )  # self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            (
                                et_times2,
                                et_estHist2,
                                et_covHist2,
                                et_innovationHist2,
                                et_innovationCovHist2,
                                et_trackErrorHist2,
                            ) = self.getEstimationHistory(
                                targetID, time_vec, filter=sat2_etEKF
                            )  # self.getEstimationHistory(sat2, targ, time_vec, 'et', sharewith=sat2)

                            # ET
                            self.plot_errors(
                                axes,
                                et_times,
                                et_estHist,
                                trueHist,
                                et_covHist,
                                label_color=satColor,
                                linewidth=2.0,
                            )
                            self.plot_track_uncertainty(
                                axes,
                                et_times,
                                et_trackErrorHist,
                                targ.tqReq,
                                label_color=satColor,
                                linewidth=2.0,
                            )

                            # ET 2
                            self.plot_errors(
                                axes,
                                et_times2,
                                et_estHist2,
                                trueHist,
                                et_covHist2,
                                label_color=sat2Color,
                                linewidth=2.0,
                            )
                            self.plot_track_uncertainty(
                                axes,
                                et_times2,
                                et_trackErrorHist2,
                                targ.tqReq,
                                label_color=sat2Color,
                                linewidth=2.0,
                            )

                            if sat12_etEKF is not None:
                                (
                                    et_common_times,
                                    et_common_estHist,
                                    et_common_covHist,
                                    et_common_innovationHist,
                                    et_common_innovationCovHist,
                                    et_common_trackErrorHist,
                                ) = self.getEstimationHistory(
                                    targetID, time_vec, filter=sat12_etEKF
                                )  # self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat2)

                                # Common ET
                                self.plot_errors(
                                    axes,
                                    et_common_times,
                                    et_common_estHist,
                                    trueHist,
                                    et_common_covHist,
                                    label_color=sat1commonColor,
                                    linewidth=2.0,
                                )
                                self.plot_track_uncertainty(
                                    axes,
                                    et_common_times,
                                    et_common_trackErrorHist,
                                    targ.tqReq,
                                    label_color=sat1commonColor,
                                    linewidth=2.0,
                                )

                            if sat21_etEKF is not None:
                                (
                                    et_common_times2,
                                    et_common_estHist2,
                                    et_common_covHist2,
                                    et_common_innovationHist2,
                                    et_common_innovationCovHist2,
                                    et_common_trackErrorHist2,
                                ) = self.getEstimationHistory(
                                    targetID, time_vec, filter=sat21_etEKF
                                )  # self.getEstimationHistory(sat2, targ, time_vec, 'et', sharewith=sat)

                                # Common ET 2
                                self.plot_errors(
                                    axes,
                                    et_common_times2,
                                    et_common_estHist2,
                                    trueHist,
                                    et_common_covHist2,
                                    label_color=sat2commonColor,
                                    linewidth=2.0,
                                )
                                self.plot_track_uncertainty(
                                    axes,
                                    et_common_times2,
                                    et_common_trackErrorHist2,
                                    targ.tqReq,
                                    label_color=sat2commonColor,
                                    linewidth=2.0,
                                )

                            # Plot Messages instead of innovations
                            self.plot_messages(
                                axes[6], sat, sat2, targ.targetID, time_vec.value
                            )
                            self.plot_messages(
                                axes[7], sat2, sat, targ.targetID, time_vec.value
                            )

                            handles = [
                                patches.Patch(
                                    color=satColor, label=f'{sat.name} ET Estimator'
                                ),
                                patches.Patch(
                                    color=sat2Color, label=f'{sat2.name} ET Estimator'
                                ),
                                patches.Patch(
                                    color=sat1commonColor,
                                    label=f'{sat.name}, {sat2.name} Common ET Estimator',
                                ),
                                patches.Patch(
                                    color=sat2commonColor,
                                    label=f'{sat2.name}, {sat.name} Common ET Estimator',
                                ),
                            ]

                            fig.legend(
                                handles=handles,
                                loc='lower center',
                                ncol=4,
                                bbox_to_anchor=(0.5, 0),
                            )
                            plt.tight_layout()

                            # Save the Plot with respective suffix
                            currSuffix = f"{sat2.name}_" + suffix_vec[4]
                            self.save_plot(fig, saveName, targ, sat, currSuffix)

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

    def getEstimationHistory2(self, sat, targ, time_vec, estimatorType, sharewith=None):
        """
        Get the estimation history for a given satellite, target, and estimator type.

        Args:
        - sat: satellite object.
        - targ: target object.
        - estimatorType: string indicating the type of estimator.

        Returns:
        - estHist: dictionary of estimation history.
        - covHist: dictionary of covariance history.
        - innovationHist: dictionary of innovation history.
        - innovationCovHist: dictionary of innovation covariance history.
        - trackErrorHist: dictionary of track quality history.
        """
        times, estHist, covHist, innovationHist, innovationCovHist, trackErrorHist = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        if estimatorType == 'central':
            times = [
                time
                for time in time_vec.value
                if time in self.centralEstimator.estHist[targ.targetID]
            ]
            estHist = self.centralEstimator.estHist[targ.targetID]
            covHist = self.centralEstimator.covarianceHist[targ.targetID]
            # innovationHist = self.centralEstimator.innovationHist[targ.targetID]
            # innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]

        elif estimatorType == 'ci':
            times = [
                time
                for time in time_vec.value
                if time in sat.ciEstimator.estHist[targ.targetID]
            ]
            estHist = sat.ciEstimator.estHist[targ.targetID]
            covHist = sat.ciEstimator.covarianceHist[targ.targetID]
            innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
            innovationCovHist = sat.ciEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = sat.ciEstimator.trackErrorHist[targ.targetID]

        elif estimatorType == 'et':
            times = [
                time
                for time in time_vec.value
                if time in sat.etEstimator.estHist[targ.targetID][sat][sharewith]
            ]
            estHist = sat.etEstimator.estHist[targ.targetID][sat][sharewith]
            covHist = sat.etEstimator.covarianceHist[targ.targetID][sat][sharewith]
            # innovationHist = sat.etEstimator.innovationHist[targ.targetID][sat][sharewith]
            # innovationCovHist = sat.etEstimator.innovationCovHist[targ.targetID][sat][sharewith]
            trackErrorHist = sat.etEstimator.trackErrorHist[targ.targetID][sat][
                sharewith
            ]

        elif estimatorType == 'local':
            times = [
                time
                for time in time_vec.value
                if time in sat.indeptEstimator.estHist[targ.targetID]
            ]
            estHist = sat.indeptEstimator.estHist[targ.targetID]
            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]

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
        for i in range(6):  # For all six states [x, vx, y, vy, z, vz]
            if times:  # If there is an estimate on target
                segments = self.segment_data(times, max_gap=self.delta_t * 2)
                for segment in segments:
                    est_errors = [
                        estHist[time][i] - trueHist[time][i] for time in segment
                    ]
                    upper_bound = [2 * np.sqrt(covHist[time][i][i]) for time in segment]
                    lower_bound = [
                        -2 * np.sqrt(covHist[time][i][i]) for time in segment
                    ]

                    ax[i].plot(
                        segment, est_errors, color=label_color, linewidth=linewidth
                    )
                    ax[i].plot(
                        segment,
                        upper_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )
                    ax[i].plot(
                        segment,
                        lower_bound,
                        color=label_color,
                        linestyle='dashed',
                        linewidth=linewidth,
                    )

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

    def plot_messages(self, ax, sat, sat2, targetID, timeVec):

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

            if isinstance(
                self.comms.used_comm_et_data_values[targetID][sat.name][sat2.name][
                    time
                ],
                np.ndarray,
            ):
                alpha, beta = self.comms.used_comm_et_data_values[targetID][sat.name][
                    sat2.name
                ][time]
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

    def segment_data(self, times, max_gap=1 / 6):
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
        # Finally plot a dashed line for the targetPriority
        # axes[8].axhline(y=targQuality*50 + 50, color='k', linestyle='dashed', linewidth=1.5)
        #         # Add a text label on the above right side of the dashed line
        # axes[8].text(1, targQuality*50 + 50 + 5, f"Target Quality: {targQuality}", fontsize=8, color='k')

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
    def plot_global_comms(self, saveName):
        """PLOTS THE TOTAL DATA SEND AND RECEIVED BY SATELLITES IN DDF ALGORITHMS"""

        ## Plot comm data sent for CI Algo
        if self.ciEstimatorBool:

            # Create a figure
            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(f"TOTAL Data Sent and Received by Satellites", fontsize=14)
            ax = fig.add_subplot(111)

            # Get the names of satellites:
            satNames = [sat.name for sat in self.sats]

            # Save previous data, to stack the bars
            # prev_data = np.zeros(len(satNames))
            # make prev_data a dictionary
            prev_data = {sat: 0 for sat in satNames}

            # Loop through all targets, in order listed in the environment
            # for target_id in self.comms.total_comm_data:
            count = 0
            for targ in self.targs:

                sent_data = defaultdict(dict)
                rec_data = defaultdict(dict)

                # Get the color for the target:
                color = targ.color

                # Get the target id
                target_id = targ.targetID

                # Now check, does that target have any communication data
                if target_id not in self.comms.total_comm_data:
                    continue

                count += 1

                for reciever in self.comms.total_comm_data[target_id]:

                    for sender in self.comms.total_comm_data[target_id][reciever]:
                        if sender == reciever:
                            continue

                        # Goal is to count the amoutn of data reciever has receieved as well as sender has sent

                        for time in self.comms.total_comm_data[target_id][reciever][
                            sender
                        ]:

                            # Get the data
                            data = self.comms.total_comm_data[target_id][reciever][
                                sender
                            ][time]

                            # Count the amount of data receiver by the receiver
                            if reciever not in rec_data:
                                rec_data[reciever] = 0
                            rec_data[reciever] += data

                            # Count the amount of data sent by the sender
                            if sender not in sent_data:
                                sent_data[sender] = 0
                            sent_data[sender] += data

                # If there are keys that dont exist in sent_data, make them and their value 0
                for key in prev_data.keys():
                    if key not in sent_data:
                        sent_data[key] = 0
                    if key not in rec_data:
                        rec_data[key] = 0

                # Order the data the same way, according to "sats" variable
                sent_data = dict(
                    sorted(sent_data.items(), key=lambda item: satNames.index(item[0]))
                )
                rec_data = dict(
                    sorted(rec_data.items(), key=lambda item: satNames.index(item[0]))
                )

                p1 = ax.bar(
                    list(sent_data.keys()),
                    list(sent_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                )

                # Add text labels to show which target is which.
                for i, v in enumerate(list(sent_data.values())):
                    ax.text(
                        i,
                        list(prev_data.values())[i],
                        targ.name,
                        ha='center',
                        va='bottom',
                        color='black',
                    )

                # Add the sent_data values to the prev_data
                for key in sent_data.keys():
                    prev_data[key] += sent_data[key]

                p2 = ax.bar(
                    list(rec_data.keys()),
                    list(rec_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                    fill=False,
                    hatch='//',
                    edgecolor=color,
                )

                # Add the rec_data values to the prev_data
                for key in rec_data.keys():
                    prev_data[key] += rec_data[key]

                if count == 1:
                    # Add legend
                    ax.legend((p1[0], p2[0]), ('Sent Data', 'Received Data'))

            # Add the labels
            ax.set_ylabel('Total Data Sent/Recieved (# of numbers)')

            # Add the x-axis labels
            ax.set_xticks(np.arange(len(satNames)))
            ax.set_xticklabels(satNames)

            # Now save the plot
            if saveName is not None:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                os.makedirs(plotPath, exist_ok=True)
                plt.savefig(
                    os.path.join(plotPath, f"{saveName}_total_ci_comms.png"), dpi=300
                )
            else:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                plt.savefig(os.path.join(plotPath, f"total_ci_comms.png"), dpi=300)

        ## Plot comm data sent for ET Algo
        if self.etEstimatorBool:

            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(f"ET Data Sent and Received by Satellites", fontsize=14)

            ax = fig.add_subplot(111)

            # Get the names of satellites:
            satNames = [sat.name for sat in self.sats]

            # Save previous data, to stack the bars
            prev_data = {sat: 0 for sat in satNames}

            # Loop through all targets, in order listed in the environment
            count = 0
            for targ in self.targs:

                sent_data = defaultdict(dict)
                rec_data = defaultdict(dict)

                # Get the color for the target:
                color = targ.color

                # Get the target id
                target_id = targ.targetID

                # Now check, does that target have any communication data
                if target_id not in self.comms.total_comm_et_data:
                    continue

                count += 1

                for reciever in self.comms.total_comm_et_data[target_id]:

                    for sender in self.comms.total_comm_et_data[target_id][reciever]:
                        if sender == reciever:
                            continue

                        # Goal is to count the amoutn of data reciever has receieved as well as sender has sent

                        for time in self.comms.total_comm_et_data[target_id][reciever][
                            sender
                        ]:

                            # Get the data
                            data = self.comms.total_comm_et_data[target_id][reciever][
                                sender
                            ][time]

                            # Count the amount of data receiver by the receiver
                            if reciever not in rec_data:
                                rec_data[reciever] = 0
                            rec_data[reciever] += data

                            # Count the amount of data sent by the sender
                            if sender not in sent_data:
                                sent_data[sender] = 0
                            sent_data[sender] += data

                # If there are keys that dont exist in sent_data, make them and their value 0
                for key in prev_data.keys():
                    if key not in sent_data:
                        sent_data[key] = 0
                    if key not in rec_data:
                        rec_data[key] = 0

                # Order the data the same way, according to "sats" variable
                sent_data = dict(
                    sorted(sent_data.items(), key=lambda item: satNames.index(item[0]))
                )
                rec_data = dict(
                    sorted(rec_data.items(), key=lambda item: satNames.index(item[0]))
                )

                p1 = ax.bar(
                    list(sent_data.keys()),
                    list(sent_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                )

                # Add text labels to show which target is which.
                for i, v in enumerate(list(sent_data.values())):
                    ax.text(
                        i,
                        list(prev_data.values())[i],
                        targ.name,
                        ha='center',
                        va='bottom',
                        color='black',
                    )

                # Add the sent_data values to the prev_data
                for key in sent_data.keys():
                    prev_data[key] += sent_data[key]

                p2 = ax.bar(
                    list(rec_data.keys()),
                    list(rec_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                    fill=False,
                    hatch='//',
                    edgecolor=color,
                )

                # Add the rec_data values to the prev_data
                for key in rec_data.keys():
                    prev_data[key] += rec_data[key]

                if count == 1:
                    # Add legend
                    ax.legend((p1[0], p2[0]), ('Sent Data', 'Received Data'))

            # Add the labels
            ax.set_ylabel('ET Data Sent/Recieved (# of numbers)')

            # Add the x-axis labels
            ax.set_xticks(np.arange(len(satNames)))
            ax.set_xticklabels(satNames)

            # Now save the plot
            filePath = os.path.dirname(os.path.realpath(__file__))
            plotPath = os.path.join(filePath, 'plots')
            os.makedirs(plotPath, exist_ok=True)
            plt.savefig(
                os.path.join(plotPath, f"{saveName}_total_et_comms.png"), dpi=300
            )

    # Plots the actual data amount used by the satellites
    def plot_used_comms(self, saveName):
        """
        PLOTS THE USED DATA SEND AND RECEIVED BY SATELLITES IN DDF ALGORITHMS

            Used means information used for a sat to meet TQ requirements

        """

        ## Plot comm data sent for CI Algo
        if self.ciEstimatorBool:

            # Create a figure
            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(f"USED Data Sent and Received by Satellites", fontsize=14)
            ax = fig.add_subplot(111)

            # Get the names of satellites:
            satNames = [sat.name for sat in self.sats]

            # Save previous data, to stack the bars
            # prev_data = np.zeros(len(satNames))
            # make prev_data a dictionary
            prev_data = {sat: 0 for sat in satNames}

            # Loop through all targets, in order listed in the environment
            # for target_id in self.comms.total_comm_data:
            count = 0
            for targ in self.targs:

                sent_data = defaultdict(dict)
                rec_data = defaultdict(dict)

                # Get the color for the target:
                color = targ.color

                # Get the target id
                target_id = targ.targetID

                # Now check, does that target have any communication data
                if target_id not in self.comms.used_comm_data:
                    continue

                count += 1

                for reciever in self.comms.used_comm_data[target_id]:

                    for sender in self.comms.used_comm_data[target_id][reciever]:
                        if sender == reciever:
                            continue

                        # Goal is to count the amoutn of data reciever has receieved as well as sender has sent

                        for time in self.comms.used_comm_data[target_id][reciever][
                            sender
                        ]:

                            # Get the data
                            data = self.comms.used_comm_data[target_id][reciever][
                                sender
                            ][time]

                            # Count the amount of data receiver by the receiver
                            if reciever not in rec_data:
                                rec_data[reciever] = 0
                            rec_data[reciever] += data

                            # Count the amount of data sent by the sender
                            if sender not in sent_data:
                                sent_data[sender] = 0
                            sent_data[sender] += data

                # If there are keys that dont exist in sent_data, make them and their value 0
                for key in prev_data.keys():
                    if key not in sent_data:
                        sent_data[key] = 0
                    if key not in rec_data:
                        rec_data[key] = 0

                # Order the data the same way, according to "sats" variable
                sent_data = dict(
                    sorted(sent_data.items(), key=lambda item: satNames.index(item[0]))
                )
                rec_data = dict(
                    sorted(rec_data.items(), key=lambda item: satNames.index(item[0]))
                )

                p1 = ax.bar(
                    list(sent_data.keys()),
                    list(sent_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                )

                # Add text labels to show which target is which.
                for i, v in enumerate(list(sent_data.values())):
                    ax.text(
                        i,
                        list(prev_data.values())[i],
                        targ.name,
                        ha='center',
                        va='bottom',
                        color='black',
                    )

                # Add the sent_data values to the prev_data
                for key in sent_data.keys():
                    prev_data[key] += sent_data[key]

                p2 = ax.bar(
                    list(rec_data.keys()),
                    list(rec_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                    fill=False,
                    hatch='//',
                    edgecolor=color,
                )

                # Add the rec_data values to the prev_data
                for key in rec_data.keys():
                    prev_data[key] += rec_data[key]

                if count == 1:
                    # Add legend
                    ax.legend((p1[0], p2[0]), ('Sent Data', 'Received Data'))

            # Add the labels
            ax.set_ylabel('Used Data Sent/Recieved (# of numbers)')

            # Add the x-axis labels
            ax.set_xticks(np.arange(len(satNames)))
            ax.set_xticklabels(satNames)

            # Now save the plot
            if saveName is not None:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                os.makedirs(plotPath, exist_ok=True)
                plt.savefig(
                    os.path.join(plotPath, f"{saveName}_used_ci_comms.png"), dpi=300
                )
            else:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                plt.savefig(os.path.join(plotPath, f"used_ci_comms.png"), dpi=300)

        ## Plot comm data sent for ET Algo
        if self.etEstimatorBool:

            # DO the exact same thing for ET data
            # Create a figure
            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(f"USED ET Data Sent and Received by Satellites", fontsize=14)
            ax = fig.add_subplot(111)

            # Get the names of satellites:
            satNames = [sat.name for sat in self.sats]

            # Save previous data, to stack the bars
            # prev_data = np.zeros(len(satNames))
            # make prev_data a dictionary
            prev_data = {sat: 0 for sat in satNames}

            # Loop through all targets, in order listed in the environment
            # for target_id in self.comms.total_comm_data:
            count = 0
            for targ in self.targs:

                sent_data = defaultdict(dict)
                rec_data = defaultdict(dict)

                # Get the color for the target:
                color = targ.color

                # Get the target id
                target_id = targ.targetID

                # Now check, does that target have any communication data
                if target_id not in self.comms.used_comm_et_data:
                    continue

                count += 1

                for reciever in self.comms.used_comm_et_data[target_id]:

                    for sender in self.comms.used_comm_et_data[target_id][reciever]:
                        if sender == reciever:
                            continue

                        # Goal is to count the amoutn of data reciever has receieved as well as sender has sent

                        for time in self.comms.used_comm_et_data[target_id][reciever][
                            sender
                        ]:

                            # Get the data
                            data = self.comms.used_comm_et_data[target_id][reciever][
                                sender
                            ][time]

                            # Count the amount of data receiver by the receiver
                            if reciever not in rec_data:
                                rec_data[reciever] = 0
                            rec_data[reciever] += data

                            # Count the amount of data sent by the sender
                            if sender not in sent_data:
                                sent_data[sender] = 0
                            sent_data[sender] += data

                # If there are keys that dont exist in sent_data, make them and their value 0
                for key in prev_data.keys():
                    if key not in sent_data:
                        sent_data[key] = 0
                    if key not in rec_data:
                        rec_data[key] = 0

                # Order the data the same way, according to "sats" variable
                sent_data = dict(
                    sorted(sent_data.items(), key=lambda item: satNames.index(item[0]))
                )
                rec_data = dict(
                    sorted(rec_data.items(), key=lambda item: satNames.index(item[0]))
                )

                p1 = ax.bar(
                    list(sent_data.keys()),
                    list(sent_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                )

                # Add text labels to show which target is which.
                for i, v in enumerate(list(sent_data.values())):
                    ax.text(
                        i,
                        list(prev_data.values())[i],
                        targ.name,
                        ha='center',
                        va='bottom',
                        color='black',
                    )

                # Add the sent_data values to the prev_data
                for key in sent_data.keys():
                    prev_data[key] += sent_data[key]

                p2 = ax.bar(
                    list(rec_data.keys()),
                    list(rec_data.values()),
                    bottom=list(prev_data.values()),
                    color=color,
                    fill=False,
                    hatch='//',
                    edgecolor=color,
                )

                # Add the rec_data values to the prev_data
                for key in rec_data.keys():
                    prev_data[key] += rec_data[key]

                if count == 1:
                    # Add legend
                    ax.legend((p1[0], p2[0]), ('Sent Data', 'Received Data'))

            # Add the labels
            ax.set_ylabel('Used Data Sent/Recieved (# of numbers)')

            # Add the x-axis labels
            ax.set_xticks(np.arange(len(satNames)))
            ax.set_xticklabels(satNames)

            # Now save the plot
            if saveName is not None:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                os.makedirs(plotPath, exist_ok=True)
                plt.savefig(
                    os.path.join(plotPath, f"{saveName}_used_et_comms.png"), dpi=300
                )

    # Sub plots for each satellite showing the track uncertainty for each target and then the comms sent/recieved about each target vs time
    def plot_timeHist_comms_ci(self, saveName):
        """PLOTS A TIME HISTORY OF THE CI COMMS RECIEVED FOR EACH SATELLITE ON EVERY TARGET"""

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
                        y=self.commandersIntent[time][sat][targ.targetID],
                        xmin=xMin,
                        xmax=xMax,
                        color=targ.color,
                        linestyle='dashed',
                        linewidth=1.5,
                    )

                    # Now plot a dashed line for the targetPriority
                    # ax1.axhline(y=self.commandersIntent[time][sat][targ.targetID], color=targ.color, linestyle='dashed', linewidth=1.5)
                    # Add a text label on the above right side of the dashed line
                    # ax1.text(xMin, self.commandersIntent[time][sat][targ.targetID] + 5, f"Target Quality: {targ.tqReq}", fontsize=8, color=targ.color)

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
            satNames = [sat.name for sat in self.sats]

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
                if targ.targetID not in self.comms.used_comm_data:
                    continue

                for sender in satNames:
                    if sender == sat:
                        continue

                    # So now, we have a satellite, the reciever, recieving information about targetID from sender
                    # We want to count, how much information did the reciever recieve from the sender in a time history and plot that on a bar chart

                    data = self.comms.used_comm_data[targ.targetID][sat.name][sender]

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

    # TODO: IDK WHAT THIS DOES - NOLAN. IT IS ONLINE PLOTITNG FOR THE GRAPH?
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

                    # If the satellites synchronize, add an edge
                    if sat.etEstimator.synchronizeFlag[targetID][sat][sat2]:
                        if (
                            sat.etEstimator.synchronizeFlag[targetID][sat][sat2].get(
                                envTime
                            )
                            == True
                        ):
                            diComms.add_edge(sat, sat2)
                            style = self.get_edge_style(
                                comms, targetID, sat, sat2, envTime, CI=True
                            )
                            edge_styles.append((sat, sat2, style, targ_color))

                    value = comms.used_comm_et_data_values[targetID][sat.name][
                        sat2.name
                    ][envTime]
                    print(f"Receiver {sat.name}, Sender {sat2.name}, Value {value}")
                    # If there is a communication between the two satellites, add an edge
                    if isinstance(
                        comms.used_comm_et_data_values[targ.targetID][sat.name][
                            sat2.name
                        ][envTime],
                        np.ndarray,
                    ):
                        diComms.add_edge(sat2, sat)
                        style = self.get_edge_style(comms, targetID, sat, sat2, envTime)
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

    def get_edge_style(self, comms, targetID, sat1, sat2, envTime, CI=False):
        """
        Helper function to determine the edge style based on communication data.
        Returns 'solid' if both alpha and beta are present, 'dashed' if only one is present,
        and None if neither is present (meaning no line).
        """

        if CI:
            return (0, ())

        alpha, beta = comms.used_comm_et_data_values[targetID][sat1.name][sat2.name][
            envTime
        ]
        if np.isnan(alpha) and np.isnan(beta):
            return (0, (1, 10))
        elif np.isnan(alpha) or np.isnan(beta):
            return (0, (3, 10, 1, 10))
        else:
            return (0, (5, 10))

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
                                        self.make_legened1(
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

                                        self.make_legened1(
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
                                        self.make_legened2(
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
                                        self.make_legened2(
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

    def plot_ellipsoid(self, ax, est_pos, cov_matrix, color, alpha):
        """
        Plots a 3D ellipsoid representing the uncertainty of the estimated position.

        Args:
        ax (Axes3D): The 3D axis to plot on.
        est_pos (np.ndarray): The estimated position.
        cov_matrix (np.ndarray): The covariance matrix.
        color (str): The color of the ellipsoid.
        alpha (float): The transparency level of the ellipsoid.

        Returns:
        None
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        radii = np.sqrt(eigenvalues)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        ellipsoid_points = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        transformed_points = ellipsoid_points @ eigenvectors.T + est_pos
        x_transformed = transformed_points[:, 0].reshape(x.shape)
        y_transformed = transformed_points[:, 1].reshape(y.shape)
        z_transformed = transformed_points[:, 2].reshape(z.shape)
        ax.plot_surface(
            x_transformed, y_transformed, z_transformed, color=color, alpha=alpha
        )

    def plot_estimate(self, ax, est_pos, true_pos, satColor):
        """
        Plots the estimated and true positions on the given axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        - est_pos: The estimated position.
        - true_pos: The true position.
        - satColor: The color for the satellite position marker.

        Returns:
        None
        """
        ax.scatter(est_pos[0], est_pos[1], est_pos[2], color=satColor, marker='o')
        ax.scatter(true_pos[0], true_pos[1], true_pos[2], color='g', marker='o')

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

    def make_legened1(
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

    def make_legened2(self, ax, ciColor, centralColor, error1, error2, ci_type=None):
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

    def render_gif(
        self,
        fileType,
        saveName,
        filePath=os.path.dirname(os.path.realpath(__file__)),
        fps=1,
    ):
        """
        Renders and saves GIFs based on the specified file type.

        Parameters:
        - fileType (str): The type of GIF to render. Options are 'satellite_simulation' or 'uncertainty_ellipse'.
        - saveName (str): The base name for the saved GIF files.
        - filePath (str): The directory path where the GIF files will be saved. Defaults to the directory of the script.
        - fps (int): Frames per second for the GIF. Defaults to 10.

        Returns:
        None
        """
        frame_duration = 1000 / fps  # in ms

        if fileType == 'satellite_simulation':
            file = os.path.join(filePath, 'gifs', f'{saveName}_satellite_sim.gif')
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self.imgs:
                    writer.append_data(img)

        if fileType == 'uncertainty_ellipse':
            for targ in self.targs:
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        for sat2 in self.sats:
                            if targ.targetID in sat2.targetIDs:
                                if sat != sat2:
                                    file = os.path.join(
                                        filePath,
                                        'gifs',
                                        f"{saveName}_{targ.name}_{sat.name}_{sat2.name}_stereo_GE.gif",
                                    )
                                    with imageio.get_writer(
                                        file, mode='I', duration=frame_duration
                                    ) as writer:
                                        for img in self.imgs_stereo_GE[targ.targetID][
                                            sat
                                        ][sat2]:
                                            writer.append_data(img)

        if fileType == 'dynamic_comms':
            file = os.path.join(filePath, 'gifs', f"{saveName}_dynamic_comms.gif")
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self.imgs_dyn_comms:
                    writer.append_data(img)

    ### Data Dump File ###
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
        sat: satelliteClass.Satellite,
        commNode: commClass.Comms,
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

    ### NEES/NIS Data Collection; NOT WORKING RN ###
    def collectNISNEESData(self) -> dict[int, defaultdict]:
        # TODO: THIS ONLY WORKS FOR CI ESTIMATOR AT THE MOMENT, DONT USE OTHERWISE!
        """
        Collects NEES and NIS data for the simulation in an easy-to-read format.

        Returns:
        dict: A dictionary containing NEES and NIS data for each targetID and satellite.
        """
        # Create a dictionary of targetIDs
        data = {
            targetID: defaultdict(dict)
            for targetID in (targ.targetID for targ in self.targs)
        }

        # Now for each targetID, make a dictionary for each satellite
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Extract the data
                    data[targ.targetID][sat.name] = {
                        'NEES': sat.indeptEstimator.neesHist[targ.targetID],
                        'NIS': sat.indeptEstimator.nisHist[targ.targetID],
                    }
                    # Also save the DDF data
                    data[targ.targetID][f"{sat.name} DDF"] = {
                        'NEES': sat.ciEstimator.neesHist[targ.targetID],
                        'NIS': sat.ciEstimator.nisHist[targ.targetID],
                    }

            # If central estimator is used, also add that data
            if self.centralEstimator:
                if targ.targetID in self.centralEstimator.neesHist:
                    data[targ.targetID]['Central'] = {
                        'NEES': self.centralEstimator.neesHist[targ.targetID],
                        'NIS': self.centralEstimator.nisHist[targ.targetID],
                    }

        return data
