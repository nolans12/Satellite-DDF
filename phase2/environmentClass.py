from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment:
    def __init__(self, sats, targs, comms, centralEstimator=None, ciEstimator=False, etEstimator=False):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """
        # Use the provided central estimator or default to None
        self.centralEstimator = centralEstimator if centralEstimator else None
        self.ciEstimator = ciEstimator
        self.etEstimator = etEstimator
        
        # Define the satellites
        self.sats = sats
        
        # Define the targets
        self.targs = targs
        
        # Define the communication network
        self.comms = comms

        # Define variables to track the comms
        self.comms.total_comm_data = NestedDict()
        self.used_comm_data = NestedDict()
        
        # Initialize time parameter to 0
        self.time = 0
        
        # Plotting parameters
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits and view angle, original limits for mono plot
        # self.ax.set_xlim([-15000, 15000])
        # self.ax.set_ylim([-15000, 15000])
        # self.ax.set_zlim([-15000, 15000])
        # self.ax.view_init(elev=30, azim=30)
        # self.ax.set_box_aspect([1, 1, 1])

        # If you want to do standard case:
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
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.earth_r = 6378.0
        self.x_earth = self.earth_r * np.outer(np.cos(u), np.sin(v))
        self.y_earth = self.earth_r * np.outer(np.sin(u), np.sin(v))
        self.z_earth = self.earth_r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Empty lists and dictionaries for simulation images
        self.imgs = []
        self.imgs_local_GE = defaultdict(lambda: defaultdict(list))
        self.imgs_ddf_GE = defaultdict(lambda: defaultdict(list))
        self.imgs_cent_GE = defaultdict(lambda: defaultdict(list))
        self.imgs_stereo_GE = defaultdict(lambda: defaultdict(list))
        self.imgs_stereo_ET_GE = defaultdict(lambda: defaultdict(list))

        self.tempCount = 0


    def simulate(self, time_vec, pause_step=0.1, showSim=False, plotEstimators=False, plotComms = False, saveGif=False, saveData=False, saveName=None):
        """
        Simulate the environment over a time range.
        
        Args:
        - time_vec: numpy array of time steps with poliastro units associated.
        - pause_step: time to pause between each step if displaying as animation (default: 0.1).
        - plotEstimators: boolean, if True will save plots (default: False).
        - saveData: boolean, if True will save data (default: False).
        - saveName: optional name for saving plots and data (default: None).
        - showSim: boolean, if True will display the plot as an animation (default: False).
        
        Returns:
        - Data collected during simulation.
        """
        
        # Initialize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time  # Get delta time to propagate, works because propagate func increases time after first itr
            
            # Propagate the satellites and environments position
            self.propagate(t_d)

            # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion()

            if plotEstimators:
                # Update the plot environment
                self.plot()

                if showSim:
                    # Display the plot
                    plt.pause(pause_step)
                    plt.draw()

        if plotEstimators:
            # Plot the results of the simulation.
            self.plot_estimator_results(time_vec, plotEstimators=plotEstimators, saveName=saveName) # marginal error, innovation, and NIS/NEES plots

        if plotComms:
            # Plot the comm results
            self.plot_global_comms(saveName=saveName)
            self.plot_used_comms(saveName=saveName)
            self.plot_local_comms(saveName=saveName)
           
        if saveGif:
            # Save the uncertainty ellipse plots
            self.plot_all_uncertainty_ellipses() # Uncertainty Ellipse Plots

        # Log the Data
        if saveData:
            self.log_data(time_vec, saveName=saveName)

        return self.collectNISNEESData()  # Return the data collected during the simulation
    

    def data_fusion(self):
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """
        # Collect all measurements
        collectedFlag, measurements = self.collect_all_measurements()

        # If a central estimator is present, perform central fusion
        if self.centralEstimator:
            self.central_fusion(collectedFlag, measurements)

        # Now send estimates for future CI
        if self.ciEstimator:
            # self.send_estimates_optimize()
            self.send_estimates()

        # Now send measurements for future ET
        if self.etEstimator:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            if self.ciEstimator:
                sat.ciEstimator.CI(sat, self.comms)
            if self.etEstimator:
                sat.etEstimator.event_triggered_fusion(sat, self.time.to_value(), self.comms.G.nodes[sat])


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

            # For each targetID in satellites object to track, add the track error to the dictionaryg
            for targetID in sat.targetIDs:

                # Check, is there a track error for this targetID?
                if not bool(sat.ciEstimator.trackErrorHist[targetID]) or len(sat.ciEstimator.trackErrorHist[targetID].keys()) == 0:
                    # Make the track uncertainty 999
                    trackUncertainty[sat.name][targetID] = 999
                    continue # Skip to next targetID

                # Otherwise, get the most recent trackUncertainty
                maxTime = max(sat.ciEstimator.trackErrorHist[targetID].keys())

                # Add the track uncertainty to the dictionary
                trackUncertainty[sat.name][targetID] = sat.ciEstimator.trackErrorHist[targetID][maxTime]

        # Now that we have the track uncertainties, we can optimize the communication network

        #### MIXED INTEGER LINEAR PROGRAMMING ####
        # Redefine goodness function to be based on a source and reciever node pair, not a path:
        def goodness(source, reciever, trackUncertainty, targetID):
            """ A paths goodness is defined as the sum of the deltas in track uncertainty on a targetID, as far as that node hasnt already recieved data from that satellite"""

            # Get the track uncertainty of the source node
            sourceTrackUncertainty = trackUncertainty[source.name][targetID]

            # Get the track uncertainty of the target node
            recieverTrackUncertainty = trackUncertainty[reciever.name][targetID]

            # Get the desired targetID track uncertainty
            desired = 50 + 50*targetID

            # Check, if the sats track uncertainty on that targetID needs help or not
            if recieverTrackUncertainty < desired:
                return 0
        
            # Else, calculate the goodness, + if the source is better, 0 if the sat is better
            if recieverTrackUncertainty - sourceTrackUncertainty < 0: 
                return 0 # EX: If i have uncertainty of 200 and share it with a sat with 100, theres no benefit to sharing that
            

            # TODO:
                # maybe make this return piecewise reward, if source is already at threshold, cap the reward to be reciever - threshold
                
            # Else, return the goodness of the link, difference between the two track uncertainties
            return recieverTrackUncertainty - sourceTrackUncertainty 
        

        # Now goal is to find the set of paths that maximize the total goodness, while also respecting the bandwidth constraints and not double counting, farying information is allowed

        # Generate all possible non cyclic paths up to a reasonable length (e.g., max 3 hops)
        def generate_all_paths(graph, max_hops):
            paths = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=max_hops):
                            paths.append(tuple(path))
            return paths
        
        # Generate all possible paths
        allPaths = generate_all_paths(self.comms.G, 3)

        # Define the fixed bandwidth consumption per CI
        fixed_bandwidth_consumption = 30

        # Define the optimization problem
        prob = pulp.LpProblem("Path_Optimization", pulp.LpMaximize)

        # Create binary decision variables for each path combination
        # 1 if the path is selected, 0 otherwise
        path_selection_vars = pulp.LpVariable.dicts(
            "path_selection", [(path, targetID) for path in allPaths for targetID in trackUncertainty[path[0].name].keys()], 0, 1, pulp.LpBinary
        )

        #### OBJECTIVE FUNCTION

        ## Define the objective function, total sum of goodness across all paths
        # Initalize a linear expression that will be used as the objective
        total_goodness_expression = pulp.LpAffineExpression()

        for path in allPaths: # Loop through all paths possible

            for targetID in trackUncertainty[path[0].name].keys(): # Loop through all targetIDs that a path could be talking about

                # Initalize a linear expression that will define the goodness of a path in talking about a targetID
                path_goodness = pulp.LpAffineExpression() 

                # Loop through the links of the path
                for i in range(len(path) - 1):

                    # Get the goodness of a link in the path on the specified targetID
                    edge_goodness = goodness(path[0], path[i+1], trackUncertainty, targetID)
                
                    # Add the edge goodness to the path goodness
                    path_goodness += edge_goodness

                # Thus we are left with a value for the goodness of the path in talking about targetID
                # But, we dont know if we are going to take that path, thats up to the optimizer
                # So make it a binary expression, so that if the path is selected,
                # the path_goodness will be added to the total_goodness_expression. 
                # Otherwsie if the path isn't selected, the path_goodness will be 0
                total_goodness_expression += path_goodness * path_selection_vars[(path, targetID)]

        # Add the total goodness expression to the linear programming problem as the objective function
        prob += total_goodness_expression, "Total_Goodness_Objective"

        #### CONSTRAINTS

        ## Ensure the total bandwidth consumption across a link does not exceed the bandwidth constraints
        for edge in self.comms.G.edges(): # Loop through all edges possible
            u, v = edge  # Unpack the edge

            # Create a list to accumulate the terms for the total bandwidth usage on this edge
            bandwidth_usage_terms = []
            
            # Iterate over all possible paths
            for (path, targetID) in path_selection_vars:

                # Check if the current path includes the edge in question
                if any((path[i], path[i+1]) == edge for i in range(len(path) - 1)):

                    # Now path_selection_vars is a binary expression/condition will either be 0 or 1
                    # Thus, the following term essentially accounts for the bandwidth usage on this edge, if its used
                    bandwidth_usage = path_selection_vars[(path, targetID)] * fixed_bandwidth_consumption

                    # Add the term to the list
                    bandwidth_usage_terms.append(bandwidth_usage)
            
            # Sum all the expressions in the list to get the total bandwidth usage on this edge
            total_bandwidth_usage = pulp.lpSum(bandwidth_usage_terms)

            # Add the constraint to the linear programming problem
            # The constraint indicates that the total bandwidth usage on this edge should not exceed the bandwidth constraint
            # This constraint will be added for all edges in the graph after the loop
            prob += total_bandwidth_usage <= self.comms.G[u][v]['maxBandwidth'], f"Bandwidth_constraint_{edge}"

        ## Ensure the reward for sharing information about a targetID from source node to another node is not double counted
        for source in self.comms.G.nodes(): # Loop over all source nodes possible

            for receiver in self.comms.G.nodes(): # Loop over all receiver nodes possible

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
                        prob += path_count <= 1, f"Single_path_for_target_{source}_{receiver}_{targetID}"


        # Solve the optimization problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Output the results, paths selected
        selected_paths = [
            (path, targetID)
            for (path, targetID) in path_selection_vars
            if path_selection_vars[(path, targetID)].value() == 1
        ]
        
        ### DEBUGGING PRINTS

        # Print the selected paths
        print("Selected paths:")
        for (path, targetID) in selected_paths:
            # also print the total goodness of the selected paths
            total_goodness = sum(
                goodness(path[0], path[i+1], trackUncertainty, targetID)
                for i in range(len(path) - 1)
            )
            # now loop through the satellites in the path, and print the satellite names, then print the total goodness
            print(
                [sat.name for sat in path],
                f"TargetID: {targetID}",
                f"Goodness: {total_goodness}",
            )

        # Print the total bandwidht usage vs avaliable across the graph
        total_bandwidth_usage = sum(
            fixed_bandwidth_consumption
            for (path, targetID) in selected_paths
            for i in range(len(path) - 1)
        )
        print(f"Total bandwidth usage: {total_bandwidth_usage}")
        print(f"Total available bandwidth: {sum(self.comms.G[u][v]['maxBandwidth'] for u, v in self.comms.G.edges())}")

        # Now, we have the selected paths, we can send the estimates
        for (path, targetID) in selected_paths:

            # Get the est, cov, and time of the most recent estimate
            sourceSat = path[0]

            # Get the most recent estimate time
            sourceTime = max(sourceSat.ciEstimator.estHist[targetID].keys())
            est = sourceSat.ciEstimator.estHist[targetID][sourceTime]
            cov = sourceSat.ciEstimator.covarianceHist[targetID][sourceTime]

            # Get the most recent 
            self.comms.send_estimate_path(path, est, cov, targetID, sourceTime)


        test = 1 
        

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
                if len(sat.ciEstimator.estHist[targetID].keys()) == 0:
                    continue  

                # This means satellite has an estimate for this target, now send it to neighbors
                neighbors = list(self.comms.G.neighbors(sat))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    # Get the most recent estimate time
                    satTime = max(sat.ciEstimator.estHist[targetID].keys())

                    est = sat.ciEstimator.estHist[targetID][satTime]
                    cov = sat.ciEstimator.covarianceHist[targetID][satTime]

                    # Send most recent estimate to neighbor
                    self.comms.send_estimate(
                        sat, 
                        neighbor, 
                        est,
                        cov,
                        targetID, 
                        satTime
                    )


    def send_measurements(self):
        """
        Send the most recent measurements from each satellite to its neighbors.
        """
        # Loop through all satellites
        for sat in self.sats:
            # For each targetID in satellites measurement history        
            for targetID in sat.measurementHist.keys():
                # Skip if there are no measurements for this targetID
                if len(sat.measurementHist[targetID][self.time.value]) == 0:
                    continue
                
                # This means satellite has a measurement for this target, now send it to neighbors
                for neighbor in self.comms.G.neighbors(sat):
                    # Get the most recent measurement time
                    satTime = max(sat.measurementHist[targetID].keys())
                    
                    # Create implicit and explicit measurements vector for this neighbor
                    meas = sat.etEstimator.event_trigger(sat, neighbor, targetID, satTime)
                    
                    # Store the measurement I am sending to neighbor
                    sat.etEstimator.measHist[targetID][sat][neighbor][satTime] = meas
                    
                    # Send that to neightbor
                    self.comms.send_measurements(
                        sat, 
                        neighbor, 
                        meas, 
                        targetID, 
                        satTime
                    )


    def collect_all_measurements(self):
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

                    if collectedFlag[targ][sat]: # If a measurement was collected
                        measurements[targ][sat] = sat.measurementHist[targ.targetID][self.time.to_value()] # Store the measurement in the dictionary

        return collectedFlag, measurements # Return the collected flags and measurements


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
            satsWithMeasurements = [sat for sat in self.sats if collectedFlag[targ][sat]]
            newMeasurements = [measurements[targ][sat] for sat in satsWithMeasurements]

            # If any satellite took a measurement on this target    
            if satsWithMeasurements:
                # Run EKF with all satellites that took a measurement on the target
                self.centralEstimator.EKF(satsWithMeasurements, newMeasurements, targ, self.time.to_value())


    def propagate(self, time_step):
        """
        Propagate the satellites and targets over the given time step.
        """
        # Update the current time
        self.time += time_step
        # print("Time: ", self.time.to_value())

        time_val = self.time.to_value(self.time.unit)
        # Update the time in targs, sats, and estimator
        for targ in self.targs:
            targ.time = time_val
        for sat in self.sats:
            sat.time = time_val

        # Propagate the targets' positions
        for targ in self.targs:
            # Propagate the target
            targ.propagate(time_step, self.time)

        # Propagate the satellites
        for sat in self.sats:
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)
            # Store the history of sat time and xyz position
            sat.orbitHist[sat.time] = sat.orbit.r.value

        # Update the communication network for the new sat positions
        self.comms.make_edges(self.sats)
            
        
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
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=sat.color, marker='x')

            box = np.array([points[0], points[3], points[1], points[2], points[0]])
            self.ax.add_collection3d(Poly3DCollection([box], facecolors=sat.color, linewidths=1, edgecolors=sat.color, alpha=.1))


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
        self.ax.plot_surface(self.x_earth, self.y_earth, self.z_earth, color='k', alpha=0.1)
        # ### ALSO USE IF YOU WANT EARTH TO NOT BE SEE THROUGH
        # self.ax.plot_surface(self.x_earth*0.9, self.y_earth*0.9, self.z_earth*0.9, color = 'white', alpha=1) 
    

    def plotCommunication(self):
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
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], color=(0.3, 1.0, 0.3), linestyle='dashed', linewidth=2)
                else:
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], color='k', linestyle='dashed', linewidth=1)


    def plotLegend_Time(self):
        """
        Plot the legend and the current simulation time.
        """
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.ax.text2D(0.05, 0.95, f"Time: {self.time:.2f}", transform=self.ax.transAxes)
    
    
    def save_envPlot_to_imgs(self):
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        self.imgs.append(img)


### Estimation Errors and Track Error Plots ###
    def plot_estimator_results(self, time_vec, plotEstimators, saveName):
        """
        Create three types of plots: Local vs Central, DDF vs Central, and Local vs DDF vs Central.

        Args:
            time_vec (list): List of time values.
            plotEstimators (bool): Flag indicating whether to save the plot.
            saveName (str): Name for the saved plot file.
        """
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/min]', 'Y [km]', 'Vy [km/min]', 'Z [km]', 'Vz [km/min]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]', 'Track Error [km]']

        # For each target and satellite, plot the estimation error, innovation, and track quality
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    for k in range(4):

                        # TODO: THIS ONLY PLOTS CI AND CENTRAL
                        if k != 3:
                            continue


                        # Create a figure
                        fig = plt.figure(figsize=(15, 8))
                        fig.suptitle(f"{targ.name}, {sat.name} Estimation Error and Innovation Plots", fontsize=14)
                        axes = self.setup_axes(fig, state_labels, meas_labels)
                        
                        # Get the Satellite Color and Target History
                        satColor = sat.color
                        trueHist = targ.hist

                        if k == 0:  # Local vs Central
                            # Get the Data
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]
                            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]
                            
                            # Get the valid times for data
                            times = [time for time in time_vec.value if time in estHist]
                            innovation_times = [time for time in time_vec.value if time in innovationHist]
                            nisnees_times = [time for time in time_vec.value if time in NISHist]
                            trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, times, innovation_times, nisnees_times, trackError_times, estHist, trueHist, covHist, 
                                                    innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, targ.tqReq, 
                                                    satColor, linewidth=2.5)

                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the Data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times for data
                                times = [time for time in time_vec.value if time in estHist]
                                trackError_times = [time for time in time_vec.value if time in trackErrorHist] 
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], trackError_times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, targ.tqReq, '#228B22', linewidth=1.5, c=True)

                            # Create a Patch for Legend
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                            
                        elif k == 1:  # DDF vs Central
                            # Get the Data
                            ddf_estHist = sat.ciEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ciEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ciEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ciEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ciEstimator.neesHist[targ.targetID]
                            ddf_trackErrorHist = sat.ciEstimator.trackErrorHist[targ.targetID]

                            # Get the valid times for data
                            ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
                            ddf_trackError_times = [time for time in time_vec.value if time in ddf_trackErrorHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, ddf_times, ddf_innovation_times, ddf_NISNEES_times, 
                                                    ddf_trackError_times, ddf_estHist, trueHist, ddf_covHist, 
                                                    ddf_innovationHist, ddf_innovationCovHist, ddf_NISHist, ddf_NEESHist, 
                                                    ddf_trackErrorHist, targ.tqReq, '#DC143C', linewidth=2.5)

                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the Data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times for data
                                times = [time for time in time_vec.value if time in estHist]
                                trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], trackError_times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, targ.tqReq, '#228B22', linewidth=1.5, c=True)
                            
                            # Create Patch for Legend
                            handles = [
                                Patch(color='#DC143C', label=f'{sat.name} DDF Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                        
                        elif k == 2: # ET-Measurements vs Central
                            # Get the Local Data
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]
                            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]
                            
                            # Get ET Data
                            et_estHist = sat.etEstimator.estHist[targ.targetID][sat][sat]
                            et_covHist = sat.etEstimator.covarianceHist[targ.targetID][sat][sat]
                            #et_innovationHist = sat.etEstimator.innovationHist[targ.targetID][sat][sat]
                            #et_innovationCovHist = sat.etEstimator.innovationCovHist[targ.targetID][sat][sat]
                            
                            # Get the valid times for data
                            times = [time for time in time_vec.value if time in estHist]
                            innovation_times = [time for time in time_vec.value if time in innovationHist]
                            nisnees_times = [time for time in time_vec.value if time in NISHist]
                            trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                            et_times = [time for time in time_vec.value if time in et_estHist]
                            trackError_et_times = [time for time in time_vec.value if time in et_estHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, times, innovation_times, nisnees_times, trackError_times, estHist, trueHist, covHist,
                                                    innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, targ.tqReq,
                                                    satColor, linewidth=3.5)
                            
                            self.plot_estimator_data(fig, axes, et_times, et_times, et_times, trackError_et_times, et_estHist, trueHist, et_covHist,
                                                    [], [], NISHist, NEESHist, trackErrorHist, targ.tqReq,
                                                    '#DC143C', linewidth=2, e = True)
                            
                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times
                                times = [time for time in time_vec.value if time in estHist]
                                trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], trackError_times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, targ.tqReq, '#228B22', linewidth=1.5, c=True)

                            # Create a patch for the legend
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color='#DC143C', label=f'{sat.name} ET Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]        

                        elif k == 3:  # Local vs DDF vs Central
                            # Get the Local Data
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]
                            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]
                            
                            # Get the DDF Data
                            ddf_estHist = sat.ciEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ciEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ciEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ciEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ciEstimator.neesHist[targ.targetID]
                            ddf_trackErrorHist = sat.ciEstimator.trackErrorHist[targ.targetID]
                            
                            # Get the valid times for data
                            times = [time for time in time_vec.value if time in estHist]
                            innovation_times = [time for time in time_vec.value if time in innovationHist]
                            nisnees_times = [time for time in time_vec.value if time in NISHist]
                            trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                            ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
                            ddf_trackError_times = [time for time in time_vec.value if time in ddf_trackErrorHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, times, innovation_times, nisnees_times, trackError_times, estHist, trueHist, covHist, 
                                                    innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, targ.tqReq,
                                                    satColor, linewidth=2.5)
                            self.plot_estimator_data(fig, axes, ddf_times, ddf_innovation_times, ddf_NISNEES_times, 
                                                    ddf_trackError_times, ddf_estHist, trueHist, ddf_covHist, 
                                                    ddf_innovationHist, ddf_innovationCovHist, ddf_NISHist, ddf_NEESHist, 
                                                    ddf_trackErrorHist, targ.tqReq, '#DC143C', linewidth=2.0, ci=True)

                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times
                                times = [time for time in time_vec.value if time in estHist]
                                trackError_times = [time for time in time_vec.value if time in trackErrorHist]
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], trackError_times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, targ.tqReq, '#228B22', linewidth=1.5, c=True)

                            # Create a patch for the legend
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color='#DC143C', label=f'{sat.name} CI Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                        
                        # Create a legend   
                        fig.legend(handles=handles, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                        plt.tight_layout()
                        
                        # Save the Plot with respective suffix
                        suffix = ['indept', 'ddf', 'et', 'ci'][k]
                        self.save_plot(fig, plotEstimators, saveName, targ, sat, suffix)
                        
                        if k == 2:
                            self.plot_messages(plotEstimators, saveName)


    def plot_estimator_data(self, fig, axes, estTimes, innTimes, nnTimes, tqTimes, estHist, trueHist, covHist, innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, targQuality, label_color, linewidth, c=False, ci=False, e=False):
        """
        Plot all data.

        Args:
            fig (matplotlib.figure.Figure): The figure to plot on.
            axes (list): List of axes to plot on.
            estTimes (list): List of times for the estimates.
            innTimes (list): List of times for the innovations.
            nnTimes (list): List of times for the NIS/NEES.
            tqTimes (list): List of times for the track quality.
            estHist (dict): Dictionary of estimated histories.
            trueHist (dict): Dictionary of true histories.
            covHist (dict): Dictionary of covariance histories.
            innovationHist (dict): Dictionary of innovation histories.
            innovationCovHist (dict): Dictionary of innovation covariance histories.
            NISHist (dict): Dictionary of NIS histories.
            NEESHist (dict): Dictionary of NEES histories.
            trackErrorHist (dict): Dictionary of track quality histories.
            label_color (str): Color for the plot.
            linewidth (float): Width of the plot lines.
            c (bool): Flag indicating whether it is a central estimator.
        """
        if e and not c:  # ET-Measurements vs Central
            self.plot_errors(axes, estTimes, estHist, trueHist, covHist, label_color, linewidth)
            
        elif c and not e:  # Not central estimator so plot everything
            self.plot_errors(axes, estTimes, estHist, trueHist, covHist, label_color, linewidth)
            self.plot_track_quality(axes, tqTimes, trackErrorHist, targQuality, label_color, linewidth)
        
        else:  # Central estimator doesn't have innovation data
            self.plot_errors(axes, estTimes, estHist, trueHist, covHist, label_color, linewidth)
            self.plot_innovations(axes, innTimes, innovationHist, innovationCovHist, label_color, linewidth)
            self.plot_track_quality(axes, tqTimes, trackErrorHist, targQuality, label_color, linewidth, ci = ci)

            
    def plot_errors(self, ax, times, estHist, trueHist, covHist, label_color, linewidth):
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
                segments = self.segment_data(times)
                for segment in segments:
                    est_errors = [estHist[time][i] - trueHist[time][i] for time in segment]
                    upper_bound = [2 * np.sqrt(covHist[time][i][i]) for time in segment]
                    lower_bound = [-2 * np.sqrt(covHist[time][i][i]) for time in segment]
                    
                    ax[i].plot(segment, est_errors, color=label_color, linewidth=linewidth)
                    ax[i].plot(segment, upper_bound, color=label_color, linestyle='dashed', linewidth=linewidth)
                    ax[i].plot(segment, lower_bound, color=label_color, linestyle='dashed', linewidth=linewidth)


    def plot_innovations(self, ax, times, innovationHist, innovationCovHist, label_color, linewidth):
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
            segments = self.segment_data(times)

            for i in range(2):  # For each measurement [in track, cross track]
                for segment in segments:
                    innovation = [innovationHist[time][i] for time in segment]
                    upper_bound = [2 * np.sqrt(innovationCovHist[time][i][i]) for time in segment]
                    lower_bound = [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in segment]
                    
                    ax[6 + i].plot(segment, innovation, color=label_color, linewidth=linewidth)
                    ax[6 + i].plot(segment, upper_bound, color=label_color, linestyle='dashed', linewidth=linewidth)
                    ax[6 + i].plot(segment, lower_bound, color=label_color, linestyle='dashed', linewidth=linewidth)


    def plot_track_quality(self, ax, times, trackErrorHist, targQuality, label_color, linewidth, ci=False):
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
            segments = self.segment_data(times)
        
            for segment in segments:
                # Figure out, does this segment have a real data point in every time step
                new_time = []
                for time in segment:
                    if not not trackErrorHist[time]:
                        new_time.append(time)
                        nonEmptyTime.append(time)

                track_quality = [trackErrorHist[time] for time in new_time]
                ax[8].plot(new_time, track_quality, color=label_color, linewidth=linewidth)

            if ci:
                # Finally plot a dashed line for the targetPriority
                ax[8].axhline(y=targQuality*50 + 50, color='k', linestyle='dashed', linewidth=1.5)
                # Add a text label on the above right side of the dashed line
                ax[8].text(min(nonEmptyTime), targQuality*50 + 50 + 5, f"Target Quality: {targQuality}", fontsize=8, color='k')


    def segment_data(self, times, max_gap = 1/2):
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
            ax = fig.add_subplot(gs[0, 2*i:2*i + 2])
            ax.grid(True)
            axes.append(ax)
            ax = fig.add_subplot(gs[1, 2*i:2*i + 2])
            ax.grid(True)
            axes.append(ax)
            
        for i in range(2):  # Create the innovation plots
            ax = fig.add_subplot(gs[2, 2*i:2*i+2])
            ax.grid(True)
            axes.append(ax)
        
        # Create the track quality plot
        ax = fig.add_subplot(gs[2, 4:6])
        ax.grid(True)
        axes.append(ax)
        
        # Label plots with respective labels
        for i in range(6):
            axes[i].set_xlabel("Time [min]")
            axes[i].set_ylabel(f"Error in {state_labels[i]}")
        for i in range(2):
            axes[6 + i].set_xlabel("Time [min]")
            axes[6 + i].set_ylabel(f"Innovation in {meas_labels[i]}")
        axes[8].set_xlabel("Time [min]")
        axes[8].set_ylabel("Track Error [km]")
        
        return axes


    def save_plot(self, fig, plotEstimators, saveName, targ, sat, suffix):
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
        if plotEstimators:
            filePath = os.path.dirname(os.path.realpath(__file__))
            plotPath = os.path.join(filePath, 'plots')
            os.makedirs(plotPath, exist_ok=True)
            if saveName is None:
                plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_{suffix}.png"), dpi=300)
            else:
                plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_{suffix}.png"), dpi=300)
        plt.close()


### Plot communications sent/recieved  
    # Plot the total data sent and received by satellites
    def plot_global_comms(self, saveName):

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

                    for time in self.comms.total_comm_data[target_id][reciever][sender]:

                        # Get the data
                        data = self.comms.total_comm_data[target_id][reciever][sender][time]

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
            sent_data = dict(sorted(sent_data.items(), key=lambda item: satNames.index(item[0])))
            rec_data = dict(sorted(rec_data.items(), key=lambda item: satNames.index(item[0])))

            p1 = ax.bar(list(sent_data.keys()), list(sent_data.values()), bottom=list(prev_data.values()), color=color)

            # Add text labels to show which target is which.
            for i, v in enumerate(list(sent_data.values())):
                ax.text(i, list(prev_data.values())[i], targ.name, ha='center', va='bottom', color='black')

            # Add the sent_data values to the prev_data
            for key in sent_data.keys():
                prev_data[key] += sent_data[key]

            p2 = ax.bar(list(rec_data.keys()), list(rec_data.values()), bottom=list(prev_data.values()), color=color, fill=False, hatch='//', edgecolor=color)
            
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
            plt.savefig(os.path.join(plotPath, f"{saveName}_total_comms.png"), dpi=300)
        else:
            filePath = os.path.dirname(os.path.realpath(__file__))
            plotPath = os.path.join(filePath, 'plots')
            plt.savefig(os.path.join(plotPath, f"total_comms.png"), dpi=300)

    # Plots the actual data amount used by the satellites
    def plot_used_comms(self, saveName):

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

                    for time in self.comms.used_comm_data[target_id][reciever][sender]:

                        # Get the data
                        data = self.comms.used_comm_data[target_id][reciever][sender][time]

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
            sent_data = dict(sorted(sent_data.items(), key=lambda item: satNames.index(item[0])))
            rec_data = dict(sorted(rec_data.items(), key=lambda item: satNames.index(item[0])))

            p1 = ax.bar(list(sent_data.keys()), list(sent_data.values()), bottom=list(prev_data.values()), color=color)

            # Add text labels to show which target is which.
            for i, v in enumerate(list(sent_data.values())):
                ax.text(i, list(prev_data.values())[i], targ.name, ha='center', va='bottom', color='black')

            # Add the sent_data values to the prev_data
            for key in sent_data.keys():
                prev_data[key] += sent_data[key]

            p2 = ax.bar(list(rec_data.keys()), list(rec_data.values()), bottom=list(prev_data.values()), color=color, fill=False, hatch='//', edgecolor=color)

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
            plt.savefig(os.path.join(plotPath, f"{saveName}_used_comms.png"), dpi=300)
        else:
            filePath = os.path.dirname(os.path.realpath(__file__))
            plotPath = os.path.join(filePath, 'plots')
            plt.savefig(os.path.join(plotPath, f"used_comms.png"), dpi=300)

    # Sub plots for each satellite showing the track uncertainty for each target and then the comms sent/recieved about each target vs time
    def plot_local_comms(self, saveName):

        # For each satellite make a plot:
        for sat in self.sats:
            
            # Create the figure and subplot:
            fig = plt.figure(figsize=(15, 8))

            fig.suptitle(f"CI DDF, Track Uncertainty and Data Received by {sat.name}", fontsize=14)
            gs = gridspec.GridSpec(2, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            # Add a subplot title
            ax1.set_title(f"Track Uncertainty for {sat.name}")
            ax1.set_ylabel('Track Uncertainty [km]')

            ax2 = fig.add_subplot(gs[1, 0])
            # Add a subplot title
            ax2.set_title(f"Data Received by {sat.name}")
            ax2.set_ylabel('Data Sent/Recieved (# of numbers)')
            ax2.set_xlabel('Time [min]')

            # Now, at the bottom of the plot, add the legends
            handles = []
            for targ in self.targs:
                handles.append(Patch(color=targ.color, label=f"{targ.name}"))
            for tempSat in self.sats:
                handles.append(Patch(color=tempSat.color, label=f"{tempSat.name}"))

            # Create a legend
            fig.legend(handles=handles, loc='lower right', ncol=2, bbox_to_anchor=(1, 0))

            nonEmptyTime = []

            # Now do plots for the first subplot, we will be plotting track uncertainty for each target
            for targ in self.targs:

                # Get the uncertainty data
                trackUncertainty = sat.ciEstimator.trackErrorHist[targ.targetID]

                # Get the times for the track_uncertainty
                times = [time for time in trackUncertainty.keys()]
                # segments = self.segment_data(times, max_gap = 1/6)
                segments = self.segment_data(times)

                for segment in segments:
                    # Does the semgnet have a real data point in eveyr time step?
                    newTime = []
                    for time in segment:
                        if not not trackUncertainty[time]:
                            newTime.append(time)
                            nonEmptyTime.append(time)

                    trackVec = [trackUncertainty[time] for time in newTime]
                    ax1.plot(newTime, trackVec, color=targ.color, linewidth=1.5)

            # Now for each target make the dashed lines for the target quality
            for targ in self.targs:

                # Now plot a dashed line for the targetPriority
                ax1.axhline(y=targ.tqReq*50 + 50, color=targ.color, linestyle='dashed', linewidth=1.5)
                # Add a text label on the above right side of the dashed line
                ax1.text(min(nonEmptyTime), targ.tqReq*50 + 50 + 5, f"Target Quality: {targ.tqReq}", fontsize=8, color=targ.color)


            # Now do the 2nd subplot, bar plot showing the data sent/recieved by each satellite about each target

            # Save previous data, to stack the bars
            prevData = defaultdict(dict)

            nonEmptyTime = list(set(nonEmptyTime)) # also make it assending
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
                    ax2.bar(nonEmptyTime, values, bottom=list(prevData.values()), color=targ.color, hatch = '//', edgecolor = sender.color, linewidth = 0, width = min_diff)

                    # Add the values to the prevData
                    for key in data.keys():
                        prevData[key] += data[key]

            # Save the plot
            if saveName is not None:
                filePath = os.path.dirname(os.path.realpath(__file__))
                plotPath = os.path.join(filePath, 'plots')
                os.makedirs(plotPath, exist_ok=True)
                plt.savefig(os.path.join(plotPath, f"{saveName}_{sat.name}_track_uncertainty.png"), dpi=300)

            # plt.close(fig)

### Plot all messages
    def plot_messages(self, plotEstimators, saveName):
        """
        Plot all messages sent between satellites.

        Args:
            plotEstimators (bool): Flag indicating whether to save the plot.
            saveName (str): Name for the saved plot file.
        """
    
        # Plot the in-track and cross-track measurements
        for sat in self.sats:
            for neighbor in self.comms.G.neighbors(sat):
                for targetID in sat.etEstimator.measHist.keys():
                    fig = plt.figure(figsize=(15, 8))
                    gs = gridspec.GridSpec(2, 1)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[1, 0])
                    for time in sat.etEstimator.measHist[targetID][sat][neighbor].keys():
                        if len(sat.etEstimator.measHist[targetID][sat][neighbor][time]) != 0:
                            alpha, beta = sat.etEstimator.measHist[targetID][sat][neighbor][time]
                            if not np.isnan(alpha):
                                ax1.scatter(time, 1, color='r')
                            else:
                                ax1.scatter(time, 0, color='b')
                            
                            if not np.isnan(beta):
                                ax2.scatter(time, 1, color='r')
                            else:
                                ax2.scatter(time, 0, color='b')
                            
                fig.suptitle(f"Satellite Msgs from {sat.name} to {neighbor.name}", fontsize=14)

                # Label the plots
                ax1.set_xlabel("Time [min]")
                ax1.set_ylabel("In-Track Measurement")
                
                ax2.set_xlabel("Time [min]")
                ax2.set_ylabel("Cross-Track Measurement")
                
                # Create a patch for the legend
                handles = [
                    Patch(color='r', label=f'{sat.name} Explicit Measurements'),
                    Patch(color='b', label=f'{sat.name} Implicit Measurements'),
                ]
                        
                        # Create a legend   
                fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 0))
                plt.tight_layout()
                
                # Save the plot
                if plotEstimators:
                    filePath = os.path.dirname(os.path.realpath(__file__))
                    plotPath = os.path.join(filePath, 'plots')
                    os.makedirs(plotPath, exist_ok=True)
                    if saveName is None:
                        plt.savefig(os.path.join(plotPath, f"Targ{targetID}_{sat.name}_{neighbor.name}_et_msg.png"), dpi=300)
                    else:
                        plt.savefig(os.path.join(plotPath, f"{saveName}_Targ{targetID}_{sat.name}_{neighbor.name}_et_msg.png"), dpi=300)
                
    
### Plot 3D Gaussian Uncertainity Ellispoids ###
    def plot_all_uncertainty_ellipses(self):
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
                    for k in range(5):
                        if k != 4:
                            continue
                        if k == 0:  # Plot Local Uncertainty Ellipsoid
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            fig.suptitle(f"{targ.name}, {sat.name} Local Gaussian Uncertainty Ellipsoids")

                            satColor = sat.color
                            alpha = 0.3

                            for time in sat.indeptEstimator.estHist[targ.targetID].keys():
                                true_pos = targ.hist[time][[0, 2, 4]]
                                est_pos = np.array([sat.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                cov_matrix = sat.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                error = np.linalg.norm(true_pos - est_pos)
                                LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                                self.plot_ellipsoid(ax, est_pos, cov_matrix, color=satColor, alpha=alpha)
                                self.plot_estimate(ax, est_pos, true_pos, satColor)
                                self.plot_LOS(ax, est_pos, LOS_vec)
                                self.plot_labels(ax, targ, sat, time, error)
                                self.set_axis_limits(ax, est_pos, np.sqrt(eigenvalues), margin=50.0)

                                img = self.save_GEplot_to_image(fig)
                                self.imgs_local_GE[targ.targetID][sat].append(img)
                                ax.cla()  # Clear the plot for the next iteration
                            plt.close(fig)

                        if k == 1:  # Plot DDF Uncertainty Ellipsoid
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            fig.suptitle(f"{targ.name}, {sat.name} DDF Gaussian Uncertainty Ellipsoids")

                            satColor = '#DC143C'
                            alpha = 0.3

                            for time in sat.ciEstimator.estHist[targ.targetID].keys():
                                true_pos = targ.hist[time][[0, 2, 4]]
                                est_pos = np.array([sat.ciEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                cov_matrix = sat.ciEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                error = np.linalg.norm(true_pos - est_pos)
                                LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                                self.plot_ellipsoid(ax, est_pos, cov_matrix, color=satColor, alpha=alpha)
                                self.plot_estimate(ax, est_pos, true_pos, satColor)
                                self.plot_LOS(ax, est_pos, LOS_vec)
                                self.plot_labels(ax, targ, sat, time, error)
                                self.set_axis_limits(ax, est_pos, np.sqrt(eigenvalues), margin=50.0)

                                img = self.save_GEplot_to_image(fig)
                                self.imgs_ddf_GE[targ.targetID][sat].append(img)
                                ax.cla()  # Clear the plot for the next iteration
                            plt.close(fig)

                        if k == 2:  # Plot Central Uncertainty Ellipsoid
                            fig = plt.figure(figsize=(10, 8))
                            fig.suptitle(f"{targ.name}, {sat.name} Central Gaussian Uncertainty Ellipsoids")

                            ax = fig.add_subplot(111, projection='3d')
                            satColor = '#228B22'
                            alpha = 0.3

                            for time in self.centralEstimator.estHist[targ.targetID].keys():
                                true_pos = targ.hist[time][[0, 2, 4]]
                                est_pos = np.array([self.centralEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                cov_matrix = self.centralEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                error = np.linalg.norm(true_pos - est_pos)
                                LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                                self.plot_ellipsoid(ax, est_pos, cov_matrix, color=satColor, alpha=alpha)
                                self.plot_estimate(ax, est_pos, true_pos, satColor)
                                self.plot_LOS(ax, est_pos, LOS_vec)
                                self.plot_labels(ax, targ, sat, time, error)
                                self.set_axis_limits(ax, est_pos, np.sqrt(eigenvalues), margin=50.0)

                                img = self.save_GEplot_to_image(fig)
                                self.imgs_cent_GE[targ.targetID][sat].append(img)
                                ax.cla()  # Clear the plot for the next iteration
                            plt.close(fig)

                        if k == 3:  # Plot Stereo Uncertainty Ellipsoid
                            for sat2 in self.sats:
                                if sat2 != sat:
                                    if targ.targetID in sat2.targetIDs:
                                        fig = plt.figure(figsize=(10, 8))
                                        fig.suptitle(f"{targ.name}, {sat.name} and {sat2.name} Gaussian Uncertainty Ellipsoids")

                                        ax = fig.add_subplot(111, projection='3d')
                                        sat1Color = sat.color
                                        sat2Color = sat2.color
                                        ddfColor = '#DC143C'
                                        alpha = 0.3

                                        sat1_times = sat.indeptEstimator.estHist[targ.targetID].keys()
                                        sat2_times = sat2.indeptEstimator.estHist[targ.targetID].keys()
                                        times = [time for time in sat1_times if time in sat2_times]

                                        for time in times:
                                            true_pos = targ.hist[time][[0, 2, 4]]
                                            est_pos1 = np.array([sat.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                            est_pos2 = np.array([sat2.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                            ddf_pos = np.array([sat.ciEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])

                                            cov_matrix1 = sat.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            cov_matrix2 = sat2.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            ddf_cov = sat.ciEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]

                                            eigenvalues1, eigenvectors1 = np.linalg.eigh(cov_matrix1)
                                            eigenvalues2, eigenvectors2 = np.linalg.eigh(cov_matrix2)
                                            ddf_eigenvalues, ddf_eigenvectors = np.linalg.eigh(ddf_cov)

                                            error1 = np.linalg.norm(true_pos - est_pos1)
                                            error2 = np.linalg.norm(true_pos - est_pos2)
                                            ddf_error = np.linalg.norm(true_pos - ddf_pos)

                                            LOS_vec1 = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])
                                            LOS_vec2 = -sat2.orbitHist[time] / np.linalg.norm(sat2.orbitHist[time])

                                            self.plot_ellipsoid(ax, est_pos1, cov_matrix1, color=sat1Color, alpha=alpha)
                                            self.plot_ellipsoid(ax, est_pos2, cov_matrix2, color=sat2Color, alpha=alpha)
                                            self.plot_ellipsoid(ax, ddf_pos, ddf_cov, color=ddfColor, alpha=alpha)

                                            self.plot_estimate(ax, est_pos1, true_pos, sat1Color)
                                            self.plot_estimate(ax, est_pos2, true_pos, sat2Color)
                                            self.plot_estimate(ax, ddf_pos, true_pos, ddfColor)

                                            self.plot_LOS(ax, est_pos1, LOS_vec1)
                                            self.plot_LOS(ax, est_pos2, LOS_vec2)

                                            self.plot_all_labels(ax, targ, sat, sat2, time, error1, error2, ddf_error)

                                            if np.max(eigenvalues1) > np.max(eigenvalues2):
                                                self.set_axis_limits(ax, est_pos1, np.sqrt(eigenvalues1), margin=50.0)
                                            else:
                                                self.set_axis_limits(ax, est_pos2, np.sqrt(eigenvalues2), margin=50.0)

                                            img = self.save_GEplot_to_image(fig)
                                            self.imgs_stereo_GE[targ.targetID][sat].append(img)
                                            ax.cla()  # Clear the plot for the next iteration
                                        plt.close(fig)
                        if k == 4: # plot stereo with et
                            for sat2 in self.sats:
                                if sat2 != sat:
                                    if targ.targetID in sat2.targetIDs:
                                        fig = plt.figure(figsize=(10, 8))
                                        fig.suptitle(f"{targ.name}, {sat.name} and {sat2.name} ET Gaussian Uncertainty Ellipsoids")

                                        ax = fig.add_subplot(111, projection='3d')
                                        sat1Color = sat.color
                                        sat2Color = sat2.color
                                        etColor = '#DC143C'
                                        alpha = 0.3

                                        sat1_times = sat.indeptEstimator.estHist[targ.targetID].keys()
                                        sat2_times = sat2.indeptEstimator.estHist[targ.targetID].keys()
                                        et_times = sat.etEstimator.estHist[targ.targetID][sat][sat].keys()
                                        times = [time for time in sat1_times if time in sat2_times and time in et_times]

                                        for time in times:
                                            true_pos = targ.hist[time][[0, 2, 4]]
                                            est_pos1 = np.array([sat.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                            est_pos2 = np.array([sat2.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                            et_pos = np.array([sat.etEstimator.estHist[targ.targetID][sat][sat][time][i] for i in [0, 2, 4]])

                                            cov_matrix1 = sat.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            cov_matrix2 = sat2.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            et_cov = sat.etEstimator.covarianceHist[targ.targetID][sat][sat][time][np.array([0, 2, 4])][:, np.array([0, 2, 4])]                                            
                                            eigenvalues1, eigenvectors1 = np.linalg.eigh(cov_matrix1)
                                            eigenvalues2, eigenvectors2 = np.linalg.eigh(cov_matrix2)
                                            et_eigenvalues, et_eigenvectors = np.linalg.eigh(et_cov)
                                            
                                            error1 = np.linalg.norm(true_pos - est_pos1)
                                            error2 = np.linalg.norm(true_pos - est_pos2)
                                            et_error = np.linalg.norm(true_pos - et_pos)
                                            
                                            LOS_vec1 = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])
                                            LOS_vec2 = -sat2.orbitHist[time] / np.linalg.norm(sat2.orbitHist[time])
                                            
                                            self.plot_ellipsoid(ax, est_pos1, cov_matrix1, color=sat1Color, alpha=alpha)
                                            self.plot_ellipsoid(ax, est_pos2, cov_matrix2, color=sat2Color, alpha=alpha)
                                            self.plot_ellipsoid(ax, et_pos, et_cov, color=etColor, alpha=alpha)
                                            
                                            self.plot_estimate(ax, est_pos1, true_pos, sat1Color)
                                            self.plot_estimate(ax, est_pos2, true_pos, sat2Color)
                                            self.plot_estimate(ax, et_pos, true_pos, etColor)
                                            
                                            self.plot_LOS(ax, est_pos1, LOS_vec1)
                                            self.plot_LOS(ax, est_pos2, LOS_vec2)
                                            
                                            self.plot_all_labels(ax, targ, sat, sat2, time, error1, error2, et_error)
                                            
                                            if np.max(eigenvalues1) > np.max(eigenvalues2):
                                                self.set_axis_limits(ax, est_pos1, np.sqrt(eigenvalues1), margin=50.0)
                                            else:
                                                self.set_axis_limits(ax, est_pos2, np.sqrt(eigenvalues2), margin=50.0)
                                                
                                            img = self.save_GEplot_to_image(fig)
                                            self.imgs_stereo_ET_GE[targ.targetID][sat].append(img)
                                            ax.cla()  # Clear the plot for the next iteration
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
        ax.plot_surface(x_transformed, y_transformed, z_transformed, color=color, alpha=alpha)


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
        ax.quiver(est_pos[0], est_pos[1], est_pos[2], LOS_vec[0], LOS_vec[1], LOS_vec[2], color='k', length=10, normalize=True)


    def plot_labels(self, ax, targ, sat, time, err):
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
        ax.text2D(0.05, 0.95, f"Time: {time:.2f}, {targ.name}, Error {sat.name}: {err:.2f} [km]", transform=ax.transAxes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=30)


    def plot_all_labels(self, ax, targ, sat1, sat2, time, err1, err2, ddf_err):
        """
        Plots labels for multiple satellites on the given axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        - targ: The target object.
        - sat1: The first satellite object.
        - sat2: The second satellite object.
        - time: The current time.
        - err1: The error value for the first satellite.
        - err2: The error value for the second satellite.
        - ddf_err: The error value for the DDF.

        Returns:
        None
        """
        ax.text2D(0.05, 0.95, f"Time: {time:.2f}, {targ.name}, Error {sat1.name}: {err1:.2f} [km], Error {sat2.name}: {err2:.2f} [km], Error DDF: {ddf_err:.2f} [km]", transform=ax.transAxes)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=30)


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
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        return img
      

    def render_gif(self, fileType, saveName, filePath=os.path.dirname(os.path.realpath(__file__)), fps=1):
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
            for sat in self.sats:
                for targ in self.targs:
                    if targ.targetID in sat.targetIDs:
                        file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_indept_GE.gif")
                        with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_local_GE[targ.targetID][sat]:
                                writer.append_data(img)

                        ddf_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_ddf_GE.gif")
                        with imageio.get_writer(ddf_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_ddf_GE[targ.targetID][sat]:
                                writer.append_data(img)

                        cent_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_central_GE.gif")
                        with imageio.get_writer(cent_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_cent_GE[targ.targetID][sat]:
                                writer.append_data(img)

                        both_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_stereo_GE.gif")
                        with imageio.get_writer(both_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_stereo_GE[targ.targetID][sat]:
                                writer.append_data(img)
                                
                        et_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_stereo_ET_GE.gif")
                        with imageio.get_writer(et_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_stereo_ET_GE[targ.targetID][sat]:
                                writer.append_data(img)
  
### Data Dump File ###        
    def log_data(self, time_vec, saveName, filePath=os.path.dirname(os.path.realpath(__file__))):
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
                    innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]

                    ddf_times = sat.ciEstimator.estHist[targ.targetID].keys()
                    ddf_estHist = sat.ciEstimator.estHist[targ.targetID]
                    ddf_covHist = sat.ciEstimator.covarianceHist[targ.targetID]
                    ddf_trackError = sat.ciEstimator.trackErrorHist[targ.targetID]

                    ddf_innovation_times = sat.ciEstimator.innovationHist[targ.targetID].keys()
                    ddf_innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
                    ddf_innovationCovHist = sat.ciEstimator.innovationCovHist[targ.targetID]
                    
                    et_times = sat.etEstimator.estHist[targ.targetID][sat][sat].keys()
                    et_estHist = sat.etEstimator.estHist[targ.targetID][sat][sat]
                    et_covHist = sat.etEstimator.covarianceHist[targ.targetID][sat][sat]
                    
                    for neighbor in self.comms.G.neighbors(sat):
                        if neighbor != sat:
                            et_measHist = sat.etEstimator.measHist[targ.targetID][sat][neighbor]
                  #et_trackError = self.etEstimator.trackErrorHist[targ.targetID][sat][sat]                  
                    

                    # File Name
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}.csv"

                    # Format the data and write it to the file
                    self.format_data(
                        filename, targ.targetID, times, sat_hist, trueHist,
                        sat_measHistTimes, sat_measHist, estTimes, estHist, covHist,
                        trackError, innovationHist, innovationCovHist, ddf_times,
                        ddf_estHist, ddf_covHist, ddf_trackError, ddf_innovation_times,
                        ddf_innovationHist, ddf_innovationCovHist, et_times, et_estHist, et_covHist,
                        et_measHist
                    )


    def format_data(
        self, filename, targetID, times, sat_hist, trueHist, sat_measHistTimes,
        sat_measHist, estTimes, estHist, covHist, trackError, innovationHist,
        innovationCovHist, ddf_times, ddf_estHist, ddf_covHist, ddf_trackError,
        ddf_innovation_times, ddf_innovationHist, ddf_innovationCovHist, et_times,
        et_estHist, et_covHist, et_measHist
    ):
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
        - ddf_times (dict_keys): DDF estimation times.
        - ddf_estHist (dict): DDF estimation history.
        - ddf_covHist (dict): DDF covariance history.
        - ddf_trackError (dict): DDF track quality history.
        - ddf_innovation_times (dict_keys): DDF innovation times.
        - ddf_innovationHist (dict): DDF innovation history.
        - ddf_innovationCovHist (dict): DDF innovation covariance history.
        - et_times (dict_keys): ET estimation times.
        - et_estHist (dict): ET estimation history.
        - et_covHist (dict): ET covariance history.
        - et_measHist (dict): ET measurement history.

        Returns:
        None
        """
        precision = 3  # Set the desired precision

        def format_list(lst):
            if isinstance(lst, np.ndarray):
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst.flatten()]
            elif isinstance(lst, int) or isinstance(lst, float):
                return [f"{float(lst):.{precision}f}" if not np.isnan(lst) else "nan"]
            else:
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst]

        # Create a single CSV file for the target
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            # Writing headers
            writer.writerow([
                'Time', 'x_sat', 'y_sat', 'z_sat',
                'True_x', 'True_vx', 'True_y', 'True_vy', 'True_z', 'True_vz',
                'InTrackAngle', 'CrossTrackAngle', 'Est_x', 'Est_vx', 'Est_y', 'Est_vy', 'Est_z', 'Est_vz',
                'Cov_xx', 'Cov_vxvx', 'Cov_yy', 'Cov_vyvy', 'Cov_zz', 'Cov_vzvz', 'Track Error',
                'Innovation_ITA', 'Innovation_CTA', 'InnovationCov_ITA', 'InnovationCov_CTA',
                'DDF_Est_x', 'DDF_Est_vx', 'DDF_Est_y', 'DDF_Est_vy', 'DDF_Est_z', 'DDF_Est_vz',
                'DDF_Cov_xx', 'DDF_Cov_vxvx', 'DDF_Cov_yy', 'DDF_Cov_vyvy', 'DDF_Cov_zz', 'DDF_Cov_vzvz', 'DDF Track Error',
                'DDF_Innovation_ITA', 'DDF_Innovation_CTA', 'DDF_InnovationCov_ITA', 'DDF_InnovationCov_CTA', 'ET_Est_x', 'ET_Est_vx',
                'ET_Est_y', 'ET_Est_vy', 'ET_Est_z', 'ET_Est_vz', 'ET_Cov_xx', 'ET_Cov_vxvx', 'ET_Cov_yy', 'ET_Cov_vyvy', 'ET_Cov_zz',
                'ET_Cov_vzvz', 'ET_Meas_alpha', 'ET_Meas_beta'
            ])

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

                if time in ddf_times:
                    row += format_list(ddf_estHist[time])
                    row += format_list(np.diag(ddf_covHist[time]))
                    row += format_list(ddf_trackError[time])

                if time in ddf_innovation_times:
                    row += format_list(ddf_innovationHist[time])
                    row += format_list(np.diag(ddf_innovationCovHist[time]))
                    
                if time in et_times:
                    row += format_list(et_estHist[time])
                    row += format_list(np.diag(et_covHist[time]))
                    row += format_list(et_measHist[time])

                writer.writerow(row)
                
                
    def collectNISNEESData(self):
        """
        Collects NEES and NIS data for the simulation in an easy-to-read format.

        Returns:
        dict: A dictionary containing NEES and NIS data for each targetID and satellite.
        """
        # Create a dictionary of targetIDs
        data = {targetID: defaultdict(dict) for targetID in (targ.targetID for targ in self.targs)}

        # Now for each targetID, make a dictionary for each satellite
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Extract the data
                    data[targ.targetID][sat.name] = {
                        'NEES': sat.indeptEstimator.neesHist[targ.targetID],
                        'NIS': sat.indeptEstimator.nisHist[targ.targetID]
                    }
                    # Also save the DDF data
                    data[targ.targetID][f"{sat.name} DDF"] = {
                        'NEES': sat.ciEstimator.neesHist[targ.targetID],
                        'NIS': sat.ciEstimator.nisHist[targ.targetID]
                    }

            # If central estimator is used, also add that data
            if self.centralEstimator:
                if targ.targetID in self.centralEstimator.neesHist:
                    data[targ.targetID]['Central'] = {
                        'NEES': self.centralEstimator.neesHist[targ.targetID],
                        'NIS': self.centralEstimator.nisHist[targ.targetID]
                    }

        return data
    

