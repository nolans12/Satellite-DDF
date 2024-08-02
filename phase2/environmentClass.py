from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment:
    def __init__(self, sats, targs, comms, centralEstimator=None):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """
        # Use the provided central estimator or default to None
        self.centralEstimator = centralEstimator if centralEstimator else None
        
        # Define the satellites
        self.sats = sats
        
        # Define the targets
        self.targs = targs
        
        # Define the communication network
        self.comms = comms
        
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


    def simulate(self, time_vec, pause_step=0.1, savePlot=False, saveGif=False, saveData=False, saveName=None, showSim=False):
        """
        Simulate the environment over a time range.
        
        Args:
        - time_vec: numpy array of time steps with poliastro units associated.
        - pause_step: time to pause between each step if displaying as animation (default: 0.1).
        - savePlot: boolean, if True will save plots (default: False).
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

            if savePlot:
                # Update the plot environment
                self.plot()

                if showSim:
                    plt.pause(pause_step)
                    plt.draw()

        if savePlot:
            # Plot the results of the simulation.
            self.plot_estimator_results(time_vec, savePlot=savePlot, saveName=saveName) # marginal error, innovation, and NIS/NEES plots
           
        if saveGif:
            # Save the uncertainty ellipse plots
            self.plot_all_uncertainty_ellipses() # Uncertainty Ellipse Plots

        # Log the Data
        if saveData:
            self.log_data(time_vec, saveName=saveName)

        return self.collectNISNEESData()  # Return the data collected during the simulation
    
    
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


    def data_fusion(self):
        """
        Perform data fusion by collecting measurements, performing central fusion, sending estimates, and performing covariance intersection.
        """
        # Collect all measurements
        collectedFlag, measurements = self.collect_all_measurements()

        # If a central estimator is present, perform central fusion
        self.central_fusion(collectedFlag, measurements)

        # Now send estimates for future CI
        self.send_estimates()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            sat.ddfEstimator.CI(sat, self.comms.G.nodes[sat])


    def send_estimates(self):
        """
        Send the most recent estimates from each satellite to its neighbors.
        """
        # Loop through all satellites
        for sat in self.sats:
            # For each targetID in the satellite estimate history
            for targetID in sat.ddfEstimator.estHist.keys():
                # Skip if there are no estimates for this targetID
                if len(sat.ddfEstimator.estHist[targetID].keys()) == 0:
                    continue

                # This means satellite has an estimate for this target, now send it to neighbors
                for neighbor in self.comms.G.neighbors(sat):
                    # Get the most recent estimate time
                    satTime = max(sat.ddfEstimator.estHist[targetID].keys())

                    # Send most recent estimate to neighbor
                    self.comms.send_estimate(
                        sat, 
                        neighbor, 
                        sat.ddfEstimator.estHist[targetID][satTime], 
                        sat.ddfEstimator.covarianceHist[targetID][satTime], 
                        targetID, 
                        satTime
                    )


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
        # Create a mask to filter points to be within plot limits
        mask = ((self.x_earth >= self.ax.get_xlim()[0]) & (self.x_earth <= self.ax.get_xlim()[1]) &
                (self.y_earth >= self.ax.get_ylim()[0]) & (self.y_earth <= self.ax.get_ylim()[1]) &
                (self.z_earth >= self.ax.get_zlim()[0]) & (self.z_earth <= self.ax.get_zlim()[1]))
        
        # x_earth_clipped = np.ma.masked_where(~mask, self.x_earth)
        # y_earth_clipped = np.ma.masked_where(~mask, self.y_earth)
        # z_earth_clipped = np.ma.masked_where(~mask, self.z_earth)
        x_earth_clipped = self.x_earth
        y_earth_clipped = self.y_earth
        z_earth_clipped = self.z_earth

        self.ax.plot_surface(x_earth_clipped, y_earth_clipped, z_earth_clipped, color='k', alpha=0.1)
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
    def plot_estimator_results(self, time_vec, savePlot, saveName):
        """
        Create three types of plots: Local vs Central, DDF vs Central, and Local vs DDF vs Central.

        Args:
            time_vec (list): List of time values.
            savePlot (bool): Flag indicating whether to save the plot.
            saveName (str): Name for the saved plot file.
        """
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/min]', 'Y [km]', 'Vy [km/min]', 'Z [km]', 'Vz [km/min]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]', 'Track Error [km]']

        # For each target and satellite, plot the estimation error, innovation, and track quality
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    for k in range(3):

                        if k != 2:
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
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, times, times, times, times, estHist, trueHist, covHist, 
                                                    innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, 
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
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, '#228B22', linewidth=1.5, c=True)

                            # Create a Patch for Legend
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                            
                        elif k == 1:  # DDF vs Central
                            # Get the Data
                            ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ddfEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ddfEstimator.neesHist[targ.targetID]
                            ddf_trackErrorHist = sat.ddfEstimator.trackErrorHist[targ.targetID]

                            # Get the valid times for data
                            ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
                            ddf_trackError_times = [time for time in time_vec.value if time in ddf_trackErrorHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, ddf_times, ddf_innovation_times, ddf_NISNEES_times, 
                                                    ddf_trackError_times, ddf_estHist, trueHist, ddf_covHist, 
                                                    ddf_innovationHist, ddf_innovationCovHist, ddf_NISHist, ddf_NEESHist, 
                                                    ddf_trackErrorHist, '#DC143C', linewidth=2.5)

                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the Data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times for data
                                times = [time for time in time_vec.value if time in estHist]
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, '#228B22', linewidth=1.5, c=True)
                            
                            # Create Patch for Legend
                            handles = [
                                Patch(color='#DC143C', label=f'{sat.name} DDF Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                            
                        elif k == 2:  # Local vs DDF vs Central
                            # Get the Local Data
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]
                            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]
                            
                            # Get the DDF Data
                            ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ddfEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ddfEstimator.neesHist[targ.targetID]
                            ddf_trackErrorHist = sat.ddfEstimator.trackErrorHist[targ.targetID]
                            
                            # Get the valid times for data
                            times = [time for time in time_vec.value if time in estHist]
                            ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
                            ddf_trackError_times = [time for time in time_vec.value if time in ddf_trackErrorHist]
                            
                            # Plot the 3x3 Grid of Data
                            self.plot_estimator_data(fig, axes, times, times, times, times, estHist, trueHist, covHist, 
                                                    innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, 
                                                    satColor, linewidth=2.5)
                            self.plot_estimator_data(fig, axes, ddf_times, ddf_innovation_times, ddf_NISNEES_times, 
                                                    ddf_trackError_times, ddf_estHist, trueHist, ddf_covHist, 
                                                    ddf_innovationHist, ddf_innovationCovHist, ddf_NISHist, ddf_NEESHist, 
                                                    ddf_trackErrorHist, '#DC143C', linewidth=2.0)

                            if self.centralEstimator:  # If central estimator is used, plot the data
                                # Get the data
                                estHist = self.centralEstimator.estHist[targ.targetID]
                                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
                                
                                # Get the valid times
                                times = [time for time in time_vec.value if time in estHist]
                                
                                # Plot the 3x3 Grid of Data
                                self.plot_estimator_data(fig, axes, times, [], [], times, estHist, trueHist, covHist, 
                                                        innovationHist, innovationCovHist, NISHist, NEESHist, 
                                                        trackErrorHist, '#228B22', linewidth=1.5, c=True)

                            # Create a patch for the legend
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color='#DC143C', label=f'{sat.name} DDF Estimator'),
                                Patch(color='#228B22', label=f'Central Estimator')
                            ]
                        
                        # Create a legend   
                        fig.legend(handles=handles, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                        plt.tight_layout()
                        
                        # Save the Plot with respective suffix
                        # suffix = ['indept', 'ddf', 'both'][k]
                        suffix = ['','',''][k]
                        self.save_plot(fig, savePlot, saveName, targ, sat, suffix)


    def plot_estimator_data(self, fig, axes, estTimes, innTimes, nnTimes, tqTimes, estHist, trueHist, covHist, innovationHist, innovationCovHist, NISHist, NEESHist, trackErrorHist, label_color, linewidth, c=False):
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
        if not c:  # Not central estimator so plot everything
            self.plot_errors(axes, estTimes, estHist, trueHist, covHist, label_color, linewidth)
            self.plot_innovations(axes, innTimes, innovationHist, innovationCovHist, label_color, linewidth)
            self.plot_track_quality(axes, tqTimes, trackErrorHist, label_color, linewidth)
        else:  # Central estimator doesn't have innovation data
            self.plot_errors(axes, estTimes, estHist, trueHist, covHist, label_color, linewidth)
            self.plot_track_quality(axes, tqTimes, trackErrorHist, label_color, linewidth)

            
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

    def plot_track_quality(self, ax, times, trackErrorHist, label_color, linewidth):
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
            segments = self.segment_data(times)
        
            for segment in segments:
                track_quality = [trackErrorHist[time] for time in segment]
                ax[8].plot(segment, track_quality, color=label_color, linewidth=linewidth)

    def segment_data(self, times, max_gap = 30):
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

    def save_plot(self, fig, savePlot, saveName, targ, sat, suffix):
        """
        Save each plot into the "plots" folder with the given suffix.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            savePlot (bool): Flag indicating whether to save the plot.
            saveName (str): Name for the saved plot file.
            targ (Target): Target object.
            sat (Satellite): Satellite object.
            suffix (str): Suffix for the saved plot file name.
        """
        if savePlot:
            filePath = os.path.dirname(os.path.realpath(__file__))
            plotPath = os.path.join(filePath, 'plots')
            os.makedirs(plotPath, exist_ok=True)
            if saveName is None:
                plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_{suffix}.png"), dpi=300)
            else:
                plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_{suffix}.png"), dpi=300)
        plt.close()

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
                    for k in range(4):
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

                            for time in sat.ddfEstimator.estHist[targ.targetID].keys():
                                true_pos = targ.hist[time][[0, 2, 4]]
                                est_pos = np.array([sat.ddfEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                cov_matrix = sat.ddfEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
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
                                            ddf_pos = np.array([sat.ddfEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])

                                            cov_matrix1 = sat.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            cov_matrix2 = sat2.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                            ddf_cov = sat.ddfEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]

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
      

    def render_gif(self, fileType, saveName, filePath=os.path.dirname(os.path.realpath(__file__)), fps=10):
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

                    ddf_times = sat.ddfEstimator.estHist[targ.targetID].keys()
                    ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                    ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                    ddf_trackError = sat.ddfEstimator.trackErrorHist[targ.targetID]

                    ddf_innovation_times = sat.ddfEstimator.innovationHist[targ.targetID].keys()
                    ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                    ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]

                    # File Name
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}.csv"

                    # Format the data and write it to the file
                    self.format_data(
                        filename, targ.targetID, times, sat_hist, trueHist,
                        sat_measHistTimes, sat_measHist, estTimes, estHist, covHist,
                        trackError, innovationHist, innovationCovHist, ddf_times,
                        ddf_estHist, ddf_covHist, ddf_trackError, ddf_innovation_times,
                        ddf_innovationHist, ddf_innovationCovHist
                    )


    def format_data(
        self, filename, targetID, times, sat_hist, trueHist, sat_measHistTimes,
        sat_measHist, estTimes, estHist, covHist, trackError, innovationHist,
        innovationCovHist, ddf_times, ddf_estHist, ddf_covHist, ddf_trackError,
        ddf_innovation_times, ddf_innovationHist, ddf_innovationCovHist
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

        Returns:
        None
        """
        precision = 3  # Set the desired precision

        def format_list(lst):
            if isinstance(lst, np.ndarray):
                return [f"{x:.{precision}f}" for x in lst.flatten()]
            else:
                return [f"{x:.{precision}f}" for x in lst]

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
                'DDF_Innovation_ITA', 'DDF_Innovation_CTA', 'DDF_InnovationCov_ITA', 'DDF_InnovationCov_CTA'
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
                    row += format_list(trackError)
                    row += format_list(innovationHist[time])
                    row += format_list(np.diag(innovationCovHist[time]))

                if time in ddf_times:
                    row += format_list(ddf_estHist[time])
                    row += format_list(np.diag(ddf_covHist[time]))
                    row += format_list(ddf_trackError)

                if time in ddf_innovation_times:
                    row += format_list(ddf_innovationHist[time])
                    row += format_list(np.diag(ddf_innovationCovHist[time]))

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
                        'NEES': sat.ddfEstimator.neesHist[targ.targetID],
                        'NIS': sat.ddfEstimator.nisHist[targ.targetID]
                    }

            # If central estimator is used, also add that data
            if self.centralEstimator:
                if targ.targetID in self.centralEstimator.neesHist:
                    data[targ.targetID]['Central'] = {
                        'NEES': self.centralEstimator.neesHist[targ.targetID],
                        'NIS': self.centralEstimator.nisHist[targ.targetID]
                    }

        return data
