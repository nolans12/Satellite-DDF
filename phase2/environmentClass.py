from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment:
    def __init__(self, sats, targs, comms, centralEstimator=None, ciEstimator=None, etEstimator=None):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """
        # Use the provided central estimator or default to None
        self.centralEstimator = centralEstimator
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
        
        # # If you want to do standard case:
        # self.ax.set_xlim([2000, 8000])
        # self.ax.set_ylim([-6000, 6000])
        # self.ax.set_zlim([2000, 8000])
        # self.ax.view_init(elev=30, azim=0)

        # If you want to do MonoTrack case:
        self.ax.set_xlim([-15000, 15000])
        self.ax.set_ylim([-15000, 15000])
        self.ax.set_zlim([-15000, 15000])
        self.ax.view_init(elev=30, azim=0)
        
        # auto scale the axis to be equal
        #self.ax.set_box_aspect([0.5, 1, 0.5])
        
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
        
        # Nested Dictionary for storing stereo estimation plots
        self.imgs_stereo_GE = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.imgs_dyn_comms = []
        self.tempCount = 0


    def simulate(self, time_vec, pause_step=0.1, showSim=False, savePlot=False, saveGif=False, saveData=False, saveComms = False, plot_dynamic_comms = False, saveName=None):
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

            self.plot()
            
            if plot_dynamic_comms:
                self.plot_dynamic_comms()

            if showSim:
                plt.pause(pause_step)
                plt.draw()
                
        print("Simulation Complete")
        
        if savePlot:
            # Plot the results of the simulation.
            self.plot_estimator_results(time_vec, savePlot=savePlot, saveName=saveName) # marginal error, innovation, and NIS/NEES plots

        if saveComms:
            # Plot the comm results
            self.plot_global_comms(saveName=saveName)
            self.plot_used_comms(saveName=saveName)
           
        if saveGif:
            # Save the uncertainty ellipse plots
            self.plot_all_uncertainty_ellipses(time_vec) # Uncertainty Ellipse Plots

        # Log the Data
        if saveData:
            self.log_data(time_vec, saveName=saveName)
            self.log_comms_data(time_vec, saveName=saveName)

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
            self.send_estimates()

        # Now send measurements for future ET
        if self.etEstimator:
            self.send_measurements()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            if self.ciEstimator:
                sat.ciEstimator.CI(sat, self.comms)
            if self.etEstimator:
                sat.etEstimator.event_trigger_processing(sat, self.time.to_value(), self.comms)
                #sat.etEstimator.event_triggered_fusion(sat, self.time.to_value(), self.comms)

        for sat in self.sats:
            if self.etEstimator:
                sat.etEstimator.event_trigger_updating(sat, self.time.to_value(), self.comms)


    def send_estimates(self):
        """
        Send the most recent estimates from each satellite to its neighbors.
        """
        # Loop through all satellites
        for sat in self.sats:
            # For each targetID in the satellite estimate history
            for targetID in sat.ciEstimator.estHist.keys():
                # Skip if there are no estimates for this targetID
                if isinstance(sat.measurementHist[targetID][self.time.to_value()], np.ndarray):#len(sat.ciEstimator.estHist[targetID].keys()) == 0:  

                    # This means satellite has an estimate for this target, now send it to neighbors
                    for neighbor in self.comms.G.neighbors(sat):
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
            for target in self.targs: # TODO: iniitalize with senders est and cov + noise? 
                if target.targetID in sat.targetIDs:
                    targetID = target.targetID
                    envTime = self.time.to_value()
                    # Skip if there are no measurements for this targetID
                    if isinstance(sat.measurementHist[target.targetID][envTime], np.ndarray):#len(sat.measurementHist[targetID][self.time.value]) == 0: ## TODO: empty lists have length of 1 i am pretty sure or you get a key error
                                            #if isinstance(sat.measurementHist[target.targetID][envTime], np.ndarray): # if the satellite took a measurement at this time

                        # This means satellite has a measurement for this target, now send it to neighbors
                        for neighbor in self.comms.G.neighbors(sat):
                            # Get the most recent measurement time
                            satTime = max(sat.measurementHist[targetID].keys()) #  this should be irrelevant and equal to  self.time since a measurement is sent on same timestep
                            
                            # Create implicit and explicit measurements vector for this neighbor
                            meas = sat.etEstimator.event_trigger(sat, neighbor, targetID, satTime)
                            
                            # Send that to neightbor
                            self.comms.send_measurements(
                                sat, 
                                neighbor, 
                                meas, 
                                targetID, 
                                satTime
                            )
                            
                            if not sat.etEstimator.estHist[target.targetID][sat][neighbor]: # if this is first meas, initialize common EKF 
                                sat.etEstimator.initialize_filter(sat, target, envTime, sharewith=neighbor)
                            
                            if not neighbor.etEstimator.estHist[target.targetID][neighbor][neighbor]: # if this is first meas, initialize sat2 local EKF
                                neighbor.etEstimator.initialize_filter(neighbor, target, envTime, sharewith=neighbor)
                                
                            if not neighbor.etEstimator.estHist[target.targetID][neighbor][sat]: # if this is first meas, initialize sat21 common EKF
                                neighbor.etEstimator.initialize_filter(neighbor, target, envTime, sharewith=sat)
                            
                            
                            # If a satellite wants to send measurements, it needs to have synchronized common filters
                            sat.etEstimator.synchronizeFlag[targetID][sat][neighbor][envTime] = True                            
                            neighbor.etEstimator.synchronizeFlag[targetID][neighbor][sat][envTime] = True
                            
                            # Search backwards through dictionary to check if there are 5 measurements sent to this neighbor
                            count = 5
                            for lastTime in reversed(list(sat.measurementHist[target.targetID].keys())): # starting now, go back in time
                                if isinstance(sat.measurementHist[target.targetID][lastTime], np.ndarray): # if the satellite took a measurement at this time
                                    count -= 1 # increment count
                                    
                                if count == 0: # if there are 5 measurements sent to this neighbor, no need to synchronize
                                    sat.etEstimator.synchronizeFlag[targetID][sat][neighbor][envTime] = False # set synchronize flag to true
                                    neighbor.etEstimator.synchronizeFlag[targetID][neighbor][sat][envTime] = False
                                    break # break out of loop
                                

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
        
        # Uncomment this for a POV sat1 viewing angle for mono-track case
        #self.calcViewingAngle()

    
    def calcViewingAngle(self):
        '''
        Calculate the viewing angle for the 3D plot in MonoTrack Case
        '''
        monoTarg = self.targs[0]
        x, y, z = monoTarg.pos
        range = jnp.sqrt(x**2 + y**2 + z**2)
        
        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x) * 180 / jnp.pi
        
        self.ax.view_init(elev=30, azim=azimuth)
    
        
    def save_envPlot_to_imgs(self):
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        self.imgs.append(img)


    def plot_estimator_results(self, time_vec, savePlot, saveName):
        """
        For each satellite and target plot the time history for all onboard kalman filters
        """
        
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/min]', 'Y [km]', 'Vy [km/min]', 'Z [km]', 'Vz [km/min]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]', 'Track Uncertainity [km]']        
        suffix_vec = ['local', 'ci', 'et', 'et_vs_ddf', 'et_pairwise']
        title_vec = ['Local vs Central', 'CI vs Central', 'ET vs Central', 'ET vs CI']
        title_vec = [title + " Estimator Results" for title in title_vec]
        
        # For Each Target
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:                                        
                    # Set up colors
                    satColor = sat.color
                    ddfColor = '#DC143C' # Crimson
                    centralColor = '#070400'  #'#228B22' # Forest Green
                    
                    trueHist = targ.hist
                    for plotNum in range(4):
                        
                        fig = plt.figure(figsize=(15, 8))
                        fig.suptitle(f"{targ.name}, {sat.name} {title_vec[plotNum]}", fontsize=14)
                        axes = self.setup_axes(fig, state_labels, meas_labels, targ.tqReq)
                        
                        if plotNum == 0:
                            # First Plot: Local vs Central
                            times, estHist, covHist, innovationHist, innovationCovHist, trackErrorHist = self.getEstimationHistory(sat, targ, time_vec,'local')
                            central_times, central_estHist, central_covHist, central_innovationHist, central_innovationCovHist, central_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'central')
                            
                            # Local
                            self.plot_errors(axes, times, estHist, trueHist, covHist, label_color=satColor, linewidth=2.5)
                            self.plot_innovations(axes, times, innovationHist, innovationCovHist, label_color=satColor, linewidth=2.5)
                            self.plot_track_error(axes, times, trackErrorHist, label_color=satColor, linewidth=2.5)
                            
                            # Central
                            self.plot_errors(axes, central_times, central_estHist, trueHist, central_covHist, label_color=centralColor, linewidth=1.5)
                            self.plot_track_error(axes, central_times, central_trackErrorHist, label_color=centralColor, linewidth=1.5)
                            
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} Indept. Estimator'),
                                Patch(color=centralColor, label=f'Central Estimator')
                            ]
                        
                        elif plotNum == 1:
                            # Second Plot: DDF vs Central
                            ddf_times, ddf_estHist, ddf_covHist, ddf_innovationHist, ddf_innovationCovHist, ddf_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'ci')
                            central_times, central_estHist, central_covHist, central_innovationHist, central_innovationCovHist, central_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'central')

                            # DDF
                            self.plot_errors(axes, ddf_times, ddf_estHist, trueHist, ddf_covHist, label_color=ddfColor, linewidth=2.5)
                            #self.plot_innovations(axes, ddf_times, ddf_innovationHist, ddf_innovationCovHist, label_color=ddfColor, linewidth=2.5)
                            self.plot_track_error(axes, ddf_times, ddf_trackErrorHist, label_color=ddfColor, linewidth=2.5)
                            
                            # Central
                            self.plot_errors(axes, central_times, central_estHist, trueHist, central_covHist, label_color=centralColor, linewidth=1.5)
                            self.plot_track_error(axes, central_times, central_trackErrorHist, label_color=centralColor, linewidth=1.5)
                            
                            handles = [
                                Patch(color=ddfColor, label=f'{sat.name} DDF Estimator'),
                                Patch(color=centralColor, label=f'Central Estimator')
                            ]
                            
                        elif plotNum == 2:
                            
                            # Third Plot is ET vs Central
                            et_times, et_estHist, et_covHist, et_innovationHist, et_innovationCovHist, et_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            central_times, central_estHist, central_covHist, central_innovationHist, central_innovationCovHist, central_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'central')
                            
                            # ET
                            self.plot_errors(axes, et_times, et_estHist, trueHist, et_covHist, label_color=satColor, linewidth=2.5)
                            self.plot_track_error(axes, et_times, et_trackErrorHist, label_color=satColor, linewidth=2.5)
                            
                            # Central
                            self.plot_errors(axes, central_times, central_estHist, trueHist, central_covHist, label_color=centralColor, linewidth=1.5)
                            self.plot_track_error(axes, central_times, central_trackErrorHist, label_color=centralColor, linewidth=1.5)
                            
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} ET Estimator'),
                                Patch(color=centralColor, label=f'Central Estimator')
                            ]
                        
                        elif plotNum == 3:
                            # ET vs DDF
                            et_times, et_estHist, et_covHist, et_innovationHist, et_innovationCovHist, et_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            ddf_times, ddf_estHist, ddf_covHist, ddf_innovationHist, ddf_innovationCovHist, ddf_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'ci')
                            
                            # ET
                            self.plot_errors(axes, et_times, et_estHist, trueHist, et_covHist, label_color=satColor, linewidth=2.5)
                            self.plot_track_error(axes, et_times, et_trackErrorHist,  label_color=satColor, linewidth=2.5)
                            
                            # DDF
                            self.plot_errors(axes, ddf_times, ddf_estHist, trueHist, ddf_covHist, label_color=ddfColor, linewidth=1.5)
                            self.plot_track_error(axes, ddf_times, ddf_trackErrorHist,  label_color=ddfColor, linewidth=1.5)
                            
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} ET Estimator'),
                                Patch(color=ddfColor, label=f'{sat.name} DDF Estimator')
                            ]
                        
                        # Add the legend and tighten the layout
                        fig.legend(handles=handles, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                        plt.tight_layout()
                            
                        # Save the Plot with respective suffix
                        self.save_plot(fig, savePlot, saveName, targ, sat, suffix_vec[plotNum])
                        
                    for sat2 in self.sats:
                        if sat != sat2:
                            fig = plt.figure(figsize=(15, 8))
                            fig.suptitle(f"{targ.name}, {sat.name}, {sat2.name} ET Filters", fontsize=14)
                            axes = self.setup_axes(fig, state_labels, meas_labels, targ.tqReq)
                            
                            sat2Color = sat2.color
                            sat1commonColor, sat2commonColor = self.shifted_colors(satColor, sat2Color, shift=50)

                            et_times, et_estHist, et_covHist, et_innovationHist, et_innovationCovHist, et_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat)
                            et_times2, et_estHist2, et_covHist2, et_innovationHist2, et_innovationCovHist2, et_trackErrorHist2 = self.getEstimationHistory(sat2, targ, time_vec, 'et', sharewith=sat2)
                            
                            et_common_times, et_common_estHist, et_common_covHist, et_common_innovationHist, et_common_innovationCovHist, et_common_trackErrorHist = self.getEstimationHistory(sat, targ, time_vec, 'et', sharewith=sat2)
                            et_common_times2, et_common_estHist2, et_common_covHist2, et_common_innovationHist2, et_common_innovationCovHist2, et_common_trackErrorHist2 = self.getEstimationHistory(sat2, targ, time_vec, 'et', sharewith=sat)
                            
                            # ET
                            self.plot_errors(axes, et_times, et_estHist, trueHist, et_covHist, label_color=satColor, linewidth=2.0)
                            self.plot_track_error(axes, et_times, et_trackErrorHist,   label_color=satColor, linewidth=2.0)
                            
                            # ET 2
                            self.plot_errors(axes, et_times2, et_estHist2, trueHist, et_covHist2, label_color=sat2Color, linewidth=2.0)
                            self.plot_track_error(axes, et_times2, et_trackErrorHist2,  label_color=sat2Color, linewidth=2.0)
                            
                            # Common ET
                            self.plot_errors(axes, et_common_times, et_common_estHist, trueHist, et_common_covHist, label_color=sat1commonColor, linewidth=2.0)
                            self.plot_track_error(axes, et_common_times, et_common_trackErrorHist,   label_color=sat1commonColor, linewidth=2.0)
                            
                            # Common ET 2
                            self.plot_errors(axes, et_common_times2, et_common_estHist2, trueHist, et_common_covHist2, label_color=sat2commonColor, linewidth=2.0)
                            self.plot_track_error(axes, et_common_times2, et_common_trackErrorHist2,  label_color=sat2commonColor, linewidth=2.0)
                            
                            # Plot Messages instead of innovations
                            self.plot_messages(axes[6], sat, sat2, targ.targetID, time_vec.value)
                            self.plot_messages(axes[7], sat2, sat, targ.targetID, time_vec.value)
                            
                            handles = [
                                Patch(color=satColor, label=f'{sat.name} ET Estimator'),
                                Patch(color=sat2Color, label=f'{sat2.name} ET Estimator'),
                                Patch(color=sat1commonColor, label=f'{sat.name}, {sat2.name} Common ET Estimator'),
                                Patch(color=sat2commonColor, label=f'{sat2.name}, {sat.name} Common ET Estimator')
                            ]
                            
                            fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0))
                            plt.tight_layout()
                                
                            # Save the Plot with respective suffix
                            currSuffix = f"{sat2.name}_" + suffix_vec[4]
                            self.save_plot(fig, savePlot, saveName, targ, sat, currSuffix)
                                        

    def getEstimationHistory(self, sat, targ, time_vec, estimatorType, sharewith=None):
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
        times, estHist, covHist, innovationHist, innovationCovHist, trackErrorHist = {}, {}, {}, {}, {}, {}
        
        if estimatorType == 'central':
            times = [time for time in time_vec.value if time in self.centralEstimator.estHist[targ.targetID]]
            estHist = self.centralEstimator.estHist[targ.targetID]
            covHist = self.centralEstimator.covarianceHist[targ.targetID]
            #innovationHist = self.centralEstimator.innovationHist[targ.targetID]
            #innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = self.centralEstimator.trackErrorHist[targ.targetID]
        
        elif estimatorType == 'ci':
            times = [time for time in time_vec.value if time in sat.ciEstimator.estHist[targ.targetID]]
            estHist = sat.ciEstimator.estHist[targ.targetID]
            covHist = sat.ciEstimator.covarianceHist[targ.targetID]
            innovationHist = sat.ciEstimator.innovationHist[targ.targetID]
            innovationCovHist = sat.ciEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = sat.ciEstimator.trackErrorHist[targ.targetID]
        
        elif estimatorType == 'et':
            times = [time for time in time_vec.value if time in sat.etEstimator.estHist[targ.targetID][sat][sharewith]]
            estHist = sat.etEstimator.estHist[targ.targetID][sat][sharewith]
            covHist = sat.etEstimator.covarianceHist[targ.targetID][sat][sharewith]
            #innovationHist = sat.etEstimator.innovationHist[targ.targetID][sat][sharewith]
            #innovationCovHist = sat.etEstimator.innovationCovHist[targ.targetID][sat][sharewith]
            trackErrorHist = sat.etEstimator.trackErrorHist[targ.targetID][sat][sharewith]
        
        elif estimatorType == 'local':
            times = [time for time in time_vec.value if time in sat.indeptEstimator.estHist[targ.targetID]]
            estHist = sat.indeptEstimator.estHist[targ.targetID]
            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
            trackErrorHist = sat.indeptEstimator.trackErrorHist[targ.targetID]
            
        return times, estHist, covHist, innovationHist, innovationCovHist, trackErrorHist
        
                    
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

                    
    def plot_messages(self, ax, sat, sat2, targetID, timeVec):

        synch_times = sat.etEstimator.synchronizeFlag[targetID][sat][sat2].keys()
        for time in timeVec:
            if time in synch_times:
                if sat.etEstimator.synchronizeFlag[targetID][sat][sat2][time] == True:
                    ax.scatter(time, 0.5, color='g', marker='D', s = 70)
                    continue
                
            if isinstance(self.comms.used_comm_et_data_values[targetID][sat.name][sat2.name][time], np.ndarray):
                alpha, beta = self.comms.used_comm_et_data_values[targetID][sat.name][sat2.name][time]
                if not np.isnan(alpha):
                    ax.scatter(time, 0.9, color='r', marker=r'$\alpha$', s = 80)
                else:
                    ax.scatter(time, 0.2, color='b', marker=r'$\alpha$', s = 80)
                    
                if not np.isnan(beta):
                    ax.scatter(time, 0.8, color='r',  marker=r'$\beta$', s = 120)
                else:
                    ax.scatter(time, 0.1, color='b',  marker=r'$\beta$', s = 120)
                        
        
        ax.set_yticks([0,0.5, 1])
        # set the axis limits to be the whole time vector
        ax.set_xlim([timeVec[0], timeVec[-1]])
        ax.set_yticklabels(['Implicit', 'CI', 'Explict'])
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Message Type')
        ax.set_title(f'{sat2.name} -> {sat.name} Messages')

        
    def plot_track_error(self, ax, times, trackErrorHist,label_color, linewidth):
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
    
    
    def shifted_colors(self, hex_colors1, hex_colors2, shift=50):
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB tuple."""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

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
        
        sat1commonColor, sat2commonColor = find_middle_colors(hex_colors1, hex_colors2, shift)
        return sat1commonColor, sat2commonColor
        

    def setup_axes(self, fig, state_labels, meas_labels, targQuality):
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
        for i in range(6): # TODO: Should we fix all x-axis to be time vector
            axes[i].set_xlabel("Time [min]")
            axes[i].set_ylabel(f"Error in {state_labels[i]}")
        for i in range(2):
            axes[6 + i].set_xlabel("Time [min]")
            axes[6 + i].set_ylabel(f"Innovation in {meas_labels[i]}")
        axes[8].set_xlabel("Time [min]")
        axes[8].set_ylabel("Track Uncertainity [km]")
         # Finally plot a dashed line for the targetPriority
        axes[8].axhline(y=targQuality*50 + 50, color='k', linestyle='dashed', linewidth=1.5)
                # Add a text label on the above right side of the dashed line
        axes[8].text(1, targQuality*50 + 50 + 5, f"Target Quality: {targQuality}", fontsize=8, color='k')
        
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


### Plot communications sent/recieved  
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
            
            
        # Do the exact sane thing fior et data
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
                    
                    for time in self.comms.total_comm_et_data[target_id][reciever][sender]:
                        
                        # Get the data
                        data = self.comms.total_comm_et_data[target_id][reciever][sender][time]
                        
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
        ax.set_ylabel('ET Data Sent/Recieved (# of numbers)')
        
        # Add the x-axis labels
        ax.set_xticks(np.arange(len(satNames)))
        ax.set_xticklabels(satNames)
        
        # Now save the plot
        filePath = os.path.dirname(os.path.realpath(__file__))
        plotPath = os.path.join(filePath, 'plots')
        os.makedirs(plotPath, exist_ok=True)
        plt.savefig(os.path.join(plotPath, f"{saveName}_total_et_comms.png"), dpi=300)


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

                    for time in self.comms.used_comm_et_data[target_id][reciever][sender]:

                        # Get the data
                        data = self.comms.used_comm_et_data[target_id][reciever][sender][time]

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
            plt.savefig(os.path.join(plotPath, f"{saveName}_used_et_comms.png"), dpi=300)
            

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
                    if (sat.etEstimator.synchronizeFlag[targetID][sat][sat2]):
                        if sat.etEstimator.synchronizeFlag[targetID][sat][sat2].get(envTime) == True:
                            diComms.add_edge(sat, sat2)
                            style = self.get_edge_style(comms, targetID, sat, sat2, envTime, CI = True)
                            edge_styles.append((sat, sat2, style, targ_color))
                    
                    value = comms.used_comm_et_data_values[targetID][sat.name][sat2.name][envTime]
                    print(f"Receiver {sat.name}, Sender {sat2.name}, Value {value}")
                    # If there is a communication between the two satellites, add an edge
                    if isinstance(comms.used_comm_et_data_values[targ.targetID][sat.name][sat2.name][envTime], np.ndarray):
                            diComms.add_edge(sat2, sat)
                            style = self.get_edge_style(comms, targetID, sat, sat2, envTime)
                            edge_styles.append((sat2, sat, style, targ_color))
                        

            # Draw the graph with the nodes and edges
        
        if once:
            pos = nx.circular_layout(diComms)
            nx.draw_networkx_nodes(diComms, pos, ax=ax, node_size=1000, node_color=node_colors)
            once = False
        # Draw edges with appropriate styles
        for i, edge in enumerate(edge_styles):
            # Adjust the curvature for each edge
            connectionstyle = f'arc3,rad={(i - len(edge_styles) / 2) / len(edge_styles)}'
            nx.draw_networkx_edges(
                diComms, pos, edgelist=[(edge[0], edge[1])], ax=ax, style=edge[2], edge_color=edge[3], arrows=True, arrowsize=10, connectionstyle=connectionstyle, min_source_margin=0, min_target_margin=40, width = 2
            )

        # Add labels
        labels = {node: node.name for node in diComms.nodes()}
        nx.draw_networkx_labels(diComms, pos, ax=ax, labels=labels)
        # Add Title
        ax.set_title(f"Dynamic Communications at Time {envTime} min")
        handles = [
            Patch(color=targ.color, label=targ.name) for targ in self.targs
        ]
        fig.legend(handles=handles, loc='lower right', ncol = 1, bbox_to_anchor=(0.9, 0.1))

            # Display and close the figure
        img = self.save_comm_plot_to_image(fig)
        self.imgs_dyn_comms.append(img)
            
        ax.cla()
        # clear graph
        diComms.clear()
        plt.close(fig)

    def get_edge_style(self, comms, targetID, sat1, sat2, envTime, CI = False):
        """
        Helper function to determine the edge style based on communication data.
        Returns 'solid' if both alpha and beta are present, 'dashed' if only one is present, 
        and None if neither is present (meaning no line).
        """
        
        if CI:
            return (0, ())
        
        alpha, beta = comms.used_comm_et_data_values[targetID][sat1.name][sat2.name][envTime]
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
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
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
                                fig.suptitle(f"{targ.name}, {sat.name}, {sat2.name} Stereo Gaussian Uncertainty Ellipsoids")

                                sat1Color = sat.color
                                sat2Color = sat2.color
                                ddfColor = '#DC143C' # Crimson
                                etColor = '#DC143C' # Crimson
                                centralColor = '#070400'  
                                alpha = 0.2
                                
                                # Create 2x2 Grid
                                ax1 = fig.add_subplot(221, projection='3d')
                                ax2 = fig.add_subplot(222, projection='3d')
                                ax3 = fig.add_subplot(223, projection='3d')
                                ax4 = fig.add_subplot(224, projection='3d')
                                plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.15, hspace=0.15)
                   

                                
                                for bigTime in time_vec.value:
                                    time = bigTime
                                    sat1_times = sat.indeptEstimator.estHist[targ.targetID].keys()
                                    sat2_times = sat2.indeptEstimator.estHist[targ.targetID].keys()
                                    stereo_times = [time for time in sat1_times if time in sat2_times]
                                    
                                    ddf_times = sat.ciEstimator.estHist[targ.targetID].keys()
                                    times = [time for time in stereo_times if time in ddf_times]
                                    
                                    if bigTime in times:
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

                                        self.plot_ellipsoid(ax1, est_pos1, cov_matrix1, color=sat1Color, alpha=alpha)
                                        self.plot_ellipsoid(ax1, est_pos2, cov_matrix2, color=sat2Color, alpha=alpha)
                                        self.plot_ellipsoid(ax1, ddf_pos, ddf_cov, color=ddfColor, alpha=alpha+0.1)

                                        self.plot_estimate(ax1, est_pos1, true_pos, sat1Color)
                                        self.plot_estimate(ax1, est_pos2, true_pos, sat2Color)
                                        self.plot_estimate(ax1, ddf_pos, true_pos, ddfColor)

                                        self.plot_LOS(ax1, est_pos1, LOS_vec1)
                                        self.plot_LOS(ax1, est_pos2, LOS_vec2)
                                        
                                        self.set_axis_limits(ax1, ddf_pos, np.sqrt(ddf_eigenvalues),  margin=50.0)
                                        self.plot_labels(ax1, time)
                                        self.make_legened1(ax1, sat, sat1Color, sat2, sat2Color, ddfColor, error1, error2, ddf_error, 'CI')
      
                                    et_times = sat.etEstimator.estHist[targ.targetID][sat][sat].keys()
                                    times = [time for time in stereo_times if time in et_times]
                                    
                                    if bigTime in times:
                                        # Plot Et
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
                                        
                                        self.plot_ellipsoid(ax2, est_pos1, cov_matrix1, color=sat1Color, alpha=alpha)
                                        self.plot_ellipsoid(ax2, est_pos2, cov_matrix2, color=sat2Color, alpha=alpha)
                                        self.plot_ellipsoid(ax2, et_pos, et_cov, color=etColor, alpha=alpha)
                                        
                                        self.plot_estimate(ax2, est_pos1, true_pos, sat1Color)
                                        self.plot_estimate(ax2, est_pos2, true_pos, sat2Color)
                                        self.plot_estimate(ax2, et_pos, true_pos, etColor)
                                        
                                        self.plot_LOS(ax2, est_pos1, LOS_vec1)
                                        self.plot_LOS(ax2, est_pos2, LOS_vec2)
                                        
                                        self.make_legened1(ax2, sat, sat1Color, sat2, sat2Color, etColor, error1, error2, et_error, 'ET')
                                        self.set_axis_limits(ax2, et_pos, np.sqrt(et_eigenvalues), margin=50.0)
                                        self.plot_labels(ax2, time)
                                        
                                        
                                    ddf_times = sat.ciEstimator.estHist[targ.targetID].keys()
                                    central_times = self.centralEstimator.estHist[targ.targetID].keys()
                                    times = [time for time in ddf_times if time in central_times]
                                    
                                    if bigTime in times:
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos = np.array([sat.ciEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                                        central_pos = np.array([self.centralEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])

                                        cov_matrix = sat.ciEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                                        central_cov = self.centralEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]

                                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                        central_eigenvalues, central_eigenvectors = np.linalg.eigh(central_cov)

                                        error = np.linalg.norm(true_pos - est_pos)
                                        central_error = np.linalg.norm(true_pos - central_pos)

                                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                                        self.plot_ellipsoid(ax3, est_pos, cov_matrix, color=ddfColor, alpha=alpha)
                                        self.plot_ellipsoid(ax3, central_pos, central_cov, color=centralColor, alpha=alpha)

                                        self.plot_estimate(ax3, est_pos, true_pos, sat1Color)
                                        self.plot_estimate(ax3, central_pos, true_pos, centralColor)

                                        self.plot_LOS(ax3, est_pos, LOS_vec)
                                        self.set_axis_limits(ax3, est_pos, np.sqrt(eigenvalues), margin=50.0)
                                        self.plot_labels(ax3, time)
                                        self.make_legened2(ax3, ddfColor, centralColor, error, central_error, 'CI')
                                        
                                    et_times = sat.etEstimator.estHist[targ.targetID][sat][sat].keys()
                                    central_times = self.centralEstimator.estHist[targ.targetID].keys()
                                    times = [time for time in et_times if time in central_times]
                                    
                                    if bigTime in times:
                                        true_pos = targ.hist[time][[0, 2, 4]]
                                        est_pos = np.array([sat.etEstimator.estHist[targ.targetID][sat][sat][time][i] for i in [0, 2, 4]])
                                        central_pos = np.array([self.centralEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])

                                        cov_matrix = sat.etEstimator.covarianceHist[targ.targetID][sat][sat][time][np.array([0, 2, 4])][:, np.array([0, 2, 4])]
                                        central_cov = self.centralEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]

                                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                                        central_eigenvalues, central_eigenvectors = np.linalg.eigh(central_cov)

                                        error = np.linalg.norm(true_pos - est_pos)
                                        central_error = np.linalg.norm(true_pos - central_pos)

                                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                                        self.plot_ellipsoid(ax4, est_pos, cov_matrix, color=etColor, alpha=alpha)
                                        self.plot_ellipsoid(ax4, central_pos, central_cov, color=centralColor, alpha=alpha)

                                        self.plot_estimate(ax4, est_pos, true_pos, sat1Color)
                                        self.plot_estimate(ax4, central_pos, true_pos, centralColor)

                                        self.plot_LOS(ax4, est_pos, LOS_vec)
                                        self.set_axis_limits(ax4, est_pos, np.sqrt(eigenvalues), margin=50.0)
                                        self.plot_labels(ax4, time)
                                        self.make_legened2(ax4, etColor, centralColor, error, central_error, 'ET')
                                    
                                    handles = [
                                        Patch(color=sat1Color, label=f'{sat.name} Local Estimator'),
                                        Patch(color=sat2Color, label=f'{sat2.name} Local Estimator'),
                                        Patch(color=ddfColor, label=f'DDF Estimator'),
                                        Patch(color=etColor, label=f'ET Estimator'),
                                        Patch(color=centralColor, label=f'Central Estimator')
                                    ]
                                    
                                    fig.legend(handles=handles, loc='lower right', ncol=5, bbox_to_anchor=(1, 0))
                                    ax1.set_title(f"Covariance Intersection")
                                    ax2.set_title(f"Event Triggered Fusion")
                                    img = self.save_GEplot_to_image(fig)
                                    self.imgs_stereo_GE[targ.targetID][sat][sat2].append(img)
                                    
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
       # Calculate the end point of the LOS vector relative to est_pos
        arrow_length = 30  # Length of the LOS vector
        LOS_vec_unit = LOS_vec / np.linalg.norm(LOS_vec)  # Normalize the LOS vector

        # Adjusted starting point of the arrow
        arrow_start = est_pos - (LOS_vec_unit * arrow_length)

        # Use quiver to plot the arrow starting from arrow_start to est_pos
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], 
                LOS_vec_unit[0], LOS_vec_unit[1], LOS_vec_unit[2], 
                color='k', length=arrow_length, normalize=True)
        #ax.quiver(est_pos[0], est_pos[1], est_pos[2], LOS_vec[0], LOS_vec[1], LOS_vec[2], color='k', length=10, normalize=True)


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


    def make_legened1(self, ax, sat1, sat1color, sat2, sat2color, ddfcolor, error1, error2, ddf_error, ddf_type=None):
        if ddf_type == 'CI':
            labels = [f'{sat1.name} Error: {error1:.2f} km', f'{sat2.name} Error: {error2:.2f} km', f'CI Error: {ddf_error:.2f} km']
            handles = [Patch(color=sat1color, label=labels[0]), Patch(color=sat2color, label=labels[1]), Patch(color=ddfcolor, label=labels[2])]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))
        
        elif ddf_type == 'ET':
            labels = [f'{sat1.name} Error: {error1:.2f} km', f'{sat2.name} Error: {error2:.2f} km', f'ET Error: {ddf_error:.2f} km']
            handles = [Patch(color=sat1color, label=labels[0]), Patch(color=sat2color, label=labels[1]), Patch(color=ddfcolor, label=labels[2])]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))


    def make_legened2(self, ax, ddfColor, centralColor, error1, error2, ddf_type=None):
        if ddf_type == 'CI':
            labels = [f'CI Error: {error1:.2f} km', f'Central Error: {error2:.2f} km']
            handles = [Patch(color=ddfColor, label=labels[0]), Patch(color=centralColor, label=labels[1])]
            ax.legend(handles=handles, loc='upper right', ncol=1, bbox_to_anchor=(1, 1))
        
        elif ddf_type == 'ET':
            labels = [f'ET Error: {error1:.2f} km', f'Central Error: {error2:.2f} km']
            handles = [Patch(color=ddfColor, label=labels[0]), Patch(color=centralColor, label=labels[1])]
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
            for targ in self.targs:
                for sat in self.sats:
                    if targ.targetID in sat.targetIDs:
                        for sat2 in self.sats:
                            if targ.targetID in sat2.targetIDs:
                                if sat != sat2:
                                    file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_{sat2.name}_stereo_GE.gif")
                                    with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                                        for img in self.imgs_stereo_GE[targ.targetID][sat][sat2]:
                                            writer.append_data(img)
                                            
        if fileType == 'dynamic_comms':
                file = os.path.join(filePath, 'gifs', f"{saveName}_dynamic_comms.gif")
                with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                    for img in self.imgs_dyn_comms:
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
                    et_trackError = sat.etEstimator.trackErrorHist[targ.targetID][sat][sat]
                    
                    # File Name
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}.csv"

                    # Format the data and write it to the file
                    self.format_data(
                        filename, targ.targetID, times, sat_hist, trueHist,
                        sat_measHistTimes, sat_measHist, estTimes, estHist, covHist,
                        trackError, innovationHist, innovationCovHist, ddf_times,
                        ddf_estHist, ddf_covHist, ddf_trackError, ddf_innovation_times,
                        ddf_innovationHist, ddf_innovationCovHist, et_times, et_estHist, et_covHist, et_trackError
                    )


    def format_data(
        self, filename, targetID, times, sat_hist, trueHist, sat_measHistTimes,
        sat_measHist, estTimes, estHist, covHist, trackError, innovationHist,
        innovationCovHist, ddf_times, ddf_estHist, ddf_covHist, ddf_trackError,
        ddf_innovation_times, ddf_innovationHist, ddf_innovationCovHist, et_times,
        et_estHist, et_covHist, et_trackError
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
        - et_trackError (dict): ET track quality history.

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
                'ET_Cov_vzvz', 'ET_Track Error'
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
                    row += format_list(et_trackError[time])

                writer.writerow(row)
                

    def log_comms_data(self, time_vec, saveName, filePath=os.path.dirname(os.path.realpath(__file__))):
        for sat in self.sats:
            for targ in self.targs:
                if targ.targetID in sat.targetIDs:
                    commNode = self.comms.G.nodes[sat]
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}_comm.csv"
                    self.format_comms_data(filename, time_vec.value, sat, commNode, targ.targetID)
                    
    def format_comms_data(self, filename, time_vec, sat, commNode, targetID):
        precision = 3
        def format_list(lst):
            if isinstance(lst, np.ndarray):
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst.flatten()]
            elif isinstance(lst, int) or isinstance(lst, float):
                return [f"{float(lst):.{precision}f}" if not np.isnan(lst) else "nan"]
            else:
                return [f"{x:.{precision}f}" if not np.isnan(x) else "nan" for x in lst]
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow([
                'Time', 'Satellite', 'Target', 'Sender', "Received Alpha", "Received Beta", "Receiver", "Sent Alpha", "Sent Beta"])
            
            times = [time for time in time_vec]
            timeReceived = [time for time in commNode['received_measurements'].keys()]
            timesSent = [time for time in commNode['sent_measurements'].keys()]
            
            for time in times:
                row = [f"{time:.{precision}f}"]
                row += [sat.name]
                row += [f"Targ{targetID}"]
                
                if time in timeReceived:
                    for i in range(len(commNode['received_measurements'][time])):
                        row += [commNode['received_measurements'][time][targetID]['sender'][i].name]#format_list(commNode['received_measurements'][time][targetID]['sender'][i])
                        row += format_list(commNode['received_measurements'][time][targetID]['meas'][i])
                else:
                    row += ['', '', '']
                    
                if time in timesSent:
                    for i in range(len(commNode['sent_measurements'][time])):
                        row +=  [commNode['sent_measurements'][time][targetID]['receiver'][i].name]#format_list(commNode['sent_measurements'][time][targetID]['receiver'][i])
                        row += format_list(commNode['sent_measurements'][time][targetID]['meas'][i])
                else:
                    row += ['', '', '']
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
    

