from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment: 
    def __init__(self, sats, targs, comms, centralEstimator = None):

        # If a central estimator is passed, use it
        if centralEstimator:
            self.centralEstimator = centralEstimator
        else:
            self.centralEstimator = None

        # Define the satellites
        self.sats = sats

        # Define the targets
        self.targs = targs

        # Define the communication network
        self.comms = comms

        # Time parameter, initalize to 0
        self.time = 0

        # Plotting parameters
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Set the axis limits
        self.ax.set_xlim([-15000, 15000])
        self.ax.set_ylim([-15000, 15000])
        self.ax.set_zlim([-15000, 15000])
        self.ax.view_init(elev=30, azim=30)
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.set_title('Satellite Orbit Visualization')
        
        # All earth parameters for plotting
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.earth_r = 6378.0
        self.x_earth = self.earth_r * np.outer(np.cos(u), np.sin(v))
        self.y_earth = self.earth_r * np.outer(np.sin(u), np.sin(v))
        self.z_earth = self.earth_r * np.outer(np.ones(np.size(u)), np.cos(v))

        # Empty images list to later make a gif of the simulation
        self.imgs = []
        
        # Make a list of images for the uncertainity ellipse for every sat target pair
        self.imgs_UC = defaultdict(lambda: defaultdict(list))
        self.imgs_UC_DDF =  defaultdict(lambda: defaultdict(list))
        self.imgs_UC_LC_DDF = defaultdict(lambda: defaultdict(list))
        
# Simulate the environment over a time range
    # Time range is a numpy array of time steps, must have poliastro units associated!
    # Pause step is the time to pause between each step, if displaying as animation
    # Display is a boolean, if true will display the plot as an animation
    def simulate(self, time_vec, pause_step = 0.1, savePlot = False, saveData = False, saveName = None, showSim = False):
        
        # Initalize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time # Get delta time to propagate, works because propagate func increases time after first itr
        
        # Propagate the satellites and environments position
            self.propagate(t_d)

        # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion()

            if savePlot:
            # Update the plot environment
                self.plot()
                self.convert_imgs()
                if showSim:
                    plt.pause(pause_step)
                    plt.draw()
        
        if savePlot:
            # Plot the results of the simulation.        
            self.plotResults2(time_vec, savePlot = savePlot, saveName = saveName)
            self.plot_UncertaintyEllipse()

        # Log the Data
        if saveData:
            self.log_data(time_vec, saveName = saveName)
        
        return self.collectData()
        
    """
        data_fusion

        Collect measurements from all satellites and do data fusion
    """
    def data_fusion(self):
        
    # TODO: CHANGE THIS CENTRAL STUFF LATER TO BE CLEANER
        collectedFlag = defaultdict(lambda: defaultdict(dict))
        measurements = defaultdict(lambda: defaultdict(dict))
        # Collect measurements on any avaliable targets
        for targ in self.targs:
            for sat in self.sats:
                # TODO: CHANGE THIS TO USE AVALIABLE TARGETS
                # Collect the bearing measurement, if avaliable, and run an EKF to update the estimate on the target
                collectedFlag[targ][sat] = sat.collect_measurements_and_filter(targ)

                if collectedFlag[targ][sat]:
                    measurements[targ][sat] = sat.measurementHist[targ.targetID][self.time.to_value()]

        # Now do central fusion
        for targ in self.targs:
            # Extract Satellites that took a measurement
            satsWithMeasurements = [sat for sat in self.sats if collectedFlag[targ][sat]]
            newMeasurements = [measurements[targ][sat] for sat in satsWithMeasurements]

            # If any satellite took a measurement on this target    
            if satsWithMeasurements:
                # Run EKF with all satellites that took a measurement on the target
                self.centralEstimator.EKF(satsWithMeasurements, newMeasurements, targ, self.time.to_value())

        # Now send estimates for future CI
        self.send_estimates()

        # Now, each satellite will perform covariance intersection on the measurements sent to it
        for sat in self.sats:
            sat.ddfEstimator.CI(sat, self.comms.G.nodes[sat])


    """
        send_estimates

        For each satellite send the most recent estimate to its neighbors
    """
    def send_estimates(self):

        # Loop through all satellites
        for sat in self.sats:

            # For each targetID in the satellite estimate history
            for targetID in sat.ddfEstimator.estHist.keys():

                # TODO: SOON WONT NEED THIS EXTRA CHECK BELOW AS TARGETIDS WILL ONLY GET INITALIZED IF THEY HAVE ESTIMATES
                if len(sat.ddfEstimator.estHist[targetID].keys()) == 0:
                    continue

                # This means satellite has an estimate for this target, now send it to neighbors
                for neighbor in self.comms.G.neighbors(sat):

                    # Get the most recent estimate time
                    satTime = max(sat.ddfEstimator.estHist[targetID].keys())

                    # Send most recent estimate to neighbor
                    self.comms.send_estimate(sat, neighbor, sat.ddfEstimator.estHist[targetID][satTime], sat.ddfEstimator.covarianceHist[targetID][satTime], targetID, satTime)

# Propagate the satellites over the time step  
    def propagate(self, time_step):
        
    # Update the current time
        self.time += time_step
        print("Time: ", self.time.to_value())

        time_val = self.time.to_value(self.time.unit)
        # Update the time in targs, sats, and estimator
        for targ in self.targs:
            targ.time = time_val
        for sat in self.sats:
            sat.time = time_val

        # Propagate the targets position
        for targ in self.targs:

            # Propagate the target
            targ.propagate(time_step, self.time)

            # Update the history of the target, time and xyz position and velocity [x xdot y ydot z zdot]
            #targ.hist[targ.time] = np.array([targ.pos[0], targ.vel[0], targ.pos[1], targ.vel[1], targ.pos[2], targ.vel[2]])


        collectedFlag = np.zeros(np.size(self.sats))
        satNum = 0
        # Propagate the satellites
        for sat in self.sats:
            
        # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)
            sat.orbitHist[sat.time] = sat.orbit.r.value # history of sat time and xyz position

        # Update the communication network for the new sat position:
            self.comms.make_edges(self.sats)

        

# Plot the current state of the environment
    def plot(self):
        # Reset plot
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()
        for text in self.ax.texts:
            text.remove()

        # Put text of current time in top left corner
        self.ax.text2D(0.05, 0.95, f"Time: {self.time:.2f}", transform=self.ax.transAxes)

    # FOR EACH SATELLITE, PLOTS
        for sat in self.sats:
        # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value
            # Cut the label of a satellite off before the first underscore
            satName = sat.name.split('.')[0]
            self.ax.scatter(x, y, z, s=40, color = sat.color, label=satName)

        # Plot the visible projection of the satellite sensor
            points = sat.sensor.projBox
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color = sat.color, marker = 'x')
            
            box = np.array([points[0], points[3], points[1], points[2], points[0]])
            self.ax.add_collection3d(Poly3DCollection([box], facecolors=sat.color, linewidths=1, edgecolors=sat.color, alpha=.1))

    # FOR EACH TARGET, PLOTS
        for targ in self.targs:
        # Plot the current xyz location of the target
            x, y, z = targ.pos
            vx, vy, vz = targ.vel
            # self.ax.scatter(x, y, z, s=20, marker = '*', color = targ.color, label=targ.name)
            mag = np.linalg.norm([vx, vy, vz])
            if mag > 0:
                vx, vy, vz = vx / mag, vy / mag, vz / mag

            self.ax.quiver(x, y, z, vx*1000, vy*1000, vz*1000, color = targ.color, arrow_length_ratio=0.75, label=targ.name)
            
        # PLOT EARTH
        #self.ax.plot_surface(self.x_earth - 1000, self.y_earth - 1000, self.z_earth - 1000, color = 'white', alpha=1)
        self.ax.plot_surface(self.x_earth, self.y_earth, self.z_earth, color = 'k', alpha=0.1)
        
        # Get rid of any duplicates in the legend:
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

    # PLOT COMMUNICATION STRUCTURE
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
        
                if self.comms.G.edges[edge]['active']:
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], color=(0.3, 1.0, 0.3), linestyle='dashed', linewidth=2)
                else:
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], color='k', linestyle='dashed', linewidth=1)
        

    def plotResults2(self, time_vec, savePlot, saveName):
        # Close the sim plot so that sizing of plots is good
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/min]', 'Y [km]', 'Vy [km/min]', 'Z [km]', 'Vz [km/min]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]', 'NIS/NEES']
        
        # FOR EACH TARGET and EACH SATELLITE MAKE A PLOT
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    for k in range(3):
                        # Create a figure
                        fig = plt.figure(figsize=(15, 8))
                        
                        # Subtitle with the name of the target
                        fig.suptitle(f"{targ.name}, {sat.name} Estimation Error and Innovation Plots", fontsize=14)
                        # Create a GridSpec object with 3 rows and 6 columns (3 col x y z, 3 col vx vy vz , 2 col alpha beta)
                        gs = gridspec.GridSpec(3, 6)
                        
                        # Collect axes in a list of lists for easy access
                        axes = []
                        # Create subplots in the first two rows (3 2-columns spanning the width)
                        for i in range(3):  # 2 rows * 6 columns = 3
                            ax = fig.add_subplot(gs[0, 2*i:2*i + 2])
                            ax.grid(True)
                            axes.append(ax)
                            
                            ax = fig.add_subplot(gs[1, 2*i:2*i + 2])
                            ax.grid(True)
                            axes.append(ax)
                            
                        # Create subplots in the third row (3 columns spanning the width)
                        for i in range(2):  # 1 row * 2 columns = 2
                            ax = fig.add_subplot(gs[2, 2*i:2*i+2])
                            ax.grid(True)
                            axes.append(ax)
                        
                        # Create a subplot for NIS/NEES
                        ax = fig.add_subplot(gs[2, 4:6])
                        ax.grid(True)
                        axes.append(ax)
                        
                        # Set the labels for the subplots
                        # Do the error vs covariance plots on the first row:
                        for i in range(6):
                            axes[i].set_xlabel("Time [min]")
                            axes[i].set_ylabel(f"Error in {state_labels[i]}")
                        # Do the innovation vs innovation covariance plots on the third row:
                        for i in range(2):  # Note: only 2 plots in the third row
                            axes[6 + i].set_xlabel("Time [min]")
                            axes[6 + i].set_ylabel(f"Innovation in {meas_labels[i]}")
                        
                        # Set the labels for the NIS/NEES plot
                        axes[8].set_xlabel("Time [min]")
                        axes[8].set_ylabel("NIS/NEES")
                        
                        # FOR EACH SATELLITE, EXTRACT ALL DATA for either the independent estimator or ddf estimator but always the central estimator
                        satColor = sat.color
                        trueHist = targ.hist
                        
                        if k == 0: # Plot Just Independent and Central Estimators
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]    
                            times = [time for time in time_vec.value if time in estHist]
                            
                            # ERROR PLOTS
                            for i in range(6):
                                if times:
                                    axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate'
                                    axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
                                    axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                                
                            # INNOVATION PLOTS
                            for i in range(2):  
                                if times:
                                    axes[6 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate')
                                    axes[6 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed')#, label='2 Sigma Bounds')
                                    axes[6 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed')
                            
                            # NIS/NEES PLOTS
                            if times:
                                axes[8].plot(times, [NISHist[time] for time in times], color='b', linewidth=2.5)
                                axes[8].plot(times, [NEESHist[time] for time in times], color='k', linewidth=2.5)
                                
                            if self.centralEstimator:
                                if targ.targetID in self.centralEstimator.estHist:
                                    trueHist = targ.hist
                                    estHist = self.centralEstimator.estHist[targ.targetID]
                                    covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                    innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                    innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                    times = [time for time in time_vec.value if time in estHist]       
                                
                                # ERROR PLOTS
                                for i in range(6):
                                    axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='#228B22')
                                    axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')#, label='2 Sigma Bounds')
                                    axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')
                                    
                            # COLLECT LEGENDS REMOVING DUPLICATES
                            handles, labels = [], []
                            for ax in axes:
                                for handle, label in zip(*ax.get_legend_handles_labels()):
                                    if label not in labels:  # Avoid duplicates in the legend
                                        handles.append(handle)
                                        labels.append(label)
                                        
                            # Set Axis Limits
                            min_time = min(targ.hist.keys())
                            max_time = max(targ.hist.keys())

                            for ax in axes:
                                ax.set_xlim(min_time, max_time)

                            # Create a Patch object for the satellite
                            satPatch = Patch(color=satColor, label=sat.name)
                            
                            # Add the Patch object to the handles and labels
                            handles.append(satPatch)
                            labels.append(sat.name)
                    
                            # ALSO ADD CENTRAL IF FLAG IS SET
                            if self.centralEstimator:
                                # Create a Patch object for the central estimator
                                centralPatch = Patch(color='#228B22', label='Central Estimator')
                                
                                # Add the Patch object to the handles and labels
                                handles.append(centralPatch)
                                labels.append('Central Estimator')
                            
                            # ADD LEGEND
                            fig.legend(handles, labels, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                            plt.tight_layout()

                            # SAVE PLOT
                            if savePlot:
                                filePath = os.path.dirname(os.path.realpath(__file__))
                                plotPath = os.path.join(filePath, 'plots')
                                os.makedirs(plotPath, exist_ok=True)
                                if saveName is None:
                                    plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_local_results.png"), dpi=300)
                                    return
                                plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_local_results.png"), dpi=300)

                            plt.close()
                        
                        
                        if k == 1: # Plot Just DDF and Central Estimators
                            ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ddfEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ddfEstimator.neesHist[targ.targetID]
                            
                            ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
                            
                            # ERROR PLOTS
                            for i in range(6):
                                if ddf_times:
                                    axes[i].plot(ddf_times, [ddf_estHist[time][i] - trueHist[time][i] for time in ddf_times], color='#DC143C', linewidth=2.5)
                                    axes[i].plot(ddf_times, [2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed')
                                    axes[i].plot(ddf_times, [-2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed')
                                    
                            # INNOVATION PLOTS
                            for i in range(2):
                                if ddf_innovation_times:
                                    axes[6 + i].plot(ddf_innovation_times, [ddf_innovationHist[time][i] for time in ddf_innovation_times], color='#DC143C', linewidth=2.5)
                                    axes[6 + i].plot(ddf_innovation_times, [2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')
                                    axes[6 + i].plot(ddf_innovation_times, [-2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')
                                    
                            # NIS/NEES PLOTS
                            if ddf_NISNEES_times:
                                axes[8].plot(ddf_NISNEES_times, [ddf_NISHist[time] for time in ddf_NISNEES_times], color='b', linewidth=2.5)
                                axes[8].plot(ddf_NISNEES_times, [ddf_NEESHist[time] for time in ddf_NISNEES_times], color='k', linewidth=2.5)
                            
                            if self.centralEstimator:
                                if targ.targetID in self.centralEstimator.estHist:
                                    trueHist = targ.hist
                                    estHist = self.centralEstimator.estHist[targ.targetID]
                                    covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                    innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                    innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                    times = [time for time in time_vec.value if time in estHist]       
                                
                                # ERROR PLOTS
                                for i in range(6):
                                    axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='#228B22')
                                    axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')#, label='2 Sigma Bounds')
                                    axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')
                                    
                            # COLLECT LEGENDS REMOVING DUPLICATES
                            handles, labels = [], []
                            for ax in axes:
                                for handle, label in zip(*ax.get_legend_handles_labels()):
                                    if label not in labels:  # Avoid duplicates in the legend
                                        handles.append(handle)
                                        labels.append(label)
                                        
                            # Set Axis Limits
                            min_time = min(targ.hist.keys())
                            max_time = max(targ.hist.keys())

                            for ax in axes:
                                ax.set_xlim(min_time, max_time)

                            # CREATE A DDF PATCH
                            ddfPatch = Patch(color='#DC143C', label='DDF Estimator')
                            handles.append(ddfPatch)
                            labels.append('DDF Estimator')
                            
                            
                            # ALSO ADD CENTRAL IF FLAG IS SET
                            if self.centralEstimator:
                                # Create a Patch object for the central estimator
                                centralPatch = Patch(color='#228B22', label='Central Estimator')
                                
                                # Add the Patch object to the handles and labels
                                handles.append(centralPatch)
                                labels.append('Central Estimator')
                                
                            # ADD LEGEND
                            fig.legend(handles, labels, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                            plt.tight_layout()
                                
                            # SAVE PLOT
                            if savePlot:
                                filePath = os.path.dirname(os.path.realpath(__file__))
                                plotPath = os.path.join(filePath, 'plots')
                                os.makedirs(plotPath, exist_ok=True)
                                if saveName is None:
                                    plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_ddf_results.png"), dpi=300)
                                    return
                                plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_ddf_results.png"), dpi=300)

                            plt.close()
                        
                        if k == 2: # Plot Everything        
                            # FOR EACH SATELLITE, EXTRACT ALL DATA for independent estimator and ddf estimator
                            satColor = sat.color
                            trueHist = targ.hist
                            estHist = sat.indeptEstimator.estHist[targ.targetID]
                            covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                            innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                            innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                            NISHist = sat.indeptEstimator.nisHist[targ.targetID]
                            NEESHist = sat.indeptEstimator.neesHist[targ.targetID]
                                
                            ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                            ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                            ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                            ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                            ddf_NISHist = sat.ddfEstimator.nisHist[targ.targetID]
                            ddf_NEESHist = sat.ddfEstimator.neesHist[targ.targetID]
                                
                                
                            times = [time for time in time_vec.value if time in estHist]
                            ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                            ddf_NISNEES_times = [time for time in time_vec.value if time in ddf_NISHist]
        
                    
                        # ERROR PLOTS
                            for i in range(6):
                                if times:
                                    axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate'
                                    axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
                                    axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                                
                                if ddf_times:
                                    axes[i].plot(ddf_times, [ddf_estHist[time][i] - trueHist[time][i] for time in ddf_times], color='#DC143C',linewidth=1.5) # Error')
                                    axes[i].plot(ddf_times, [2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed', linewidth=1.5)# label='DDF 2 Sigma Bounds')
                                    axes[i].plot(ddf_times, [-2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed',linewidth=1.5)
                                        
                        # INNOVATION PLOTS
                            for i in range(2):  # Note: only 3 plots in the third row
                                if times:
                                    axes[6 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate')
                                    axes[6 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed')#, label='2 Sigma Bounds')
                                    axes[6 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed')
                                
                                if ddf_innovation_times:
                                    axes[6 + i].plot(ddf_innovation_times, [ddf_innovationHist[time][i] for time in ddf_innovation_times], color='#DC143C',linewidth=1.5)#, label='DDF Estimate')
                                    axes[6 + i].plot(ddf_innovation_times, [2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')#, label='DDF 2 Sigma Bounds')
                                    axes[6 + i].plot(ddf_innovation_times, [-2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')

                        
                        # NIS/NEES PLOTS
                            if times:
                                axes[8].plot(times, [NISHist[time] for time in times], color='b', linewidth=1.5)
                                axes[8].plot(times, [NEESHist[time] for time in times], color='k', linewidth=1.5)
                            
                            if ddf_NISNEES_times:
                                axes[8].plot(ddf_NISNEES_times, [ddf_NISHist[time] for time in ddf_NISNEES_times], color='b', linewidth=1.5)
                                axes[8].plot(ddf_NISNEES_times, [ddf_NEESHist[time] for time in ddf_NISNEES_times], color='k', linewidth=1.5)
                            
                            if self.centralEstimator:
                                if targ.targetID in self.centralEstimator.estHist:
                                    trueHist = targ.hist
                                    estHist = self.centralEstimator.estHist[targ.targetID]
                                    covHist = self.centralEstimator.covarianceHist[targ.targetID]
                                    innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                                    innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                                    times = [time for time in time_vec.value if time in estHist]       
                                
                                # ERROR PLOTS
                                for i in range(6):
                                    axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='#228B22')
                                    axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')#, label='2 Sigma Bounds')
                                    axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')
                                    
                                #NO INNOVATION PLOTS FOR CENTRAL ESTIMATOR?
                            
                            # COLLECT LEGENDS REMOVING DUPLICATES
                            handles, labels = [], []
                            for ax in axes:
                                for handle, label in zip(*ax.get_legend_handles_labels()):
                                    if label not in labels:  # Avoid duplicates in the legend
                                        handles.append(handle)
                                        labels.append(label)
                            
                            
                            min_time = min(targ.hist.keys())
                            max_time = max(targ.hist.keys())

                            for ax in axes:
                                ax.set_xlim(min_time, max_time)

                            # SATELLITE COLORS
                            satColor = sat.color
                            
                            # Create a Patch object for the satellite
                            satPatch = Patch(color=satColor, label=sat.name)
                            
                            # Add the Patch object to the handles and labels
                            handles.append(satPatch)
                            labels.append(sat.name)
                            
                            # ALSO ADD CENTRAL IF FLAG IS SET
                            if self.centralEstimator:
                                # Create a Patch object for the central estimator
                                centralPatch = Patch(color='#228B22', label='Central Estimator')
                                
                                # Add the Patch object to the handles and labels
                                handles.append(centralPatch)
                                labels.append('Central Estimator')
                                
                            # CREATE A DDF PATCH
                            ddfPatch = Patch(color='#DC143C', label='DDF Estimator')
                            handles.append(ddfPatch)
                            labels.append('DDF Estimator')
                            
                            # ADD LEGEND
                            fig.legend(handles, labels, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
                            plt.tight_layout()
                                
                            # SAVE PLOT
                            if savePlot:
                                filePath = os.path.dirname(os.path.realpath(__file__))
                                plotPath = os.path.join(filePath, 'plots')
                                os.makedirs(plotPath, exist_ok=True)
                                if saveName is None:
                                    plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_total_results.png"), dpi=300)
                                    return
                                plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_total_results.png"), dpi=300)

                            plt.close()
                            
                                
    
# Plots all of the results to the user.
    # def plotResults(self, time_vec, savePlot, saveName):
    #     # Close the sim plot so that sizing of plots is good
    #     plt.close('all')
    #     state_labels = ['X [km]', 'Vx [km/min]', 'Y [km]', 'Vy [km/min]', 'Z [km]', 'Vz [km/min]']
    #     meas_labels = ['In Track [deg]', 'Cross Track [deg]']

    # # FOR EACH TARGET and EACH SATELLITE MAKE A PLOT
    #     for targ in self.targs:
    #         for sat in self.sats:
    #             if targ.targetID in sat.targetIDs:
                    
    #                 # Create a figure
    #                 fig = plt.figure(figsize=(15, 8))
                    
    #                 # Subtitle with the name of the target
    #                 fig.suptitle(f"{targ.name}, {sat.name} Estimation Error and Innovation Plots", fontsize=14)
    #                 # Create a GridSpec object with 3 rows and 6 columns (3 col x y z, 3 col vx vy vz , 2 col alpha beta)
    #                 gs = gridspec.GridSpec(3, 6)
                    
    #                 # Collect axes in a list of lists for easy access
    #                 axes = []
    #                 # Create subplots in the first two rows (3 2-columns spanning the width)
    #                 for i in range(3):  # 2 rows * 6 columns = 3
    #                     ax = fig.add_subplot(gs[0, 2*i:2*i + 2])
    #                     ax.grid(True)
    #                     axes.append(ax)
                        
    #                     ax = fig.add_subplot(gs[1, 2*i:2*i + 2])
    #                     ax.grid(True)
    #                     axes.append(ax)
                        
    #                 # Create subplots in the third row (3 columns spanning the width)
    #                 for i in range(2):  # 1 row * 2 columns = 2
    #                     ax = fig.add_subplot(gs[2, 3*i:3*i+3])
    #                     ax.grid(True)
    #                     axes.append(ax)
    #                 # Set the labels for the subplots
    #                 # Do the error vs covariance plots on the first row:
    #                 for i in range(6):
    #                     axes[i].set_xlabel("Time [min]")
    #                     axes[i].set_ylabel(f"Error in {state_labels[i]}")
    #                 # Do the innovation vs innovation covariance plots on the third row:
    #                 for i in range(2):  # Note: only 2 plots in the third row
    #                     axes[6 + i].set_xlabel("Time [min]")
    #                     axes[6 + i].set_ylabel(f"Innovation in {meas_labels[i]}")
                        
    #             # FOR EACH SATELLITE, EXTRACT ALL DATA for independent estimator and ddf estimator
    #                 satColor = sat.color
    #                 trueHist = targ.hist
    #                 estHist = sat.indeptEstimator.estHist[targ.targetID]
    #                 covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
    #                 innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
    #                 innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                        
    #                 ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
    #                 ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
    #                 ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
    #                 ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                        
    #                 estHist = sat.indeptEstimator.estHist[targ.targetID]
    #                 covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
    #                 innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
    #                 innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                        
    #                 ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
    #                 ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
    #                 ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
    #                 ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                        
    #                 times = [time for time in time_vec.value if time in estHist]
    #                 ddf_times = [time for time in time_vec.value if time in ddf_estHist]
    #                 ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
  
               
    #             # ERROR PLOTS
    #                 for i in range(6):
    #                     if times:
    #                         axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate'
    #                         axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
    #                         axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                        
    #                     if ddf_times:
    #                         axes[i].plot(ddf_times, [ddf_estHist[time][i] - trueHist[time][i] for time in ddf_times], color='#DC143C') # Error')
    #                         axes[i].plot(ddf_times, [2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed')# label='DDF 2 Sigma Bounds')
    #                         axes[i].plot(ddf_times, [-2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='#DC143C', linestyle='dashed')
                                
    #             # INNOVATION PLOTS
    #                 for i in range(2):  # Note: only 3 plots in the third row
    #                     if times:
    #                         axes[6 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor, linewidth=2.5)#, label='Local Estimate')
    #                         axes[6 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
    #                         axes[6 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                        
    #                     if ddf_innovation_times:
    #                         axes[6 + i].plot(ddf_innovation_times, [ddf_innovationHist[time][i] for time in ddf_innovation_times], color='#DC143C')#, label='DDF Estimate')
    #                         axes[6 + i].plot(ddf_innovation_times, [2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')#, label='DDF 2 Sigma Bounds')
    #                         axes[6 + i].plot(ddf_innovation_times, [-2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='#DC143C', linestyle='dashed')
        
    #                 if self.centralEstimator:
    #                     if targ.targetID in self.centralEstimator.estHist:
    #                         trueHist = targ.hist
    #                         estHist = self.centralEstimator.estHist[targ.targetID]
    #                         covHist = self.centralEstimator.covarianceHist[targ.targetID]
    #                         innovationHist = self.centralEstimator.innovationHist[targ.targetID]
    #                         innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
    #                         times = [time for time in time_vec.value if time in estHist]       
                        
    #                     # ERROR PLOTS
    #                     for i in range(6):
    #                         axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='#228B22')
    #                         axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')#, label='2 Sigma Bounds')
    #                         axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='#228B22', linestyle='dashed')
                            
    #                     #NO INNOVATION PLOTS FOR CENTRAL ESTIMATOR?
                    
    #                 # COLLECT LEGENDS REMOVING DUPLICATES
    #                 handles, labels = [], []
    #                 for ax in axes:
    #                     for handle, label in zip(*ax.get_legend_handles_labels()):
    #                         if label not in labels:  # Avoid duplicates in the legend
    #                             handles.append(handle)
    #                             labels.append(label)
                    
                    
    #                 min_time = min(targ.hist.keys())
    #                 max_time = max(targ.hist.keys())

    #                 for ax in axes:
    #                     ax.set_xlim(min_time, max_time)

    #                 # SATELLITE COLORS
    #                 satColor = sat.color
                    
    #                 # Create a Patch object for the satellite
    #                 satPatch = Patch(color=satColor, label=sat.name)
                    
    #                 # Add the Patch object to the handles and labels
    #                 handles.append(satPatch)
    #                 labels.append(sat.name)
                    
    #                 # ALSO ADD CENTRAL IF FLAG IS SET
    #                 if self.centralEstimator:
    #                     # Create a Patch object for the central estimator
    #                     centralPatch = Patch(color='#228B22', label='Central Estimator')
                        
    #                     # Add the Patch object to the handles and labels
    #                     handles.append(centralPatch)
    #                     labels.append('Central Estimator')
                        
    #                 # CREATE A DDF PATCH
    #                 ddfPatch = Patch(color='#DC143C', label='DDF Estimator')
    #                 handles.append(ddfPatch)
    #                 labels.append('DDF Estimator')
                    
    #                 # ADD LEGEND
    #                 fig.legend(handles, labels, loc='lower right', ncol=3, bbox_to_anchor=(1, 0))
    #                 plt.tight_layout()
                        
    #                 # SAVE PLOT
    #                 if savePlot:
    #                     filePath = os.path.dirname(os.path.realpath(__file__))
    #                     plotPath = os.path.join(filePath, 'plots')
    #                     os.makedirs(plotPath, exist_ok=True)
    #                     if saveName is None:
    #                         plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_results.png"), dpi=300)
    #                         return
    #                     plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_results.png"), dpi=300)

    #                 plt.close()


# Returns the NEES and NIS data for the simulation
    def collectData(self):
        # We want to return the NEES and NIS data for the simulation in an easy to read format
        # Create a dictionary of targetIDs
        data = {targetID: defaultdict(dict) for targetID in (targ.targetID for targ in self.targs)}
        # Now for each targetID, make a dictionary for each satellite:
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # Extract the data
                    data[targ.targetID][sat.name] = {'NEES': sat.indeptEstimator.neesHist[targ.targetID], 'NIS': sat.indeptEstimator.nisHist[targ.targetID]}
                    data[targ.targetID][sat.name] = {'NEES': sat.indeptEstimator.neesHist[targ.targetID], 'NIS': sat.indeptEstimator.nisHist[targ.targetID]}

            # If central estimator is used, also add that data
            if self.centralEstimator:
                if targ.targetID in self.centralEstimator.neesHist:
                    data[targ.targetID]['Central'] = {'NEES': self.centralEstimator.neesHist[targ.targetID], 'NIS': self.centralEstimator.nisHist[targ.targetID]}
                if targ.targetID in self.centralEstimator.neesHist:
                    data[targ.targetID]['Central'] = {'NEES': self.centralEstimator.neesHist[targ.targetID], 'NIS': self.centralEstimator.nisHist[targ.targetID]}

        return data


# Convert images to a gif
    # Save in the img struct
    def convert_imgs(self):
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        self.imgs.append(img)
        
# Render the gif, saving it to a file
    # File is the name of the file to save the gif to
    # Frame duration is the time between each frame in the gif (in ms???)
    def render_gif(self, fileType, saveName, filePath = os.path.dirname(os.path.realpath(__file__)), fps = 10):
        frame_duration = 1000/fps  # in ms
        if fileType == 'satellite_simulation':
            file = os.path.join(filePath, 'gifs', f'{saveName}_satellite_sim.gif')
            with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                for img in self.imgs:
                    writer.append_data(img)
        
        if fileType == 'uncertainity_ellipse':
            for sat in self.sats:
                for targ in self.targs:
                    if targ.targetID in sat.targetIDs:
                        file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_uncertainty_ellipse.gif")
                        with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_UC[targ.targetID][sat]:
                                writer.append_data(img)
                        
                        ddf_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_uncertainty_ellipse_DDF.gif")
                        with imageio.get_writer(ddf_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_UC_DDF[targ.targetID][sat]:
                                writer.append_data(img)
                                
                        both_file = os.path.join(filePath, 'gifs', f"{saveName}_{targ.name}_{sat.name}_uncertainty_ellipse_both.gif")
                        with imageio.get_writer(both_file, mode='I', duration=frame_duration) as writer:
                            for img in self.imgs_UC_LC_DDF[targ.targetID][sat]:
                                writer.append_data(img)


    def log_data(self, time_vec, saveName, filePath = os.path.dirname(os.path.realpath(__file__))):
        # # Delete all files already within the data folder
        # for file in os.listdir(filePath + '/data/'):
        #     os.remove(filePath + '/data/' + file)
        
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
                    innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                    
                    ddf_times = sat.ddfEstimator.estHist[targ.targetID].keys()
                    ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                    ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                    
                    ddf_innovation_times = sat.ddfEstimator.innovationHist[targ.targetID].keys()
                    ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                    ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                    
                    # File Name
                    filename = f"{filePath}/data/{saveName}_{targ.name}_{sat.name}.csv"
                    
                    # Format the data and write it to the file
                    self.format_data(filename, targ.targetID, times, sat_hist, trueHist, sat_measHistTimes, sat_measHist, estTimes, estHist, covHist, innovationHist, innovationCovHist, ddf_times, ddf_estHist, ddf_covHist, ddf_innovation_times, ddf_innovationHist, ddf_innovationCovHist)
                    

    def format_data(self, filename, targetID, times, sat_hist, trueHist, sat_measHistTimes, sat_measHist, estTimes, estHist, covHist, innovationHist, innovationCovHist,
                    ddf_times, ddf_estHist, ddf_covHist, ddf_innovation_times, ddf_innovationHist, ddf_innovationCovHist):
        
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
                'Cov_xx', 'Cov_vxvx', 'Cov_yy', 'Cov_vyvy', 'Cov_zz', 'Cov_vzvz',
                'Innovation_ITA', 'Innovation_CTA', 'InnovationCov_ITA', 'InnovationCov_CTA',
                'DDF_Est_x', 'DDF_Est_vx', 'DDF_Est_y', 'DDF_Est_vy', 'DDF_Est_z', 'DDF_Est_vz',
                'DDF_Cov_xx', 'DDF_Cov_vxvx', 'DDF_Cov_yy', 'DDF_Cov_vyvy', 'DDF_Cov_zz', 'DDF_Cov_vzvz',
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
                    row += format_list(innovationHist[time])
                    row += format_list(np.diag(innovationCovHist[time]))
                    
                if time in ddf_times:
                    row += format_list(ddf_estHist[time])
                    row += format_list(np.diag(ddf_covHist[time]))
                    
                if time in ddf_innovation_times:
                    row += format_list(ddf_innovationHist[time])
                    row += format_list(np.diag(ddf_innovationCovHist[time]))
                    
                writer.writerow(row)

    def plot_UncertaintyEllipse(self):
        def set_axis_limits(ax, est_pos, radii, margin=50.0):
            min_vals = est_pos - radii - margin
            max_vals = est_pos + radii + margin
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            ax.set_zlim(min_vals[2], max_vals[2])
            ax.set_box_aspect([1, 1, 1])

        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    for time in sat.indeptEstimator.estHist[targ.targetID].keys():
                        true_pos = targ.hist[time][[0, 2, 4]]
                        est_pos = np.array([sat.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                        cov_matrix = sat.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                        
                        err = np.linalg.norm(true_pos - est_pos)
                        
                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

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

                        set_axis_limits(ax, est_pos, radii)

                        ax.plot_surface(x_transformed, y_transformed, z_transformed, color='b', alpha=0.3)
                        ax.scatter(est_pos[0], est_pos[1], est_pos[2], color='r', marker='o')
                        ax.scatter(true_pos[0], true_pos[1], true_pos[2], color='g', marker='o')
                        ax.quiver(est_pos[0], est_pos[1], est_pos[2], LOS_vec[0], LOS_vec[1], LOS_vec[2], color='k', length=10, normalize=True)
                        ax.text2D(0.05, 0.95, f"Time: {time:.2f}, Error: {err:.2f} [km]", transform=ax.transAxes)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title(f"{targ.name}, {sat.name} Gaussian Uncertainty Ellipsoids")
                        ax.view_init(elev=10, azim=30)

                        ios = io.BytesIO()
                        fig.savefig(ios, format='raw')
                        ios.seek(0)
                        w, h = fig.canvas.get_width_height()
                        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]

                        self.imgs_UC[targ.targetID][sat].append(img)

                        ax.cla()  # Clear the plot for the next iteration
                    
                    plt.close(fig)
                    
                    
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    for time in sat.ddfEstimator.estHist[targ.targetID].keys():
                        true_pos = targ.hist[time][[0, 2, 4]]
                        est_pos = np.array([sat.ddfEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                        cov_matrix = sat.ddfEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                        
                        err = np.linalg.norm(true_pos - est_pos)
                        
                        LOS_vec = -sat.orbitHist[time] / np.linalg.norm(sat.orbitHist[time])

                        
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

                        set_axis_limits(ax, est_pos, radii)

                        ax.plot_surface(x_transformed, y_transformed, z_transformed, color='b', alpha=0.3)
                        ax.scatter(est_pos[0], est_pos[1], est_pos[2], color='r', marker='o')
                        ax.scatter(true_pos[0], true_pos[1], true_pos[2], color='g', marker='o')
                        ax.quiver(est_pos[0], est_pos[1], est_pos[2], LOS_vec[0], LOS_vec[1], LOS_vec[2], color='k', length=10, normalize=True)

                        ax.text2D(0.05, 0.95, f"Time: {time:.2f}, Error: {err:.2f} [km]", transform=ax.transAxes)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_title(f"{targ.name}, {sat.name} DDF Gaussian Uncertainty Ellipsoids")
                        ax.view_init(elev=10, azim=30)

                        ios = io.BytesIO()
                        fig.savefig(ios, format='raw')
                        ios.seek(0)
                        w, h = fig.canvas.get_width_height()
                        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]

                        self.imgs_UC_DDF[targ.targetID][sat].append(img)

                        ax.cla()  # Clear the plot for the next iteration
                    
                    plt.close(fig)
        
        for targ in self.targs: # plot both uncertainity ellispes and ddf ellipses
            for sat1 in self.sats:
                for sat2 in self.sats:
                    if sat1 != sat2 and targ.targetID in sat1.targetIDs and targ.targetID in sat2.targetIDs:
                        sat1_times = sat1.indeptEstimator.estHist[targ.targetID].keys()
                        sat2_times = sat2.indeptEstimator.estHist[targ.targetID].keys()
                        times = [time for time in sat1_times if time in sat2_times]
                        
                        for time in times:
                            true_pos = targ.hist[time][[0, 2, 4]]
                            est_pos1 = np.array([sat1.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                            cov_matrix1 = sat1.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                            
                            est_pos2 = np.array([sat2.indeptEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                            cov_matrix2 = sat2.indeptEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                            
                            ddf_est_pos1 = np.array([sat1.ddfEstimator.estHist[targ.targetID][time][i] for i in [0, 2, 4]])
                            ddf_cov_matrix1 = sat1.ddfEstimator.covarianceHist[targ.targetID][time][[0, 2, 4]][:, [0, 2, 4]]
                            
                            err1 = np.linalg.norm(true_pos - est_pos1)
                            err2 = np.linalg.norm(true_pos - est_pos2)
                            err3 = np.linalg.norm(true_pos - ddf_est_pos1)
                            
                            LOS_vec1 = -sat1.orbitHist[time] / np.linalg.norm(sat1.orbitHist[time])
                            LOS_vec2 = -sat2.orbitHist[time] / np.linalg.norm(sat2.orbitHist[time])
                            
                            eigenvalues1, eigenvectors1 = np.linalg.eigh(cov_matrix1)
                            radii1 = np.sqrt(eigenvalues1)
                            
                            eigenvalues2, eigenvectors2 = np.linalg.eigh(cov_matrix2)
                            radii2 = np.sqrt(eigenvalues2)
                            
                            eigenvalues3, eigenvectors3 = np.linalg.eigh(ddf_cov_matrix1)
                            radii3 = np.sqrt(eigenvalues3)
                            
                            u = np.linspace(0, 2 * np.pi, 100)
                            v = np.linspace(0, np.pi, 100)
                            
                            x1 = radii1[0] * np.outer(np.cos(u), np.sin(v))
                            y1 = radii1[1] * np.outer(np.sin(u), np.sin(v))
                            z1 = radii1[2] * np.outer(np.ones_like(u), np.cos(v))
                            
                            ellipsoid_points1 = np.array([x1.flatten(), y1.flatten(), z1.flatten()]).T
                            transformed_points1 = ellipsoid_points1 @ eigenvectors1.T + est_pos1
                            
                            x1_transformed = transformed_points1[:, 0].reshape(x1.shape)
                            y1_transformed = transformed_points1[:, 1].reshape(y1.shape)
                            z1_transformed = transformed_points1[:, 2].reshape(z1.shape)
                            
                            x2 = radii2[0] * np.outer(np.cos(u), np.sin(v))
                            y2 = radii2[1] * np.outer(np.sin(u), np.sin(v))
                            z2 = radii2[2] * np.outer(np.ones_like(u), np.cos(v))
                         
                            ellipsoid_points2 = np.array([x2.flatten(), y2.flatten(), z2.flatten()]).T
                            transformed_points2 = ellipsoid_points2 @ eigenvectors2.T + est_pos2
                            
                            x2_transformed = transformed_points2[:, 0].reshape(x2.shape)
                            y2_transformed = transformed_points2[:, 1].reshape(y2.shape)
                            z2_transformed = transformed_points2[:, 2].reshape(z2.shape)
                            
                            x3 = radii3[0] * np.outer(np.cos(u), np.sin(v))
                            y3 = radii3[1] * np.outer(np.sin(u), np.sin(v))
                            z3 = radii3[2] * np.outer(np.ones_like(u), np.cos(v))
                            
                            ellipsoid_points3 = np.array([x3.flatten(), y3.flatten(), z3.flatten()]).T
                            transformed_points3 = ellipsoid_points3 @ eigenvectors3.T + ddf_est_pos1
                            
                            x3_transformed = transformed_points3[:, 0].reshape(x3.shape)
                            y3_transformed = transformed_points3[:, 1].reshape(y3.shape)
                            z3_transformed = transformed_points3[:, 2].reshape(z3.shape)
                            
                            
                            ax.plot_surface(x1_transformed, y1_transformed, z1_transformed, color='b', alpha=0.3)
                            ax.plot_surface(x2_transformed, y2_transformed, z2_transformed, color='m', alpha=0.3)
                            ax.plot_surface(x3_transformed, y3_transformed, z3_transformed, color='c', alpha=0.3)
                            ax.scatter(est_pos1[0], est_pos1[1], est_pos1[2], color='b', marker='*')
                            ax.scatter(est_pos2[0], est_pos2[1], est_pos2[2], color='m', marker='*')
                            ax.scatter(ddf_est_pos1[0], ddf_est_pos1[1], ddf_est_pos1[2], color='c', marker='*')
                            ax.scatter(true_pos[0], true_pos[1], true_pos[2], color='g', marker='x')
                            ax.quiver(est_pos1[0], est_pos1[1], est_pos1[2], LOS_vec1[0], LOS_vec1[1], LOS_vec1[2], color='k', length=20, normalize=True)
                            ax.quiver(est_pos2[0], est_pos2[1], est_pos2[2], LOS_vec2[0], LOS_vec2[1], LOS_vec2[2], color='k', length=20, normalize=True)
                            
                            margin = 50.0
                            # Calculate min and max values for each position
                            min_vals1 = est_pos1 - radii1 - margin
                            max_vals1 = est_pos1 + radii1 + margin

                            min_vals2 = est_pos2 - radii2 - margin
                            max_vals2 = est_pos2 + radii2 + margin

                            min_vals3 = ddf_est_pos1 - radii3 - margin
                            max_vals3 = ddf_est_pos1 + radii3 + margin

                            # Determine the overall min and max values for each axis
                            min_vals = [
                                min(min_vals1[0], min_vals2[0], min_vals3[0]),
                                min(min_vals1[1], min_vals2[1], min_vals3[1]),
                                min(min_vals1[2], min_vals2[2], min_vals3[2])
                            ]

                            max_vals = [
                                max(max_vals1[0], max_vals2[0], max_vals3[0]),
                                max(max_vals1[1], max_vals2[1], max_vals3[1]),
                                max(max_vals1[2], max_vals2[2], max_vals3[2])
                            ]

                            # Set axis limits
                            ax.set_xlim(min_vals[0], max_vals[0])
                            ax.set_ylim(min_vals[1], max_vals[1])
                            ax.set_zlim(min_vals[2], max_vals[2])
                            ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
                            
                            
                            ax.text2D(0.05, 0.95, f"Time: {time:.2f}: Error: Sat1={err1:.2f}, Sat2={err2:.2f},  DDF={err3:.2f} [km]", transform=ax.transAxes)
                            ax.set_xlabel('X')
                            ax.set_ylabel('Y')
                            ax.set_zlabel('Z')
                            ax.set_title(f"{targ.name}, {sat.name} Gaussian Uncertainty Ellipsoids")
                            ax.view_init(elev=10, azim=30)

                            ios = io.BytesIO()
                            fig.savefig(ios, format='raw')
                            ios.seek(0)
                            w, h = fig.canvas.get_width_height()
                            img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]

                            self.imgs_UC_LC_DDF[targ.targetID][sat].append(img)

                            ax.cla()  # Clear the plot for the next iteration
                        
                        plt.close(fig)