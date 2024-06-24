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
        self.ax.view_init(elev=30, azim=-10)
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

# Simulate the environment over a time range
    # Time range is a numpy array of time steps, must have poliastro units associated!
    # Pause step is the time to pause between each step, if displaying as animation
    # Display is a boolean, if true will display the plot as an animation
    def simulate(self, time_vec, pause_step = 0.1, savePlot = False, saveName = None, showSim = False):
        
        # Initalize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time # Get delta time to propagate, works because propagate func increases time after first itr
        
        # Propagate the satellites and environments position
            self.propagate(t_d)

        # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion(t_d)

        # Collect individual data measurements for satellites and then do data fusion
            self.data_fusion(t_d)

            if savePlot:
            # Update the plot environment
                self.plot()
                self.convert_imgs()
                self.convert_imgs()
                if showSim:
                    plt.pause(pause_step)
                    plt.draw()
        
        if savePlot:
            # Plot the results of the simulation.        
            self.plotResults(time_vec, central = True, savePlot = savePlot, saveName = saveName)

        return self.collectData()
        
# Collect measurements from all satellites and do data fusion
    def data_fusion(self, time_step):
        # Collect measurements on any avaliable targets
        # Create a dictionary of flags for each satellite. 
        # Dictionary contains true or false on if the given sat/targ combo collected a measurement
        collectedFlag = defaultdict(lambda: defaultdict(dict))
        for targ in self.targs:
            for sat in self.sats:
                # This will collect the bearing measurement, if avaliable, and also perform local estimation on the target
                # If a measurement is collected, the flag will be set to true
                collectedFlag[targ][sat] = sat.collect_measurements_and_filter(targ)

        # Now perform data fusion on the collected measurements
        # First check, did any new satellites get measurements?
        
        # Loop through all targets
        for targ in self.targs:
            # Check, did any of the satellies collect a measurement on this target?
            # If so, we should perform centralized fusion
            if any(collectedFlag[targ].values()):
                self.centralEstimator.EKF(self.sats, targ, self.time.to_value())

            for sat in self.sats:
                if sat.ddfEstimator:
                    for neighbor in self.comms.G.neighbors(sat):
                    # Check if the most recent estimate time is newer than the neighbor
                        # Check for if sat has any estimates yet
                        if len(sat.ddfEstimator.estHist[targ.targetID].keys()) == 0:
                            continue # If sat doesnt have estimates yet, nothing to send

                        # Most recent measurement time by satellite
                        satTime = max(sat.ddfEstimator.estHist[targ.targetID].keys())

                        # Check if the neighbor has any estimates yet
                        # TODO: IMPLCIT REQUIREMENT FOR TIMINGS TO BE KNOWN FOR NEIGHBOR
                        if len(neighbor.ddfEstimator.estHist[targ.targetID].keys()) == 0:
                            # If neighbor doesnt have estimates yet, should send the most recent estimate, initalizing the neighbors filter
                            self.comms.send_measurement(sat, neighbor, sat.ddfEstimator.estHist[targ.targetID][satTime], sat.ddfEstimator.covarianceHist[targ.targetID][satTime], targ.targetID, satTime)
                            continue

                        # Most recent measurement time by neighbor
                        neighborTime = max(neighbor.ddfEstimator.estHist[targ.targetID].keys())

<<<<<<< Updated upstream
                        # If the neighbor has an older estimate, send the most recent estimate
                        if satTime >= neighborTime:
=======
                        # If this neighbor has an old estimate, lets send our most recent estimate
                        if satTime > neighborTime:
>>>>>>> Stashed changes
                            self.comms.send_measurement(sat, neighbor, sat.ddfEstimator.estHist[targ.targetID][satTime], sat.ddfEstimator.covarianceHist[targ.targetID][satTime], targ.targetID, satTime)

                    # Now, each satellite will perform covariance intersection on the measurements sent to it
                    for sat in self.sats:
                        # For each satellite, perform CI on the measurements it received
                        sat.ddfEstimator.CI(sat, self.comms.G.nodes[sat], targ.targetID)

# Propagate the satellites over the time step  
    def propagate(self, time_step):
        
    # Update the current time
        self.time += time_step

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
            targ.hist[targ.time] = np.array([targ.pos[0], targ.vel[0], targ.pos[1], targ.vel[1], targ.pos[2], targ.vel[2]])


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

    # PLOT EARTH
        self.ax.plot_surface(self.x_earth, self.y_earth, self.z_earth, color = 'k', alpha=0.1)

    # FOR EACH SATELLITE, PLOTS
        for sat in self.sats:
        # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value
            self.ax.scatter(x, y, z, s=40, color = sat.color, label=sat.name)

        # Plot the visible projection of the satellite sensor
            points = sat.sensor.projBox
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color = sat.color, marker = 'x')
            
            box = np.array([points[0], points[3], points[1], points[2], points[0]])
            self.ax.add_collection3d(Poly3DCollection([box], facecolors=sat.color, linewidths=1, edgecolors=sat.color, alpha=.1))

    # FOR EACH TARGET, PLOTS
        for targ in self.targs:
        # Plot the current xyz location of the target
            x, y, z = targ.pos
            self.ax.scatter(x, y, z, s=20, marker = '*', color = targ.color, label=targ.name)
            
        self.ax.legend()
        
    # Set the axis limits
        self.ax.set_xlim([-15000, 15000])
        self.ax.set_ylim([-15000, 15000])
        self.ax.set_zlim([-15000, 15000])
        
        self.ax.view_init(elev=30, azim=-10)

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
        

# Plots all of the results to the user.
    def plotResults(self, time_vec, savePlot, saveName, central = False):
        # Close the sim plot so that sizing of plots is good
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/s]', 'Y [km]', 'Vy [km/s]', 'Z [km]', 'Vz [km/s]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]']

    # FOR EACH TARGET and EACH SATELLITE MAKE A PLOT
        for targ in self.targs:
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    
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
                        ax = fig.add_subplot(gs[2, 3*i:3*i+3])
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
            # FOR EACH SATELLITE, EXTRACT ALL DATA for independent estimator and ddf estimator
                    satColor = sat.color
                    trueHist = targ.hist
                    estHist = sat.indeptEstimator.estHist[targ.targetID]
                    covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                    innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                        
                    ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                    ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                    ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                    ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                        
                    estHist = sat.indeptEstimator.estHist[targ.targetID]
                    covHist = sat.indeptEstimator.covarianceHist[targ.targetID]
                    innovationHist = sat.indeptEstimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.indeptEstimator.innovationCovHist[targ.targetID]
                        
                    ddf_estHist = sat.ddfEstimator.estHist[targ.targetID]
                    ddf_covHist = sat.ddfEstimator.covarianceHist[targ.targetID]
                    ddf_innovationHist = sat.ddfEstimator.innovationHist[targ.targetID]
                    ddf_innovationCovHist = sat.ddfEstimator.innovationCovHist[targ.targetID]
                        
                    times = [time for time in time_vec.value if time in estHist]
                    ddf_times = [time for time in time_vec.value if time in ddf_estHist]
                    ddf_innovation_times = [time for time in time_vec.value if time in ddf_innovationHist]
                    # ERROR PLOTS
                    for i in range(6):
                        axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, label='Local', linewidth=2.5)#, label='Local Estimate'
                        axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
                        axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                        
                        axes[i].plot(ddf_times, [ddf_estHist[time][i] - trueHist[time][i] for time in ddf_times], color='r', label='DDF') # Error')
                        axes[i].plot(ddf_times, [2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='r', linestyle='dashed')# label='DDF 2 Sigma Bounds')
                        axes[i].plot(ddf_times, [-2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='r', linestyle='dashed')
                            
                        axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, label='Local', linewidth=2.5)#, label='Local Estimate'
                        axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)#, label='2 Sigma Bounds')
                        axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dashed', linewidth=2.5)
                        
                        axes[i].plot(ddf_times, [ddf_estHist[time][i] - trueHist[time][i] for time in ddf_times], color='r', label='DDF') # Error')
                        axes[i].plot(ddf_times, [2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='r', linestyle='dashed')# label='DDF 2 Sigma Bounds')
                        axes[i].plot(ddf_times, [-2 * np.sqrt(ddf_covHist[time][i][i]) for time in ddf_times], color='r', linestyle='dashed')
                            
                    # INNOVATION PLOTS
                    for i in range(2):  # Note: only 3 plots in the third row
                        axes[6 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor)#, label='Local Estimate')
                        axes[6 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')#, label='2 Sigma Bounds')
                        axes[6 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')
                        
                        axes[6 + i].plot(ddf_innovation_times, [ddf_innovationHist[time][i] for time in ddf_innovation_times], color='r')#, label='DDF Estimate')
                        axes[6 + i].plot(ddf_innovation_times, [2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='r', linestyle='dotted')#, label='DDF 2 Sigma Bounds')
                        axes[6 + i].plot(ddf_innovation_times, [-2 * np.sqrt(ddf_innovationCovHist[time][i][i]) for time in ddf_innovation_times], color='r', linestyle='dotted')
                        
                    # IF CENTRAL ESTIMATOR FLAG IS SET, ALSO PLOT THAT:
                    # USE COLOR PINK FOR CENTRAL ESTIMATOR
                    if central:
                        if targ.targetID in self.centralEstimator.estHist:
                            trueHist = targ.hist
                            estHist = self.centralEstimator.estHist[targ.targetID]
                            covHist = self.centralEstimator.covarianceHist[targ.targetID]
                            innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                            innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                            times = [time for time in time_vec.value if time in estHist]
                        # ERROR PLOTS
                        for i in range(6):
                            axes[i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='purple', label='Central')
                            axes[i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dashed')#, label='2 Sigma Bounds')
                            axes[i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dashed')
                    
                    # COLLECT LEGENDS REMOVING DUPLICATES
                    handles, labels = [], []
                    for ax in axes:
                        for handle, label in zip(*ax.get_legend_handles_labels()):
                            if label not in labels:  # Avoid duplicates in the legend
                                handles.append(handle)
                                labels.append(label)
                    
                    # SATELLITE COLORS
                    satColor = sat.color
                    
                    # Create a Patch object for the satellite
                    satPatch = Patch(color=satColor, label=sat.name)
                    
                    # Add the Patch object to the handles and labels
                    handles.append(satPatch)
                    labels.append(sat.name)
                    
                    # ALSO ADD CENTRAL IF FLAG IS SET
                    if central:
                        # Create a Patch object for the central estimator
                        centralPatch = Patch(color='purple', label='Central Estimator')
                        
                        # Add the Patch object to the handles and labels
                        handles.append(centralPatch)
                        labels.append('Central Estimator')
                        
                    # CREATE A DDF PATCH
                    ddfPatch = Patch(color='r', label='DDF Estimator')
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
                            plt.savefig(os.path.join(plotPath, f"{targ.name}_{sat.name}_results.png"), dpi=300)
                            return
                        plt.savefig(os.path.join(plotPath, f"{saveName}_{targ.name}_{sat.name}_results.png"), dpi=300)

                    plt.close()

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

# For each satellite, saves the measurement history of each target to a csv file:
    def log_data(self):
        # Make the file, current directory /data/satellite_name.csv
        filePath = os.path.dirname(os.path.realpath(__file__))
        # Delete all files already within the data folder
        for file in os.listdir(filePath + '/data/'):
            os.remove(filePath + '/data/' + file)
            
    # Loop through all satellites
        for sat in self.sats:
        # Loop through all targets for each satellite
            for targ in self.targs:
                if targ.targetID in sat.targetIDs:
                    with open(filePath + '/data/' + sat.name + '_' + targ.name + '.csv', mode='w') as file:
                        writer = csv.writer(file)
                        writer.writerow(sat.sensor.stringHeader)
                        for time, meas in sat.measurementHist[targ.targetID].items():
                            # combine time into the measurment array
                            combine = [time] + list(meas)
                            writer.writerow(combine)

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
    def render_gif(self, fileName = '/satellite_orbit.gif', filePath = os.path.dirname(os.path.realpath(__file__)), fps = 10):
        frame_duration = 1000/fps  # in ms
        file = os.path.join(filePath, fileName)
        with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
            for img in self.imgs:
                writer.append_data(img)
