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
        self.ax.set_xlim([-8000, 8000])
        self.ax.set_ylim([-8000, 8000])
        self.ax.set_zlim([-8000, 8000])
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
    def simulate(self, time_vec, pause_step = 0.1, display = False):
        
        # Initalize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time # Get delta time to propagate, works because propagate func increases time after first itr
        
        # Propagate the satellites and environments position
            self.propagate(t_d)

        # Update the plot environment
            self.plot()

        # Save the current plot to the images list, for gif later
            self.convert_imgs()

            if display:
            # Display the plot in a animation
                plt.pause(pause_step) 
                plt.draw()

        # Save the data for each satellite to a csv file
        self.log_data()
        
        # self.plotResults(time_vec)
        self.plotBaselines(time_vec)

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

        # Collect measurements on any avaliable targets
            collectedFlag[satNum] = sat.collect_measurements(self.targs) # if any measurement was collected, will be true
            satNum += 1

        # Update Central Estimator on all targets if measurments were collected
        # if any(collectedFlag == 1) and self.centralEstimator:
        #     for targ in self.targs:
        #         # Run the central estimator on the measurements
        #         centralEstimate = self.centralEstimator.EKF(self.sats, targ, time_val) 
   
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
            self.ax.scatter(x, y, z, s=20, color = targ.color, label=targ.name, marker='*')
            
        self.ax.legend()

    # PLOT COMMUNICATION STRUCTURE
        if self.comms.displayStruct:
            for edge in self.comms.G.edges:
                sat1 = edge[0]
                sat2 = edge[1]
                x1, y1, z1 = sat1.orbit.r.value
                x2, y2, z2 = sat2.orbit.r.value
                self.ax.plot([x1, x2], [y1, y2], [z1, z2], color='k', linestyle='dashed', linewidth=1)

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


# Plots all of the results to the user.
# Using bearings only measurement now
    def plotResults(self, time_vec, central = False):
        # Close the sim plot so that sizing of plots is good
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/s]', 'Y [km]', 'Vy [km/s]', 'Z [km]', 'Vz [km/s]']
        meas_labels = ['In Track [deg]', 'Cross Track [deg]']

    # FOR EACH TARGET MAKE A PLOT
        for targ in self.targs:
            # Create a figure
            fig = plt.figure(figsize=(15, 8))
            # Subtitle with the name of the target
            fig.suptitle(f"{targ.name} State, Error, and Innovation Plots", fontsize=14)

            # Create a GridSpec object with 3 rows and 6 columns
            gs = gridspec.GridSpec(3, 6)

            # Collect axes in a list of lists for easy access
            axes = []

            # Create subplots in the first two rows (6 columns each)
            for i in range(12):  # 2 rows * 6 columns = 12
                ax = fig.add_subplot(gs[i // 6, i % 6])
                axes.append(ax)

            # Create subplots in the third row (3 columns spanning the width)
            for i in range(2):  # 1 row * 2 columns = 2
                ax = fig.add_subplot(gs[2, 3*i:3*i+3])
                axes.append(ax)

            # Set the labels for the subplots
            for i in range(6):
                axes[i].set_xlabel("Time [min]")
                axes[i].set_ylabel(f"{state_labels[i]} measurements")

            # Do the error vs covariance plots on the second row:
            for i in range(6):
                axes[6 + i].set_xlabel("Time [min]")
                axes[6 + i].set_ylabel(f"Error in {state_labels[i]}")

            # Do the innovation vs innovation covariance plots on the third row:
            for i in range(2):  # Note: only 2 plots in the third row
                axes[12 + i].set_xlabel("Time [min]")
                axes[12 + i].set_ylabel(f"Innovation in {meas_labels[i]}")

    # FOR EACH SATELLITE, ADD DATA
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # EXTRACT ALL DATA
                    satColor = sat.color
                    trueHist = targ.hist
                    estHist = sat.estimator.estHist[targ.targetID]
                    covHist = sat.estimator.covarianceHist[targ.targetID]
                    innovationHist = sat.estimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.estimator.innovationCovHist[targ.targetID]
                    times = [time for time in time_vec.value if time in estHist]
                
                    # MEASUREMENT PLOTS
                    for i in range(6):
                        axes[i].scatter(times, [trueHist[time][i] for time in times], color='k', label='Truth', marker='o', s=15)
                        axes[i].plot(times, [estHist[time][i] for time in times], color=satColor, label='Kalman Estimate')

                    # ERROR PLOTS
                    for i in range(6):
                        axes[6 + i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, linestyle='dashed', label='Error')
                        axes[6 + i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dotted', label='2 Sigma Bounds')
                        axes[6 + i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')

                    # INNOVATION PLOTS
                    for i in range(2):  # Note: only 3 plots in the third row
                        axes[12 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor, label='Kalman Estimate')
                        axes[12 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted', label='2 Sigma Bounds')
                        axes[12 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')

        # IF CENTRAL ESTIMATOR FLAG IS SET, ALSO PLOT THAT:
        # USE COLOR PINK FOR CENTRAL ESTIMATOR
            if central:
                trueHist = targ.hist
                estHist = self.centralEstimator.estHist[targ.targetID]
                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                times = [time for time in time_vec.value if time in estHist]

                # MEASUREMENT PLOTS
                for i in range(6):
                    axes[i].scatter(times, [trueHist[time][i] for time in times], color='k', label='Truth', marker='o', s=15)
                    axes[i].plot(times, [estHist[time][i] for time in times], color='purple', label='Central Estimate')

                # ERROR PLOTS
                for i in range(6):
                    axes[6 + i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='purple', linestyle='dashed', label='Error')
                    axes[6 + i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dotted', label='2 Sigma Bounds')
                    axes[6 + i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dotted')

        # COLLECT LEGENDS REMOVING DUPLICATES
            handles, labels = [], []
            for ax in axes:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:  # Avoid duplicates in the legend
                        handles.append(handle)
                        labels.append(label)

        # AND SATELLITE COLORS
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # EXTRACT ALL DATA
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

        # ADD LEGEND
            fig.legend(handles, labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.01))
            plt.tight_layout()
            plt.show()


# Plots all of the results to the user.
    def plotResults_old(self, time_vec, central = False):
        # Close the sim plot so that sizing of plots is good
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/s]', 'Y [km]', 'Vy [km/s]', 'Z [km]', 'Vz [km/s]']

    # FOR EACH TARGET MAKE A PLOT
        for targ in self.targs:
            # Create a figure
            fig = plt.figure(figsize=(15, 8))
            # Subtitle with the name of the target
            fig.suptitle(f"{targ.name} State, Error, and Innovation Plots", fontsize=14)

            # Create a GridSpec object with 3 rows and 6 columns
            gs = gridspec.GridSpec(3, 6)

            # Collect axes in a list of lists for easy access
            axes = []

            # Create subplots in the first two rows (6 columns each)
            for i in range(12):  # 2 rows * 6 columns = 12
                ax = fig.add_subplot(gs[i // 6, i % 6])
                axes.append(ax)

            # Create subplots in the third row (3 columns spanning the width)
            for i in range(3):  # 1 row * 3 columns = 3
                ax = fig.add_subplot(gs[2, 2*i:2*i+2])
                axes.append(ax)

            # Set the labels for the subplots
            for i in range(6):
                axes[i].set_xlabel("Time [min]")
                axes[i].set_ylabel(f"{state_labels[i]} measurements")

            # Do the error vs covariance plots on the second row:
            for i in range(6):
                axes[6 + i].set_xlabel("Time [min]")
                axes[6 + i].set_ylabel(f"Error in {state_labels[i]}")

            # Do the innovation vs innovation covariance plots on the third row:
            for i in range(3):  # Note: only 3 plots in the third row
                axes[12 + i].set_xlabel("Time [min]")
                axes[12 + i].set_ylabel(f"Innovation in {state_labels[i*2]}")

    # FOR EACH SATELLITE, ADD DATA
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # EXTRACT ALL DATA
                    satColor = sat.color
                    trueHist = targ.hist
                    measHist = sat.estimator.measHist[targ.targetID]
                    estHist = sat.estimator.estHist[targ.targetID]
                    covHist = sat.estimator.covarianceHist[targ.targetID]
                    innovationHist = sat.estimator.innovationHist[targ.targetID]
                    innovationCovHist = sat.estimator.innovationCovHist[targ.targetID]
                    times = [time for time in time_vec.value if time in estHist]
                
                    # MEASUREMENT PLOTS
                    for i in range(6):
                        axes[i].scatter(times, [trueHist[time][i] for time in times], color='k', label='Truth', marker='o', s=15)
                        # if i % 2 == 0:
                            # axes[i].scatter(times, [measHist[time][i // 2] for time in times], color=satColor, label='Measurement', marker='x', s=10)
                        axes[i].plot(times, [estHist[time][i] for time in times], color=satColor, label='Kalman Estimate')

                    # ERROR PLOTS
                    for i in range(6):
                        axes[6 + i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color=satColor, linestyle='dashed', label='Error')
                        axes[6 + i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dotted', label='2 Sigma Bounds')
                        axes[6 + i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')

                    # INNOVATION PLOTS
                    for i in range(2):  # Note: only 3 plots in the third row
                        axes[12 + i].plot(times, [innovationHist[time][i] for time in times], color=satColor, label='Kalman Estimate')
                        axes[12 + i].plot(times, [2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted', label='2 Sigma Bounds')
                        axes[12 + i].plot(times, [-2 * np.sqrt(innovationCovHist[time][i][i]) for time in times], color=satColor, linestyle='dotted')

        # IF CENTRAL ESTIMATOR FLAG IS SET, ALSO PLOT THAT:
        # USE COLOR PINK FOR CENTRAL ESTIMATOR
            if central:
                trueHist = targ.hist
                estHist = self.centralEstimator.estHist[targ.targetID]
                covHist = self.centralEstimator.covarianceHist[targ.targetID]
                innovationHist = self.centralEstimator.innovationHist[targ.targetID]
                innovationCovHist = self.centralEstimator.innovationCovHist[targ.targetID]
                times = [time for time in time_vec.value if time in estHist]

                # MEASUREMENT PLOTS
                for i in range(6):
                    axes[i].scatter(times, [trueHist[time][i] for time in times], color='k', label='Truth', marker='o', s=15)
                    axes[i].plot(times, [estHist[time][i] for time in times], color='purple', label='Central Estimate')

                # ERROR PLOTS
                for i in range(6):
                    axes[6 + i].plot(times, [estHist[time][i] - trueHist[time][i] for time in times], color='purple', linestyle='dashed', label='Error')
                    axes[6 + i].plot(times, [2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dotted', label='2 Sigma Bounds')
                    axes[6 + i].plot(times, [-2 * np.sqrt(covHist[time][i][i]) for time in times], color='purple', linestyle='dotted')

        # COLLECT LEGENDS REMOVING DUPLICATES
            handles, labels = [], []
            for ax in axes:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:  # Avoid duplicates in the legend
                        handles.append(handle)
                        labels.append(label)

        # AND SATELLITE COLORS
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    # EXTRACT ALL DATA
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

        # ADD LEGEND
            fig.legend(handles, labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.01))
            plt.tight_layout()
            plt.show()


    def plotBaselines(self, time_vec):
        
        # Close the pripr sim plot so that the sizing of plots is good
        plt.close('all')
        state_labels = ['X [km]', 'Vx [km/s]', 'Y [km]', 'Vy [km/s]', 'Z [km]', 'Vz [km/s]']
        
        # For Each Target plot both of the satellites estimate and the central estimate
        for targ in self.targs:
            fig1, axs1 = plt.subplots(2, 3, figsize=(15, 6))
            fig1.suptitle(f"{targ.name} State Position and Error Plots", fontsize=16)
            
            fig2, axs2 = plt.subplots(2, 3, figsize=(15, 6))  # Create a 2x3 grid of subplots
            fig2.suptitle(f"{targ.name} State Velocity and Error Plots", fontsize=16)
            once = True
            for sat in self.sats:
                if targ.targetID in sat.targetIDs:
                    
                    # Get the Measurement, Estimate, True, and Covariance history for the target
                    measHist = sat.estimator.measHist[targ.targetID]
                    estHist = sat.estimator.estHist[targ.targetID]
                    covHist = sat.estimator.covarianceHist[targ.targetID]
                    trueHist = targ.hist
                    
                    # Filter out times that don't have estimates
                    times = [time for time in time_vec.value if time in estHist]
                    
                    for i in range(6):  # Create subplots for states and their errors
                        if i % 2 == 0: # Position
                            axs = axs1
                            fig = fig1
                        else: # Velocity
                            axs = axs2
                            fig = fig2

                        j = i // 2  # Adjust index for 2x3 grid

                        # Set Axis Labels for Both Plots
                        axs[0, j].set_xlabel("Time [min]")
                        axs[0, j].set_ylabel(f"State {state_labels[i]}")

                        axs[1, j].set_xlabel("Time")
                        axs[1, j].set_ylabel("Error / Covariance")

                        # Get all valid measurements, estimates, errors, and covariances
                        measurements = [measHist[time][round(i/2)] for time in times]
                        true_positions = [trueHist[time][i] for time in time_vec.value]
                        estimates = [estHist[time][i] for time in times]
                        errors = [estHist[time][i] - trueHist[time][i] for time in times]
                        covariances = [2 * np.sqrt(covHist[time][i][i]) for time in times]
                        
                        # Plot the estimate    
                        axs[0, j].plot(times, estimates, color=sat.color, label=f"Satellite: {sat.name} Estimate")

                        # Plot Position Measurements
                        if i % 2 == 0:
                            axs[0, j].scatter(times, measurements, color=sat.color, label=f"Satellite: {sat.name} Measurement")

                        # Plot Error and Covariance
                        axs[1, j].plot(times, errors, color=sat.color, label=f"Satellite: {sat.name} Error")
                        axs[1, j].plot(times, [-c for c in covariances], color=sat.color, linestyle='dashed', label=f"Satellite: {sat.name} 2 Sigma Bounds")


                        if once:
                            axs[0, j].plot(time_vec.value, true_positions, color='g')
                            axs[0, j].scatter(time_vec.value, true_positions, color='g', label='Truth')
                    
                    once = False
            
            # Plot Central Estimation
        centralEstimate = self.centralEstimator.estHist[targ.targetID]
        centralCov = self.centralEstimator.covarianceHist[targ.targetID]
        centralTimes = [time for time in time_vec.value if time in centralEstimate]
        for i in range(6):  # Create subplots for states and their errors
            if i % 2 == 0:
                axs = axs1
                fig = fig1
            else:
                axs = axs2
                fig = fig2
                        
            j = i // 2  # Adjust index for 2x3 grid

            centralEstimates = [centralEstimate[time][i] for time in centralTimes]
            centralCovs = [2 * np.sqrt(centralCov[time][i][i]) for time in centralTimes]
            centralErrors = [centralEstimate[time][i] - trueHist[time][i] for time in centralTimes]
                
            axs[0, j].plot(centralTimes, centralEstimates, color='k', label='Central Estimate')
                
            
            # Plot Error and Covariance    
            axs[1, j].plot(centralTimes, centralErrors, color='r', label='Central Error')
            axs[1, j].plot(centralTimes, centralCovs, color='b', linestyle='dashed', label='Central 2 Sigma Bounds')
            axs[1, j].plot(centralTimes, [-c for c in centralCovs], color='k', linestyle='dashed')
                    
            if i // 2 == 2:
                axs[0, j].legend()
                axs[1, j].legend()
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                
        plt.show()


    def plotResultsOld(self, time_vec):

        # Close the pripr sim plot so that the sizing of plots is good
        plt.close('all')
        state_labels = ['X', 'Vx', 'Y', 'Vy', 'Z', 'Vz']
            
        for sat in self.sats:
            # Get the targetID for that satellite and check estimator accuracy
             for targ in self.targs:
                 if targ.targetID in sat.targetIDs:
                     # Get the measurement history for that target
                        measHist = sat.estimator.measHist[targ.targetID]
                     # Get the estimate history for that target
                        estHist = sat.estimator.estHist[targ.targetID]
                    # Get the true position history for that target X, Y, Z and Vx, Vy, Vz
                        trueHist = targ.hist
                    # Get the covariance history for that target
                        covHist = sat.estimator.covarianceHist[targ.targetID]
                    # Get the times for which there are estimates
                        times = [time for time in time_vec.value if time in estHist]


                        fig, axs = plt.subplots(2, 6, figsize=(15, 6))  # Create a 2x6 grid of subplots
                        fig.suptitle(f"{sat.name} and {targ.name} State and Error Plots", fontsize=16)

                        for i in range(6):  # Create subplots for states and their errors
                            #axs[0, i].set_title(f"{sat.name} and {targ.name} Estimate and Truth for: {state_labels[i]}")
                            axs[0, i].set_xlabel("Time")
                            axs[0, i].set_ylabel(f"State {state_labels[i]}")

                            #axs[1, i].set_title(f"{sat.name} and {targ.name} Error and Covariance for: {state_labels[i]}")
                            axs[1, i].set_xlabel("Time")
                            axs[1, i].set_ylabel("Error / Covariance")

                            measurements = [measHist[time][round(i/2)] for time in times]
                            true_positions = [trueHist[time][i] for time in times]
                            estimates = [estHist[time][i] for time in times]
                            errors = [estHist[time][i] - trueHist[time][i] for time in times]
                            covariances = [2 * np.sqrt(covHist[time][i][i]) for time in times]

                            axs[0, i].plot(times, true_positions, color='k', label='Truth')
                            axs[0, i].plot(times, estimates, color='r', label='Estimate')
                            if i % 2 == 0:
                                axs[0, i].scatter(times, measurements, color='b', label='Measurement')
                                

                            axs[1, i].plot(times, errors, color='r', label='Error')
                            axs[1, i].plot(times, covariances, color='k', linestyle='dashed', label='2 Sigma Bounds')
                            axs[1, i].plot(times, [-c for c in covariances], color='k', linestyle='dashed')

                    # Add legends to all subplots
                        for ax in axs.flat:
                            ax.legend()

                        plt.tight_layout()  # Adjust the layout to prevent overlap
                        plt.show()  # Show all subplots at once            

        # NOW ALSO FOR EACH SATELLITE PLOT THE INNOVATION AND INNOVATION COVARIANCE
        # INNOVATION IS THE DIFFERENCE BETWEEN THE MEASUREMENT AND THE H*ESTIMATE
        # NEED 3 SUBPLOTS
                        meas_labels = ['X', 'Y', 'Z']
                        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                        fig.suptitle(f"{sat.name} and {targ.name} Innovation and Innovation Covariance Plots", fontsize=16)
                        for i in range(3):
                            ax = axes[i]
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Innovation & Innovation Covariance For " + meas_labels[i])

                            innovation_i = [sat.estimator.innovationHist[targ.targetID][time][i] for time in times] # innovation is the difference between the measurement and the H*estimate
                            innovationCovar_i = [sat.estimator.innovationCovHist[targ.targetID][time][i][i] for time in times] # want the diagonal of the covariance matrix

                            # Calculate the 2 sigma bounds for the innovation covariance
                            innovationCovar_i_2sigma = [2 * np.sqrt(cov) for cov in innovationCovar_i]

                            ax.plot(times, innovation_i, color='r', label='Innovation')
                            ax.plot(times, innovationCovar_i_2sigma, color='k', linestyle='dashed', label='Innovation Covar 2 Sigma Bounds')
                            ax.plot(times, [-cov for cov in innovationCovar_i_2sigma], color='k', linestyle='dashed')

                        for ax in axes.flat:
                            ax.legend()

                        plt.tight_layout()
                        plt.show()


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
