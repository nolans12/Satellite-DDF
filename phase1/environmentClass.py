from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment: 
    def __init__(self, sats, targs, comms, centralEstimator = None):

    # If a central estimator is passed, use it
        if centralEstimator:
            self.estimator = centralEstimator

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

    # Plot Earth
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
            self.ax.scatter(x, y, z, s=20, color = targ.color, label=targ.name)
            
        self.ax.legend()

    # PLOT COMMUNICATION STRUCTURE
        if self.comms.displayStruct:
            for edge in self.comms.G.edges:
                sat1 = edge[0]
                sat2 = edge[1]
                x1, y1, z1 = sat1.orbit.r.value
                x2, y2, z2 = sat2.orbit.r.value
                self.ax.plot([x1, x2], [y1, y2], [z1, z2], color='k', linestyle='dashed', linewidth=1)

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

        # Propagate the satellites
        for sat in self.sats:
            
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)

            # Update the communication network for the new sat position:
            self.comms.make_edges(self.sats)

            # Collect measurements on any avaliable targets
            sat.collect_measurements(self.targs)

            # Update the history of the orbit
            sat.orbitHist[sat.time] = sat.orbit.r.value # history of sat time and xyz position


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


# Plot the results of the simulation: 
#   For each satellite
#       For each target of a satellite
#               Plot state estimate of the target and its real position
#               Plot the difference between the estimate and the real position
#                   Include the 2 sigma bounds: X_i = 2*sqrt(P_ii) for all states

    def plotResults(self, time_vec):
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
        
        
        #                 for i in range(6):  # Create figures for states and their errors
        #                     plt.figure(i)
        #                     plt.title(f"{sat.name} and {targ.name} Estimate and Truth for: {state_labels[i]}")
        #                     plt.xlabel("Time")
        #                     plt.ylabel(f"State {state_labels[i]}")
                            
        #                     plt.figure(i + 6)
        #                     plt.title(f"{sat.name} and {targ.name} Error and Covariance for: {state_labels[i]}")
        #                     plt.xlabel("Time")
        #                     plt.ylabel("Error / Covariance")
                        
                      
        #                     true_positions = [trueHist[time][i] for time in times]
        #                     estimates = [estHist[time][i] for time in times]
        #                     errors = [estHist[time][i] - trueHist[time][i] for time in times]
        #                     covariances = [2 * np.sqrt(covHist[time][i][i]) for time in times]
                            
        #                     plt.figure(i)
        #                     plt.plot(times, true_positions, color='k', label='Truth')
        #                     plt.plot(times, estimates, color='r', label='Estimate')
                            
        #                     plt.figure(i + 6)
        #                     plt.plot(times, errors, color='r', label='Error')
        #                     plt.plot(times, covariances, color='k', linestyle='dashed', label='2 Sigma Bounds')
        #                     plt.plot(times, [-c for c in covariances], color='k', linestyle='dashed')
        
        # for i in range(12):
        #     plt.figure(i)
        #     plt.legend()
        
        # plt.show()

     
  
            
           
        
        # # Create empty array for each target to store estimates:
        # targ_est = []

        # # Now, loop through time, and for each time, plot the estimates of each satellite
        # for t in time:
        #     # Clear the plot
        #     plt.clf()
        #     plt.xlim([-300, 300])
        #     plt.ylim([-300, 300])
        #     plt.xlabel('X (km)')
        #     plt.ylabel('Y (km)')
        #     plt.title(f"Satellite Data Collection at Time: {t:.2f}")

        #     # Make a scatter plot for each satellite, and label the color with legend, just so we can see the estimates
        #     for sat in self.sats:
        #         plt.scatter(-9999999, -9999999, s = 20, color = sat.color, label=sat.name)
        #     for targ in self.targs:
        #         plt.scatter(-9999999, -9999999, s = 20, color = targ.color, label=targ.name)
        #     plt.legend()

        #      # Initialize an empty array to store estimates for each target at this time step
        #     current_estimates = np.ones((len(self.targs), 2))*9999999

        #     for sat in self.sats:
        #         # Get the estimates of the satellite at the time
        #         estimates = sat.estimateHist
        #         for est in estimates:
        #             if est[0, 3] == t:
        #             # Now loop through the targets
        #                 for i, targ in enumerate(self.targs):
        #                     x, y, z = est[i, 0:3]
        #                     if x != 0 or y != 0 or z != 0:

        #                     # Store the estimates in array for each target
        #                         current_estimates[i] = [x, y]
        #                     # Append the current estimates to targ_est
        #                         targ_est.append(current_estimates.copy())

        #                     # Plot the estimate
        #                         plt.scatter(x, y, s = 40, color = sat.color)

        #                 # For the given target, plot the estimate in dashed plot
        #                     targ_data = [arr[i, :] for arr in targ_est]
        #                     x_tot = [point[0] for point in targ_data]
        #                     y_tot = [point[1] for point in targ_data]   
        #                     plt.scatter(x_tot, y_tot, s = 10, color = targ.color)
                        
        #     plt.pause(pause_step) 
        #     plt.draw()
        # plt.show()
            
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
