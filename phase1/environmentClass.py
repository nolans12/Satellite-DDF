from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment: 
    def __init__(self, sats, targs, centralEstimator = None):

    # If a central estimator is passed, use it
        if centralEstimator:
            self.estimator = centralEstimator

    # Define the satellites
        self.sats = sats

    # Define the targets
        self.targs = targs

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
        
        # Plot the trail of the satellite, but only up to last n points
            n = 5
            if len(sat.orbitHist) > n:
                t, r = zip(*sat.orbitHist[-n:])
                x, y, z = np.array(r).T
            else:
                t, r = zip(*sat.orbitHist)
                x, y, z = np.array(r).T
            self.ax.plot(x, y, z, color = sat.color, linestyle='--', linewidth = 1)

    # FOR EACH TARGET, PLOTS
        for targ in self.targs:
        # Plot the current xyz location of the target
            x, y, z = targ.pos
            self.ax.scatter(x, y, z, s=20, color = targ.color, label=targ.name)
            
        self.ax.legend()

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
            targ.x = targ.propagate(time_step, self.time)

            # Update the history of the target
            targ.hist.append([targ.time, targ.x]) # history of target time and xyz position
        
    # Propagate the satellites
        for sat in self.sats:
            
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)

            # Collect measurements on any avaliable targets
            sat.collect_measurements(self.targs)

            # # Update local estimators
            # for targ in self.targs:
            #     if targ.targetID in sat.targetIDs:
            #         test = sat.estimator.EKF(sat.measurementHist, targ.targetID, time_step.value, sat.sensor)
            #         if test != 0:
            #             print("Truth")
            #             print(self.targs[0].pos)
            #             print("Estimate")
            #             print(test)
            
            # Update the history of the orbit
            sat.orbitHist.append([sat.time, sat.orbit.r.value]) # history of sat time and xyz position

# Simulate the environment over a time range
    # Time range is a numpy array of time steps, must have poliastro units associated!
    # Pause step is the time to pause between each step, if displaying as animation
    # Display is a boolean, if true will display the plot as an animation
    def simulate(self, time_vec, pause_step = 0.1, display = False):
        
        # Initalize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time # Get delta time to propagate, works because propagate func increases time
        
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
                        for i in sat.measurementHist[targ.targetID]:
                            writer.writerow(i)


# # Plot the results of the simulation, loop through all satellites and plot on a xy the estimates for each target over time
# # This function is horrible, but just a quick way to demo results. 
# # NEED to clean up, and have better way/structure to store estimates
#     def plotResults(self, pause_step = 0.1):
#         # Grab the time values, just use one of the sats
#         time = np.array(self.sats[0].estimateHist)[:, 0][:, 3] # Assumes at least 1 sat 1 targ

#         # Create empty array for each target to store estimates:
#         targ_est = []

#         # Now, loop through time, and for each time, plot the estimates of each satellite
#         for t in time:
#             # Clear the plot
#             plt.clf()
#             plt.xlim([-300, 300])
#             plt.ylim([-300, 300])
#             plt.xlabel('X (km)')
#             plt.ylabel('Y (km)')
#             plt.title(f"Satellite Data Collection at Time: {t:.2f}")

#             # Make a scatter plot for each satellite, and label the color with legend, just so we can see the estimates
#             for sat in self.sats:
#                 plt.scatter(-9999999, -9999999, s = 20, color = sat.color, label=sat.name)
#             for targ in self.targs:
#                 plt.scatter(-9999999, -9999999, s = 20, color = targ.color, label=targ.name)
#             plt.legend()

#              # Initialize an empty array to store estimates for each target at this time step
#             current_estimates = np.ones((len(self.targs), 2))*9999999

#             for sat in self.sats:
#                 # Get the estimates of the satellite at the time
#                 estimates = sat.estimateHist
#                 for est in estimates:
#                     if est[0, 3] == t:
#                     # Now loop through the targets
#                         for i, targ in enumerate(self.targs):
#                             x, y, z = est[i, 0:3]
#                             if x != 0 or y != 0 or z != 0:

#                             # Store the estimates in array for each target
#                                 current_estimates[i] = [x, y]
#                             # Append the current estimates to targ_est
#                                 targ_est.append(current_estimates.copy())

#                             # Plot the estimate
#                                 plt.scatter(x, y, s = 40, color = sat.color)

#                         # For the given target, plot the estimate in dashed plot
#                             targ_data = [arr[i, :] for arr in targ_est]
#                             x_tot = [point[0] for point in targ_data]
#                             y_tot = [point[1] for point in targ_data]   
#                             plt.scatter(x_tot, y_tot, s = 10, color = targ.color)
                        
#             plt.pause(pause_step) 
#             plt.draw()
#         plt.show()
            
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
