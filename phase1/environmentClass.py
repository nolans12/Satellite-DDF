from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment: 
    def __init__(self, sats, targs, estimator):
    # Define the satellites
        self.sats = sats

    # Define the targets
        self.targs = targs

    # Define the estimator
        self.estimator = estimator

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

        # filePath = os.path.dirname(os.path.realpath(__file__))
        # bm = PIL.Image.open(filePath + '/extra/blue_marble.jpg')
        # self.bm = np.array(bm.resize([int(d/5) for d in bm.size]))/256.
        # lons = np.linspace(-180, 180, self.bm.shape[1]) * np.pi/180 
        # lats = np.linspace(-90, 90, self.bm.shape[0])[::-1] * np.pi/180 
        # self.earth_r = 6378.0
        # self.x_earth = np.outer(np.cos(lons), np.cos(lats)).T*self.earth_r
        # self.y_earth = np.outer(np.sin(lons), np.cos(lats)).T*self.earth_r
        # self.z_earth = np.outer(np.ones(np.size(lons)), np.sin(lats)).T*self.earth_r

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
        # self.ax.plot_surface(self.x_earth, self.y_earth, self.z_earth, rstride = 4, cstride = 4, facecolors=self.bm, alpha=0.1)
    
    # FOR EACH SATELLITE, PLOTS
        for sat in self.sats:
        # Plot the current xyz location of the satellite
            x, y, z = sat.orbit.r.value
            self.ax.scatter(x, y, z, s=40, color = sat.color, label=sat.name)

        # Plot the visible projection of the satellite
            box = sat.projBox
            self.ax.add_collection3d(Poly3DCollection([box], facecolors=sat.color, linewidths=1, edgecolors=sat.color, alpha=.1))

        # Plot the trail of the satellite, but only up to last 10 points
            if len(sat.orbitHist) > 10:
                x, y, z = np.array(sat.orbitHist[-10:]).T
            else:
                x, y, z = np.array(sat.orbitHist).T
            self.ax.plot(x, y, z, color = sat.color, linestyle='--', linewidth = 1)


    # FOR EACH TARGET, PLOTS
        for targ in self.targs:
        # Plot the current xyz location of the target
            x, y, z = targ.x
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
        self.estimator.time = time_val

    # Propagate the targets position
        for targ in self.targs:

            # Propagate the target
            targ.x = targ.propagate(time_step)

            # Update the history of the target
            targ.hist.append([targ.x, targ.time]) # history of target xyz position
        
    # Propagate the satellites
        for sat in self.sats:
            
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)
            
            # Update the satellites xyz projection
            sat.projBox = sat.visible_projection()

            # Update the history of the orbit
            sat.orbitHist.append(sat.orbit.r.value)
            sat.fullHist.append([sat.orbit.r.value, sat.time])

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

        # Do estimatino, this updates each satellites raw estimate of the targs, if they see it
            self.estimator.estimate_raw()

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

# Saves all satellite estimates to a csv file
    def log_data(self):
        # Make the file, current directory /data/satellite_name.csv
        filePath = os.path.dirname(os.path.realpath(__file__))
        for sat in self.sats:
            with open(filePath + '/data/' + sat.name + '.csv', mode='w') as file:
                writer = csv.writer(file)
                writer.writerow([sat.name + " Raw Estimation History"])
                writer.writerow(["Data Order is: X Estimate, Y Estimate, Z Estimate, Time"])
            # Make string header:
                header = []
                for targ in self.targs:
                    header.append(targ.name + " Estimates")
                writer.writerow(header)
                for i in sat.estimateHist:
                    writer.writerow(i)

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
