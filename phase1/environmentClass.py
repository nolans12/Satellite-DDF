from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class environment: 
    def __init__(self, sats):
    # Define the satellites
        self.sats = sats

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
        # bm = PIL.Image.open(filePath + '/extra/countries.jpg')
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

        self.ax.legend()

# Propagate the satellites over the time step  
    def propagate(self, time_step):
        for sat in self.sats:
            
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)
            
            # Update the satellites xyz projection
            sat.projBox = self.visible_projection(sat)

            # Update the history of the orbit
            sat.orbitHist.append(sat.orbit.r.value)

        # Update the current time
        self.time += time_step

# Simulate the environment over a time range
    # Time range is a numpy array of time steps, must have poliastro units associated!
    # Pause step is the time to pause between each step, if displaying as animation
    # Display is a boolean, if true will display the plot as an animation
    def simulate(self, time_vec, pause_step = 0.1, display = False):
        # Initalize based on the current time
        time_vec = time_vec + self.time
        for t_net in time_vec:
            t_d = t_net - self.time # Get delta time to propagate, works because propagate func increases time
        
        # Propagate the satellites and environment
            self.propagate(t_d)

        # Update the plot environment
            self.plot()

        # Save the current plot to the images list, for gif later
            self.convert_imgs()

            if display:
            # Display the plot in a animation
                plt.pause(pause_step) 
                plt.draw()

# Calculate the visible projection of the satellite
    # Takes in a satellite object
    # Returns the 4 points of xyz intersection with the earth that approximately define the visible projection
    def visible_projection(self, sat):

    # Need the 4 points of intersection with the earth
        # Get the current xyz position of the satellite
        x, y, z = sat.orbit.r.value

        # Get the altitude above earth of the satellite
        alt = np.linalg.norm([x, y, z]) - self.earth_r

        # Now calculate the magnitude of fov onto earth
        wideMag = np.tan(np.radians(sat.fovWide)/2) * alt
        narrowMag = np.tan(np.radians(sat.fovNarrow)/2) * alt

        # Then vertices of the fov box onto the earth is xyz projection +- magnitudes
        # Get the pointing vector of the satellite
        point_vec = np.array([x, y, z])/np.linalg.norm([x, y, z])
        
        # Now get the projection onto earth of center of fov box
        center_proj = np.array([x - point_vec[0] * alt, y - point_vec[1] * alt, z - point_vec[2] * alt])

        # Now get the 4 xyz points that define the fov box
        # Define vectors representing the edges of the FOV box
        wide_vec = np.cross(point_vec, [0, 0, 1])/np.linalg.norm(np.cross(point_vec, [0, 0, 1]))
        narrow_vec = np.cross(point_vec, wide_vec)/np.linalg.norm(np.cross(point_vec, wide_vec))

        # Calculate the four corners of the FOV box
        corner1 = center_proj + wide_vec * wideMag + narrow_vec * narrowMag
        corner2 = center_proj + wide_vec * wideMag - narrow_vec * narrowMag
        corner3 = center_proj - wide_vec * wideMag - narrow_vec * narrowMag
        corner4 = center_proj - wide_vec * wideMag + narrow_vec * narrowMag

        box = np.array([corner1, corner2, corner3, corner4])

        return box

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
