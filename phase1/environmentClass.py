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
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.title('Satellite Orbit Visualization')
        
        # All earth parameters for plotting
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.x_earth = 6378 * np.outer(np.cos(u), np.sin(v))
        self.y_earth = 6378 * np.outer(np.sin(u), np.sin(v))
        self.z_earth = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Empty images list to later make a gif of the simulation
        self.imgs = []

# Propagate the satellites over the time step  
    def propagate(self, time_step):
        for sat in self.sats:
            # Propagate the orbit
            sat.orbit = sat.orbit.propagate(time_step)
            # Update the history of the orbit
            sat.orbitHist.append(sat.orbit.r.value)
        

        # Update the current time
        self.time += time_step
        # print(f"Time: {self.time}")

# Plot the current state of the environment
    def plot(self):
        # Reset plot
        self.ax.clear()
        self.ax.set_xlim([-8000, 8000])
        self.ax.set_ylim([-8000, 8000])
        self.ax.set_zlim([-8000, 8000])

        # Put text of current time in top left corner
        self.ax.text2D(0.05, 0.95, f"Time: {self.time:.2f}", transform=self.ax.transAxes)

        # Plot Earth
        self.ax.plot_surface(self.x_earth, self.y_earth, self.z_earth, color='k', alpha=0.1)
    
        # Plot each satellite's current position
        for sat in self.sats:
            x, y, z = sat.orbit.r.value
            self.ax.scatter(x, y, z, s=40, color = sat.color, label=sat.name)

        # Plot the orbit history of each satellite
        for sat in self.sats:
            # But only plot up to the last 10 points
            if len(sat.orbitHist) > 10:
                x, y, z = np.array(sat.orbitHist[-10:]).T
            else:
                x, y, z = np.array(sat.orbitHist).T
            self.ax.plot(x, y, z, color = sat.color, linestyle='--', linewidth = 1)

        plt.legend()

# Animate the environment over a time range
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
            # Simulate the plot
                plt.pause(pause_step) 
                plt.draw()

        # display the sat history
        # print("Satellite 1 History: ", self.sats[0].orbitHist)
        

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
    def render_gif(self, fileName = '/satellite_orbit.gif', filePath = os.path.dirname(os.path.realpath(__file__)), frame_duration=50):
        file = os.path.join(filePath, fileName)
        with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
            for img in self.imgs:
                writer.append_data(img)
