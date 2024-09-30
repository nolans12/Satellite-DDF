from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

# Import classes
from satelliteClass import satellite
from targetClass import target
from commClass import comms

class environment:
    def __init__(self, sats, targs, comms, commandersIntent):
        """
        Initialize an environment object with satellites, targets, communication network, and optional central estimator.
        """
       
        ## Populate the environment variables   
        self.sats = sats # define the satellites
        
        self.targs = targs # define the targets
        
        self.comms = comms # define the communication network

        self.commandersIntent = commandersIntent # define the commanders intent

        # Initialize time parameter to 0
        self.time = 0
        self.delta_t = None
        
        # Environemnt Plotting parameters
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # If you want to do clustered case:
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
        self.earth_r = 6371.0
        self.x_earth = self.earth_r * np.outer(np.cos(u), np.sin(v))
        self.y_earth = self.earth_r * np.outer(np.sin(u), np.sin(v))
        self.z_earth = self.earth_r * np.outer(np.ones(np.size(u)), np.cos(v))

        # Empty lists and dictionaries for simulation images
        self.imgs = [] # dictionary for 3D gif images


    def simulate(self, time_vec, pause_step=0.1):
        """
        Simulate the environment over a time range.
        
        Args:
        - time_vec (array): Array of time steps to simulate over.
        - pause_step (float): Time to pause between each step in the simulation.
        
        Returns:
        - Data collected during simulation.
        """

        print("Simulation Started")
        
        # Initialize based on the current time
        time_vec = time_vec + self.time
        self.delta_t = (time_vec[1] - time_vec[0]).to_value(time_vec.unit)
        for t_net in time_vec:

            print(f"Time: {t_net:.2f}")

            # Get the delta time to propagate
            t_d = t_net - self.time  
            
            # Propagate the environments positions
            self.propagate(t_d)

            for targ in self.targs:
                satsForTarg = []
                for sat in self.sats:
                    # Check if the target is in the sensor field of view
                    if sat.sensor.inFOV(sat, targ):
                        satsForTarg.append(sat)

                # Now, for each targ, calc the PDOP based on the satsForTarg
                if len(satsForTarg) > 0:
                    self.calcPDOP(targ, satsForTarg)

            # Update the plot environment
            self.plot()
            plt.pause(pause_step)
            plt.draw()
        

        print("Simulation Complete")

        
    def propagate(self, time_step):
        """
        Propagate the satellites and targets over the given time step.
        """

        # Update the current time
        self.time += time_step

        time_val = self.time.to_value(self.time.unit) # extract the numerical value of time
        
        # Update the time for all targets and satellites
        for targ in self.targs:
            targ.time = time_val
        for sat in self.sats:
            sat.time = time_val

        # Propagate the targets' positions
        for targ in self.targs:
            targ.propagate(time_step) # Stores the history of target time and xyz position and velocity

        # Propagate the satellites
        for sat in self.sats:
            sat.orbit = sat.orbit.propagate(time_step)
            # Also now update the sensor projection
            sat.sensor.visible_projection(sat)

        # Update the communication network for the new sat positions
        self.comms.make_edges(self.sats)
        
    
    def calcPDOP(self, target, satsForTarg):
        """
        Input is list of satellites that can see the target and the target object
        """
        
        # For position DOP, first compute the A matrix

        if len(satsForTarg) < 4:
            GDOP = 100
        else:

            target_pos = target.pos

            # Now, iterative compute the rows for the A matrix
            A = []
            for sat in satsForTarg:
                sat_pos = sat.orbit.r.value
                x = sat_pos[0] - target_pos[0]
                y = sat_pos[1] - target_pos[1]
                z = sat_pos[2] - target_pos[2]
                r = np.linalg.norm([x, y, z])
                A.append([x/r, y/r, z/r, 1])

            # Now, compute the PDOP
            A = np.array(A)
            cov = np.linalg.inv(A.T @ A)
            
            # Use only the first 3 elements of the trace
            GDOP = np.sqrt(np.trace(cov))

        print(f"GDOP for {target.name}: {GDOP:.2f}")


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
            self.ax.add_collection3d(Poly3DCollection([box], facecolors=sat.color, linewidths=1, edgecolors=sat.color, alpha=.025))


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
            self.ax.scatter(x, y, z, s=40, marker='x', color=targ.color, label=targ.name)


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
    
        
    def save_envPlot_to_imgs(self):
        ios = io.BytesIO()
        self.fig.savefig(ios, format='raw')
        ios.seek(0)
        w, h = self.fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        self.imgs.append(img)


    def render_gif(self, filePath=os.path.dirname(os.path.realpath(__file__)), fps=1):
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
        file = os.path.join(filePath, 'gifs', f'satellite_sim.gif')
        with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
            for img in self.imgs:
                writer.append_data(img)
