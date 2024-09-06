from import_libraries import *

class comms:
    """
    Communication network class.
    """
    def __init__(self, sats, maxNeighbors, maxRange, minRange, maxBandwidth = 100000000, dataRate=0, displayStruct=False):
        """Initialize the communications network.
                Using networkx, a python library, to create a graph of the satellites and their connections.

        Args:
            sats (list): List of satellites.
            max_neighbors (int): Maximum number of neighbors.
            max_range (float): Maximum range for communication.
            min_range (float): Minimum range for communication.
            data_rate (int, optional): Data rate. Defaults to 0.
            display_struct (bool, optional): Flag to display structure. Defaults to False.
        """
        # Create a graph instance with the satellites as nodes
        self.G = nx.DiGraph()

        # Add nodes with a dict for queued data (list of arrays)
        for sat in sats:
            self.G.add_node(sat, estimate_data={}, received_measurements={}, sent_measurements={})
            

        self.maxBandwidth = maxBandwidth
    
        # Create a empty dicitonary to store the amount of data sent/recieved between satellites
        self.total_comm_data = NestedDict()
        self.used_comm_data = NestedDict()
        
        self.total_comm_et_data = NestedDict()
        self.used_comm_et_data = NestedDict()
        self.used_comm_et_data_values = NestedDict()
        
        self.max_neighbors = maxNeighbors
        self.max_range = maxRange
        self.min_range = minRange
        self.data_rate = dataRate
        self.displayStruct = displayStruct

    def send_estimate_path(self, path, est_meas_source, cov_meas_source, target_id, time):
        """ Send an estimate along a path of satellites.
                Includes ferrying the data!!!
        """

        # Take the path, and from path[0] (source node) share the est_meas and cov_meas to all other nodes in the path, send the data
        for i in range(1, len(path)):
            self.send_estimate(path[i-1], path[i], est_meas_source, cov_meas_source, target_id, time)
            # Activate that edge
            # self.G.edges[path[i-1], path[i]]['active'] = True



    def send_estimate(self, sender, receiver, est_meas, cov_meas, target_id, time):
        """Send an estimate from one satellite to another
                First checks if two satellits are neighbors,
                then, if they are neighbors, we want to share the most recent estimate
                from sender to the receiver. We will simulate "sending a measurement"
                by adding the estimate to the receiver's queued data on the commnication node.
                This way, at the end of the time step, the reciever satellite can just loop through
                the queued data and update its estimate using DDF algorithms on it.

                The queued data is a dictionary of dictionaries of lists. The first key is the time,
                the second key is the target ID, and the list contains the estimates, covariances, and who sent them.

        Args:
            sender (Satellite): Satellite sending the estimate.
            receiver (Satellite): Satellite receiving the estimate.
            est_meas (array): Estimate to send.
            cov_meas (array): Covariance estimate to send.
            target_id (int): ID of the target the estimate is from.
            time (float): Time the estimate was taken.
        """
        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return 
        
        # Before we decide if we want to send the estimate, need to make sure it wont violate the bandwidth constraints
        # Check if the bandwidth is available
        if self.G.edges[sender, receiver]['usedBandwidth'] + est_meas.size*2 + cov_meas.size/2 > self.G.edges[sender, receiver]['maxBandwidth']:
            # print(f"Bandwidth exceeded between {sender.name} and {receiver.name} with current bandwith of {self.G.edges[sender, receiver]['usedBandwidth']} and max bandwidth of {self.G.edges[sender, receiver]['maxBandwidth']}")
            return
        else:
            # Update the used bandwidth
            self.G.edges[sender, receiver]['usedBandwidth'] += est_meas.size*2 + cov_meas.size/2
        
        # Initialize the target_id key in the receiver's queued data if not present
        if time not in self.G.nodes[receiver]['estimate_data']:
            self.G.nodes[receiver]['estimate_data'][time] = {}
        
        # Initialize the time key in the target_id's queued data if not present
        if target_id not in self.G.nodes[receiver]['estimate_data'][time]:
            self.G.nodes[receiver]['estimate_data'][time][target_id] = {'est': [], 'cov': [], 'sender': []}

        # Add the estimate to the receiver's queued data at the specified target_id and time
        self.G.nodes[receiver]['estimate_data'][time][target_id]['est'].append(est_meas)
        self.G.nodes[receiver]['estimate_data'][time][target_id]['cov'].append(cov_meas)
        self.G.nodes[receiver]['estimate_data'][time][target_id]['sender'].append(sender.name)

        self.total_comm_data[target_id][receiver.name][sender.name][time] = est_meas.size*2 + cov_meas.size/2

    def send_measurements(self, sender, receiver, meas_vector, target_id, time):
            """Send a vector of measurements from one satellite to another.
                    First checks if two satellites are neighbors,
                    then, if they are neighbors, we share the measurement vector
                    from the sender to the receiver by adding it to the receiver's
                    measurement data on the communication node.

                    The measurement data is a dictionary of dictionaries of lists.
                    The first key is the time, the second key is the target ID,
                    and the list contains the measurement vectors and who sent them.

            Args:
                sender (Satellite): Satellite sending the measurements.
                receiver (Satellite): Satellite receiving the measurements.
                meas_vector (array): Measurement vector to send.
                target_id (int): ID of the target the measurements are from.
                time (float): Time the measurements were taken.
            """
            # Check if the receiver is in the sender's neighbors
            if not self.G.has_edge(sender, receiver):
                return
            
            # Initialize the time key in the receiver's measurement data if not present
            if time not in self.G.nodes[receiver]['received_measurements']:
                self.G.nodes[receiver]['received_measurements'][time] = {}
            
            # Initialize the target_id key in the measurement data if not present
            if target_id not in self.G.nodes[receiver]['received_measurements'][time]:
                self.G.nodes[receiver]['received_measurements'][time][target_id] = {'meas': [], 'sender': []}

            # Add the measurement vector to the receiver's measurement data at the specified target_id and time
            self.G.nodes[receiver]['received_measurements'][time][target_id]['meas'].append(meas_vector)
            self.G.nodes[receiver]['received_measurements'][time][target_id]['sender'].append(sender)
            
            # Add the measurement vector to the senders sent measurements at the specified target_id and time
            if time not in self.G.nodes[sender]['sent_measurements']:
                self.G.nodes[sender]['sent_measurements'][time] = {}
            
            if target_id not in self.G.nodes[sender]['sent_measurements'][time]:
                self.G.nodes[sender]['sent_measurements'][time][target_id] = {'meas': [], 'receiver': []}
                
            self.G.nodes[sender]['sent_measurements'][time][target_id]['meas'].append(meas_vector)
            self.G.nodes[sender]['sent_measurements'][time][target_id]['receiver'].append(receiver)
            
            measVector_size = 2 + 2 # 2 for the meas vector, 2 for the sensor noise
            if np.isnan(meas_vector[0]):
                measVector_size -= 1
            
            if np.isnan(meas_vector[1]):
                measVector_size -= 1
            
                
            self.total_comm_et_data[target_id][receiver.name][sender.name][time] = measVector_size
            
           #self.total_comm_data[target_id][receiver.name][sender.name][time] = meas_vector.size # TODO: need a new dicitonary to store this and sent data

    def make_edges(self, sats):
        """Reset the edges in the graph and redefine them based on range and if the Earth is blocking them.
                Performs double loop through all satellites to check known pairs.
                An edge represnts a theorical communication link between two satellites. 

        Args:
            sats (list): List of satellites.
        """
        # Clear all edges in the graph
        self.G.clear_edges()

        # Loop through each satellite pair and remake the edges
        for sat1 in sats:
            for sat2 in sats:
                if sat1 != sat2:
                    # Check if an edge already exists between the two satellites
                    if not self.G.has_edge(sat1, sat2):
                        # Check if the distance is within range
                        dist = np.linalg.norm(sat1.orbit.r - sat2.orbit.r)
                        if self.min_range < dist < self.max_range:
                            # Check if the Earth is blocking the two satellites
                            if not self.intersect_earth(sat1, sat2):
                                # Add the edge
                                self.G.add_edge(sat1, sat2, maxBandwidth = self.maxBandwidth, usedBandwidth = 0, active=False)
                                # also add the edge in the opposite direction
                                self.G.add_edge(sat2, sat1, maxBandwidth = self.maxBandwidth, usedBandwidth = 0, active=False)

        # Restrict to just the maximum number of neighbors
        for sat in sats:
            # If the number of neighbors is greater than the max, remove the extra neighbors
            if len(list(self.G.neighbors(sat))) > self.max_neighbors:
                # Get the list of neighbors
                neighbors = list(self.G.neighbors(sat))

                # Get the distances to each neighbor
                dists = [np.linalg.norm(neighbor.orbit.r - sat.orbit.r) for neighbor in neighbors]

                # Sort the neighbors by distance
                sorted_neighbors = [x for _, x in sorted(zip(dists, neighbors), key=lambda pair: pair[0])]

                # Remove the extra neighbors
                for i in range(self.max_neighbors, len(sorted_neighbors)):
                    self.G.remove_edge(sat, sorted_neighbors[i])
        
    def intersect_earth(self, sat1, sat2):
        """Check if the Earth is blocking the two satellites using line-sphere intersection.
                Performs a line-sphere intersection check b/w the line connecting the two satellites to see if they intersect Earth.

        Args:
            sat1 (Satellite): The first satellite.
            sat2 (Satellite): The second satellite.

        Returns:
            bool: True if the Earth is blocking the line of sight, False otherwise.
        """
        # Make a line between the two satellites
        line = (sat2.orbit.r - sat1.orbit.r).value  # This is from sat1 to sat2

        # Check if there is an intersection with the Earth
        if self.sphere_line_intersection([0, 0, 0], 6378, sat1.orbit.r.value, line) is not None:
            return True

        # If there is no intersection, return False
        return False

    
    def sphere_line_intersection(self, sphere_center, sphere_radius, line_point, line_direction):
        """Check if a line intersects with a sphere.
                Uses known fomrula for line-sphere intersection in 3D space.

        Args:
            sphere_center (list): Coordinates of the sphere center.
            sphere_radius (float): Radius of the sphere.
            line_point (list): Point on the line.
            line_direction (list): Direction of the line.

        Returns:
            array or None: Intersection point(s) or None if no intersection.
        """
        # Unpack sphere parameters
        x0, y0, z0 = sphere_center
        r = sphere_radius
        
        # Unpack line parameters
        x1, y1, z1 = line_point
        dx, dy, dz = line_direction
        
        # Compute coefficients for the quadratic equation
        a = dx**2 + dy**2 + dz**2
        b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
        c = (x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2 - r**2
        
        # Compute discriminant
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            # No intersection
            return None
        elif discriminant == 0:
            # One intersection
            t = -b / (2 * a)
            intersection_point = np.array([x1 + t * dx, y1 + t * dy, z1 + t * dz])
            return intersection_point
        else:
            # Two intersections
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            intersection_point1 = np.array([x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz])
            intersection_point2 = np.array([x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz])
            
            # Calculate distances
            dist1 = np.linalg.norm(intersection_point1 - line_point)
            dist2 = np.linalg.norm(intersection_point2 - line_point)
            
            if dist1 < dist2:
                return intersection_point1
            else:
                return intersection_point2
