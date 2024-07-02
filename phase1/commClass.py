from import_libraries import *
## Creates the communication class, will contain the communication network and all other parameters needed to define the network

class comms:
    def __init__(self, sats, maxNeighbors, maxRange, minRange, dataRate = 0, displayStruct = False):
        
    # Create a graph instance with the satellites as nodes
        self.G = nx.Graph()
        
        # Add nodes with a dict for queued data (list of arrays)
        for sat in sats:
            self.G.add_node(sat, queued_data={})

        self.maxNeighbors = maxNeighbors
        self.maxRange = maxRange
        self.minRange = minRange
        self.dataRate = dataRate
        self.displayStruct = displayStruct

# Function to send a measurement from one satellite to another
    # Send from sender to receiver, with the measurement, targetID, and time
    # Inputs:
    # sender: Satellite sending the measurement
    # receiver: Satellite receiving the measurement
    # est_meas: Estimate measurement to send
    # cov_meas: Covariance measurement to send
    # targetID: ID of the target the measurement is from
    # time: time the measurement was taken
    def send_measurement(self, sender, receiver, est_meas, cov_meas, targetID, time):
        # Check if the receiver is in the sender's neighbors
        if not self.G.has_edge(sender, receiver):
            return 
        
        # Initialize the targetID key in the receiver's queued data if not present
        if targetID not in self.G.nodes[receiver]['queued_data']:
            self.G.nodes[receiver]['queued_data'][targetID] = {}
        
        # Initialize the time key in the targetID's queued data if not present
        if time not in self.G.nodes[receiver]['queued_data'][targetID]:
            self.G.nodes[receiver]['queued_data'][targetID][time] = {'est': [], 'cov': [], 'sender': []}

        # Add the estimate to the receiver's queued data at the specified targetID and time
        self.G.nodes[receiver]['queued_data'][targetID][time]['est'].append(est_meas)
        self.G.nodes[receiver]['queued_data'][targetID][time]['cov'].append(cov_meas)
        self.G.nodes[receiver]['queued_data'][targetID][time]['sender'].append(sender.name)

        # Also activate the edge between the two satellites
        self.G.edges[sender, receiver]['active'] = True

# Reset the edges in the graph and redefine them based on the range and if the Earth is blocking them.
    def make_edges(self, sats):
        
    # Clear all edges in the graph
        self.G.clear_edges()

    # Now loop through each satellite set and remake the edges
        for sat in sats:
            for sat2 in sats:
                if sat != sat2:
                # Check if an edge already exists b/w the two satellites:
                    if (self.G.has_edge(sat, sat2) == False):
                    # Check if the distance is within range
                        if np.linalg.norm(sat.orbit.r - sat2.orbit.r) < self.maxRange and np.linalg.norm(sat.orbit.r - sat2.orbit.r) > self.minRange:
                        # Check if the Earth is blocking the two satellites
                            if self.intersect_earth(sat, sat2) == False:
                                # Add the edge
                                self.G.add_edge(sat, sat2, active = False)

    # Now restrict to just the max number of neighbors
        for sat in sats:
            # If the number of neighbors is greater than the max, remove the extra neighbors
            if len(list(self.G.neighbors(sat))) > self.maxNeighbors:

                # Get the list of neighbors
                neighbors = list(self.G.neighbors(sat))

                # Get the distances to each neighbor
                dists = []
                for neighbor in neighbors:
                    dists.append(np.linalg.norm(neighbor.orbit.r.value - sat.orbit.r.value))

                # Sort the neighbors by distance
                sorted_neighbors = [x for _, x in sorted(zip(dists, neighbors), key=lambda pair: pair[0])]# print(test)

                # Now remove the extra neighbors
                for i in range(self.maxNeighbors, len(sorted_neighbors)):
                    self.G.remove_edge(sat, sorted_neighbors[i])

# Check if the Earth is blocking the two satellites, uses line-sphere intersection
    def intersect_earth(self, sat1, sat2):

        # First make a line between the two satellites:
        line = (sat2.orbit.r - sat1.orbit.r).value # This is from sat1 to sat2

        # Now check if there isn't an intersection with the Earth
        if self.sphere_line_intersection([0, 0, 0], 6378, sat1.orbit.r.value, line) is not None:
            return True
        
        # If there is an intersection, return false, dont connect the two sats
        return False
    
# Line-sphere intersection function to check if the Earth is blocking the two satellites
    def sphere_line_intersection(self, sphere_center, sphere_radius, line_point, line_direction):
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