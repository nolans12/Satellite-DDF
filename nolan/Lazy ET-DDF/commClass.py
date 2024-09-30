from import_libraries import *

class comms:
    """
    Communication network class.
    """
    def __init__(self, sats, maxNeighbors, maxRange, minRange, maxBandwidth = 100000000, dataRate = 0, displayStruct = False):
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
        self.max_neighbors = maxNeighbors
        self.max_range = maxRange
        self.min_range = minRange
        self.data_rate = dataRate
        self.displayStruct = displayStruct


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
