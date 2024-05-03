from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, fov, sensorError, name, color, sensorDetectError = 0.1):
    
    # Create the orbit
        # Check if already in units, if not convert
        if type(a) == int:
            a = a * u.km
        self.a = a
        
        if type(ecc) == int:
            ecc = ecc * u.dimensionless_unscaled
        self.ecc = ecc 
       
        if type(inc) == int:
            inc = inc * u.deg
        self.inc = inc 
        
        if type(raan) == int:
            raan = raan * u.deg
        self.raan = raan 
        
        if type(argp) == int:
            argp = argp * u.deg
        self.argp = argp 
        
        if type(nu) == int:
            nu = nu * u.deg
        self.nu = nu 
        
        # Create the poliastro orbit
        self.orbit = Orbit.from_classical(Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)
        self.orbitHist = []; # contains xyz of orbit history
        self.fullHist = []; # contains xyz and time of full history
        self.time = 0

        self.estimateHist = [] # contains xyz and t of estimated history

    # Initalize the projection xyz
        self.projBox = np.array([0, 0, 0])

    # Define sensor settings
        self.fov = fov
        # self.fovNarrow = fovNarrow
        # self.fovWide = fovWide
        self.sensorDetectError = sensorDetectError
        self.sensorError = sensorError

    # Other parameters
        self.name = name
        self.color = color

    # Returns the 4 direction vectors of the projection box
    def projection_vectors(self):
        # Get original xyz position of the satellite
        x_sat, y_sat, z_sat = self.orbit.r.value

        # Get magnitude, r
        r = np.linalg.norm([x_sat, y_sat, z_sat])
        # print(r)
        # Get original direction vector
        dir_orig = -np.array([x_sat, y_sat, z_sat])/r

        # Rotate the vector such that y axis is alligned with direction vector
        elevation = np.arcsin(z_sat/r)
        azimuth = np.arctan2(y_sat, x_sat) # change to x_sat, y_sat
        
        # There should be no rotation when Satellite is looking from south pole
        theta = 3*np.pi/2 - elevation
        
        # Rotate about second axis so that z axis alligned with direction vector
        R2 = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        
        # Rotate about third axis so that x axis alligned with direction vector
        
        R2_inv = np.linalg.inv(R2)
        dir_orig = r*np.dot(R2, dir_orig)
        x_sat, y_sat, z_sat = dir_orig[0:3]
 
        # Define the rotation axes for the four directions
        rotation_axes = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]

        # Initialize list to store the new direction vectors
        dir_new_list = []

        # Loop over the four directions
        for axis in rotation_axes:
            # Calculate the change in position for this direction
            change = r*np.tan(np.radians(self.fov/2))
            # Calculate the perpendicular vector for this direction
            perp_vec = np.cross([x_sat, y_sat, z_sat], axis)
            # Normalize the perpendicular vector
            perp_vec = perp_vec/np.linalg.norm(perp_vec)
            # Apply the change to the original position to get the new position
            x_new = x_sat + change*perp_vec[0]
            y_new = y_sat + change*perp_vec[1]
            z_new = z_sat + change*perp_vec[2]
            # Normalize the new position to get the new direction vector
            dir_new = -np.array([x_new, y_new, z_new])/np.linalg.norm([x_new, y_new, z_new])
        
            # take the inverse of R2
            dir_new = r*np.dot(R2_inv, dir_new)
            
            # Rotate z axis 45 deg
            # R3 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0], [np.sin(np.pi/4), np.cos(np.pi/4), 0], [0, 0, 1]])
            # dir_new = np.dot(R3, dir_new)
            
            # Add the new direction vector to the list
            dir_new_list.append(dir_new)

        return np.array(dir_new_list)
    
    # Return the 4 xyz projection points of where on earth the satellite can see
    # CURRENTLY RETURNS NONE IF NO INTERSECTION, AKA CAN SEE EVERYTHING
    def visible_projection(self):

        # Get the current xyz position of the satellite
        x, y, z = self.orbit.r.value

        # Now get the projection_vectors 
        proj_vecs = self.projection_vectors()

        # Now find where the projection vectors intersect with the earth, given that they start at the satellite
        points = []
        for vec in proj_vecs:
            # Find the intersection of the line from the satellite to the earth
            # with the earth
            intersection = self.sphere_line_intersection([0, 0, 0], 6378, [x, y, z], vec)
            points.append(intersection)

        return np.array(points)
    
    # Returns the closest intersection point of a line with a sphere
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
            

# Calculate the visible projection of the satellite
    # Returns the 4 points of xyz intersection with the earth that approximately define the visible projection
    def visible_projection_orig(self):

    # Need the 4 points of intersection with the earth
        # Get the current xyz position of the satellite
        x, y, z = self.orbit.r.value

        # Get the altitude above earth of the satellite
        alt = np.linalg.norm([x, y, z]) - 6378.0

        # Now calculate the magnitude of fov onto earth
        wideMag = np.tan(np.radians(self.fov)/2) * alt
        narrowMag = np.tan(np.radians(self.fov)/2) * alt

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