from import_libraries import *
# Sensor Class
from satelliteClass import satellite
from targetClass import target

# TODO: Add more sensor models, can maybe be subclasses?
class sensor:
    def __init__(self, fov, resolution, sensorError, name, detectError = 0.1):
        self.fov = fov
        self.sensorError = sensorError
        self.detectError = detectError
        self.name = name
        self.resolution = resolution
        self.projBox = np.array([0, 0, 0])

        # Used for saving data
        # TODO: Update later based on different sensor models, for now just bearings
        self.stringHeader = ["Time", "x_sat", "y_sat", "z_sat", "InTrackAngle", "CrossTrackAngle"]


        ## DEFINE R MATRIX
        # Use monte-carlo sampling to sample the sensor error from bearings model to xyz error

        # For now, define the target to be at [0, 0, Earth.R] 
        # And the satellite to be at [0, 0, Earth.R + 1000]
        # This makes the assumption that we are at a constant height, 1000km, above the Earth
        satTemp = satellite(name = 'Temp', sensor = self, targetIDs=[1], estimator = None, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 90, nu = 0, color='k')
        targTemp = target(name = 'Temp', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
        

        # for i in range(1000):
        #     # Sample from the sensor model
        #     alpha = np.random.uniform(-self.fov/2, self.fov/2)
        #     beta = np.random.uniform(-self.fov/2, self.fov/2)
        #     # Get the error
        #     error = self.sensor_model(satellite(0, 0, 0, 0, 0, 0), np.array([alpha, beta]))
        #     # Add to the list
        #     if i == 0:
        #         self.sensorError = np.array([error])
        #     else:
        #         self.sensorError = np.append(self.sensorError, error)
    
    # Input: A satellite and target object
    # Output: If visible, returns a sensor measurement of target, otherwise returns 0
    def get_measurement(self, sat, targ):
        
        # Get fov box:
        self.visible_projection(sat)

        # Check if target is in the fov box
        if self.inFOV(sat, targ):
            # Sample from detection error
            detect = np.random.uniform(0, 1)
            if detect < self.detectError:
                # Didn't detect
                return 0
            else:
                # Get the measurement and return
                return self.sensor_model(sat, targ)
        else:
        # If target isnt visible, return just 0
            return 0
        
    # Input: A satellite object and a bearings measurement
    # Output: A single raw ECI position, containing time and target position in ECI
    def convert_to_ECI(self, sat, measurement):

        # Get the data
        alpha, beta = measurement[0], measurement[1] 
        # convert to radians
        alpha, beta = np.radians(alpha), np.radians(beta)

        # Convert satellite position to be in in-track, cross-track, radial
        rVec = sat.orbit.r.value/np.linalg.norm(sat.orbit.r.value)
        vVec = sat.orbit.v.value/np.linalg.norm(sat.orbit.v.value)
        u = rVec
        w = np.cross(rVec, vVec)
        v = np.cross(w,u)
        T = np.array([v, w, u])
        Tinv = np.linalg.inv(T)

        # Now reverse the bearings calculation
        height = np.linalg.norm(sat.orbit.r.value) - 6378
        in_track_targ = np.tan(alpha)*height
        cross_track_targ = np.tan(beta)*height

        # Desired magntidue is 6378, calculate the Z value that will make the magnitude needed
        desired = 6378
        z_targ_local = np.sqrt(desired**2 - in_track_targ**2 - cross_track_targ**2)

        # Now rotate back to ECI
        dir_rot = np.dot(Tinv, np.array([in_track_targ, cross_track_targ, z_targ_local]))
        x_targ_eci, y_targ_eci, z_targ_eci = dir_rot[0:3]

        return  np.array([x_targ_eci, y_targ_eci, z_targ_eci])
    
    # Input: A satellite and target object (one that is visible)
    # Output: A bearings only measurement of the target with error
    def sensor_model(self, sat, targ):

        # In track, cross cross, along track values
        rVec = sat.orbit.r.value/np.linalg.norm(sat.orbit.r.value)
        vVec = sat.orbit.v.value/np.linalg.norm(sat.orbit.v.value)
        
        # Radial vector
        u = rVec
        # Cross Track vector
        w = np.cross(rVec, vVec)
        # In Track vector
        v = np.cross(w,u)
        # Define rotation matrix
        T = np.array([v, w, u])
        
        # Get the target in-track, cross-track, z components
        dir_rot = np.dot(T, np.array(targ.pos))
        in_track_targ, cross_track_targ, NaN = dir_rot[0:3]
        
        # Now have target truth position in in-track and cross-track components
        height = np.linalg.norm(sat.orbit.r.value) - 6378
        alpha_truth = np.arctan2(in_track_targ, height)*180/np.pi
        beta_truth = np.arctan2(cross_track_targ, height)*180/np.pi

        # Add sensor error, assuming gaussian
        alpha_meas = alpha_truth + np.random.normal(0, self.sensorError[0])
        beta_meas = beta_truth + np.random.normal(0, self.sensorError[1])
        
        return np.array([alpha_meas, beta_meas])
    
    # Input: A satellite object
    # Output: The 4 xyz points of the projection box, based on FOV and sat position
    def visible_projection(self, sat):

        # Get the current xyz position of the satellite
        x, y, z = sat.orbit.r.value

        # Now get the projection_vectors 
        proj_vecs = self.projection_vectors(sat)

        # Now find where the projection vectors intersect with the earth, given that they start at the satellite
        points = []
        for vec in proj_vecs:
            # Find the intersection of the line from the satellite to the earth
            # with the earth
            intersection = self.sphere_line_intersection([0, 0, 0], 6378, [x, y, z], vec)

            # # Edge case for if the sensor can see all of Earth, do the intersection of a 119.5 degree FOV
            # if intersection is None:
            #     intersection = 6378*vec/np.linalg.norm(vec)

            points.append(intersection)

        self.projBox = np.array(points)
        return 
    
    # Input: A satellite object
    # Output: The 4 direction vectors of the projection box, based on FOV and sat position
    def projection_vectors(self, sat):

        # Get original xyz position of the satellite
        x_sat, y_sat, z_sat = sat.orbit.r.value
        
        # In track, cross cross, along track values
        rVec = sat.orbit.r.value/np.linalg.norm(sat.orbit.r.value)
        vVec = sat.orbit.v.value/np.linalg.norm(sat.orbit.v.value)
        
        # Radial vector
        u = rVec
        
        # Cross Track vector
        w = np.cross(rVec, vVec)
        
        # along track vector
        v = np.cross(w,u)
        
        T = np.array([v, w, u])
        Tinv = np.linalg.inv(T)
        
        # Get magnitude, r
        r = np.linalg.norm([x_sat, y_sat, z_sat])

        # Get original direction vector
        dir_orig = np.array([x_sat, y_sat, z_sat])/r

        # Rotate into CAR Frame
        dir_orig = r*np.dot(T, dir_orig)
        x_sat, y_sat, z_sat = dir_orig[0:3]
 
        # Define the rotation axes for the four directions
        # rotation_axes = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        rotation_axes = [[np.sqrt(2), np.sqrt(2), 0], [-np.sqrt(2), -np.sqrt(2), 0], [np.sqrt(2), -np.sqrt(2), 0], [-np.sqrt(2), np.sqrt(2), 0]] # sqrt(2)/2

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
        
            # take the inverse of T and rotate back to ECI
            dir_new = r*np.dot(Tinv, dir_new)
            
            # Add the new direction vector to the list
            dir_new_list.append(dir_new)

        return np.array(dir_new_list)

    # Input: Earth data, projection vector
    # Output: Intersection point of projection vector with earth
    # Description: Uses basic sphere line intersection
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
            
    # Input: Target, Satellite
    # Output: True or False if the target is within the sensor's field of view
    # Will use 4 3D triangle, line intersection algorithms and 1 3D plane intersection
    def inFOV(self, sat, targ):

        # Get the target point
        l0 = targ.pos
        # Now choose another random point
        # So, because I am choosing 100* targPos dont need to worry about the backside of the FOV shape,
        # aka dont need to do a 3d plane - point intersection because 100 * targ.pos will always point away from origin.
        # May want to investigate this later, but for now works great
        l1 = targ.pos * 100

        # Count how many times the line intersects the 3d shape:
        count = 0

        # Get the projection box
        box = self.projBox
        # Do 4 triangle line intersections
        for i in range(4):
            # Get the triangle points
            p0 = sat.orbit.r.value
            p1 = box[i]
            p2 = box[(i+1)%4]

            # Count how many times the line intersects the triangle
            if self.triangle_line_intersection(p0, p1, p2, l0, l1):
                count += 1

        # If the count is odd, the target is in the FOV
        if count % 2 == 1:
            return True
        else:
            return False
        
    def triangle_line_intersection(self, p0, p1, p2, l0, l1):
        # Define the triangle vertices
        v0 = np.array(p0)
        v1 = np.array(p1)
        v2 = np.array(p2)

        # Define the line segment endpoints
        o = np.array(l0)
        d = np.array(l1) - np.array(l0)

        # Calculate edges and normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(d, edge2)
        a = np.dot(edge1, h)

        # If a is close to 0, then the line is parallel to the triangle
        if abs(a) < 1e-8:
            return False

        f = 1.0 / a
        s = o - v0
        u = f * np.dot(s, h)

        # Check if intersection is within the triangle
        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, edge1)
        v = f * np.dot(d, q)

        if v < 0.0 or u + v > 1.0:
            return False

        # Calculate t to find out where the intersection point is on the line
        t = f * np.dot(edge2, q)

        if t >= 0.0 and t <= 1.0:
            return True
        else:
            return False
        
    