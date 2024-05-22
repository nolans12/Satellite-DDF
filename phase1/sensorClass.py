from import_libraries import *
# Sensor Class

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
    

    # Input: A satellite and target object
    # Output: If visible, returns a sensor measurement of target, otherwise returns 0
    def get_measurement(self, sat, targ):
        
        # Get fov box:
        self.visible_projection(sat)

        # Check if target is in the fov box
        if self.point_box_intersection(targ):
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
        x_sat_orig, y_sat_orig, z_sat_orig = sat.orbit.r.value
        alpha, beta = measurement[0], measurement[1]

    # Now perform ray tracing to get the direction vector the satellite is measuring
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
        Tinv = np.linalg.inv(T)

        # Get satellite in-track, cross-track, z components
        dir_rot = np.dot(T, np.array([x_sat_orig, y_sat_orig, z_sat_orig]))
        x_sat, y_sat, z_sat = dir_rot[0:3]

        # Get initial direction vector, pointing to center of Earth
        inital_vec = np.array([x_sat, y_sat, z_sat])/np.linalg.norm([x_sat, y_sat, z_sat])

        # Now apply a R2 of alpha and R1 of beta to the initial vector
        R2 = np.array([[np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],
                          [np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0],
                            [0, 0, 1]])
        R1 = np.array([[np.cos(np.radians(beta)), 0, np.sin(np.radians(beta))],
                            [0, 1, 0],
                            [-np.sin(np.radians(beta)), 0, np.cos(np.radians(beta))]])

        # Get the direction vector of the target in bearing frame
        dir_meas = np.dot(R2, np.dot(R1, inital_vec))

        # Now rotate back to ECI
        dir_meas = np.dot(Tinv, dir_meas)

        # print
        print("Initial Vector: ", inital_vec)
        print("Direction Vector: ", dir_meas)
        print("Truth Bearings: ", alpha, beta)

        # calculate the angle between the two vectors
        angle = np.arccos(np.dot(inital_vec, dir_meas)/(np.linalg.norm(inital_vec)*np.linalg.norm(dir_meas)))*180/np.pi
        print("Angle between vectors: ", np.degrees(angle))

        # Now get the intersection with Earth:
        intersection = self.sphere_line_intersection([0, 0, 0], 6378, [x_sat_orig, y_sat_orig, z_sat_orig], dir_meas)

        return intersection


    # Input: A satellite and target object (one that is visible)
    # Output: A bearings only measurement of the target with error
    def sensor_model(self, sat, targ):

        # Get original location of satellite
        x_sat, y_sat, z_sat = sat.orbit.r.value

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

        # Get satellite in-track, cross-track, z components
        dir_rot = np.dot(T, np.array([x_sat, y_sat, z_sat]))
        x_sat, y_sat, z_sat = dir_rot[0:3]

        # Get the target in-track, cross-track, z components
        dir_rot = np.dot(T, np.array(targ.pos))
        x_targ, y_targ, z_targ = dir_rot[0:3]
        
        # Zero target values by satellite values
        in_track_targ = x_targ - x_sat
        cross_track_targ = y_targ - y_sat

        # # Print test
        # print("Satellite: ", x_sat, y_sat, z_sat)
        # print("Target: ", in_track_targ, cross_track_targ)

        # Now have target truth position in in-track and cross-track components
        height = z_sat - 6378 
        alpha_truth = np.arctan2(in_track_targ, height)*180/np.pi
        beta_truth = np.arctan2(cross_track_targ, height)*180/np.pi

        # Add sensor error, assuming gaussian
        alpha_meas = alpha_truth + np.random.normal(0, self.sensorError[0])
        beta_meas = beta_truth + np.random.normal(0, self.sensorError[1])

        # # Print test
        # print("Satellite: ", x_sat, y_sat, z_sat)
        # print("Truth: ", alpha_truth, beta_truth)
        # print("Measured: ", alpha_meas, beta_meas)
        
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
    # TODO: Implement edge case for if sensor can see all of Earth
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
            
    # Input: Target
    # Output: True or False if the target is within the sensor's field of view
    def point_box_intersection(self, targ):

        # Get the current xyz position of the target
        x, y, z = targ.pos

        # Get all x values of the projection box
        x_vals = self.projBox[:, 0]
        y_vals = self.projBox[:, 1]
        z_vals = self.projBox[:, 2]
        z_tolerance = 0.15 * 6378

        # Now, just check if the targets xyz is within the xyz of the projBox
        if (x > min(x_vals) and x < max(x_vals) and y > min(y_vals) and y < max(y_vals) and z > min(z_vals) - z_tolerance and z < max(z_vals) + z_tolerance):
            return True
        else:
            return False

