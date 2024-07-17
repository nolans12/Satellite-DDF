from import_libraries import *

def state_transition_RK_O(self, estPrior, dt):
        def derivatives(state):
            x, vx, y, vy, z, vz = state
            
            # Turn into Spherical Coordinates
            range = jnp.sqrt(x**2 + y**2 + z**2)
            elevation = jnp.arcsin(z / range)
            azimuth = jnp.arctan2(y, x)
            
            rangeRate = (x * vx + y * vy + z * vz) / range
            elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))
            azimuthRate = (x * vy - y * vx) / (x**2 + y**2)

            # Convert rates back to Cartesian coordinates
            cos_elevation = jnp.cos(elevation)
            sin_elevation = jnp.sin(elevation)
            cos_azimuth = jnp.cos(azimuth)
            sin_azimuth = jnp.sin(azimuth)

            dx = rangeRate * cos_elevation * cos_azimuth \
                - range * elevationRate * sin_elevation * cos_azimuth \
                - range * azimuthRate * cos_elevation * sin_azimuth

            dy = rangeRate * cos_elevation * sin_azimuth \
                - range * elevationRate * sin_elevation * sin_azimuth \
                + range * azimuthRate * cos_elevation * cos_azimuth

            dz = rangeRate * sin_elevation + range * elevationRate * cos_elevation
            
            dvx = 0
            dvy = 0
            dvz = 0

            return jnp.array([dx, dvx, dy, dvy, dz, dvz])

        # RK4 Integration
        k1 = dt * derivatives(estPrior)
        k2 = dt * derivatives(estPrior + 0.5 * dt * k1)
        k3 = dt * derivatives(estPrior + 0.5 * dt * k2)
        k4 = dt * derivatives(estPrior + dt * k3)

        newState = estPrior + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return newState    
    
def state_transition_Orig(self, estPrior, dt):
    # Takes in previous ECI State and returns the next state after dt
    x = estPrior[0]
    vx = estPrior[1]
    y = estPrior[2]
    vy = estPrior[3]
    z = estPrior[4]
    vz = estPrior[5]
    
    # Turn into Spherical Coordinates
    range = jnp.sqrt(x**2 + y**2 + z**2)
    elevation = jnp.arcsin(z / range)
    azimuth = jnp.arctan2(y, x)
    
    rangeRate = (x * vx + y * vy + z * vz) / (range)
    # Calculate elevation rate
    elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))
    # Calculate azimuth rate
    azimuthRate = (x * vy - y * vx) / (x**2 + y**2)
    # Print intermediate values (comment out if not needed in production)
    # jax.debug.print(
    #     "Predic: Range: {range}, Range Rate: {rangeRate}, Elevation: {elevation}, Elevation Rate: {elevationRate}, Azimuth: {azimuth}, Azimuth Rate: {azimuthRate}",
    #     range=range, rangeRate=rangeRate, elevation=elevation, elevationRate=elevationRate, azimuth=azimuth, azimuthRate=azimuthRate)
    # print('*'*50)
    # Propagate the State
    range = range + rangeRate * dt
    elevation = elevation + elevationRate * dt
    azimuth = azimuth + azimuthRate * dt
    
    # Convert back to Cartesian
    x = range * jnp.cos(elevation) * jnp.cos(azimuth)
    y = range * jnp.cos(elevation) * jnp.sin(azimuth)
    z = range * jnp.sin(elevation)
    
    # Approximate velocities conversion (simplified version)
    vx = rangeRate * jnp.cos(elevation) * jnp.cos(azimuth) - \
        range * elevationRate * jnp.sin(elevation) * jnp.cos(azimuth) - \
        range * azimuthRate * jnp.cos(elevation) * jnp.sin(azimuth)
    vy = rangeRate * jnp.cos(elevation) * jnp.sin(azimuth) - \
        range * elevationRate * jnp.sin(elevation) * jnp.sin(azimuth) + \
        range * azimuthRate * jnp.cos(elevation) * jnp.cos(azimuth)
    vz = rangeRate * jnp.sin(elevation) + \
        range * elevationRate * jnp.cos(elevation)
    
    return jnp.array([x, vx, y, vy, z, vz])


def transform_eci_to_bearings_O(self, sat, meas_ECI):
    rVec = self.normalize(jnp.array(sat.orbit.r.value))
    vVec = self.normalize(jnp.array(sat.orbit.v.value))
    wVec = self.normalize(jnp.cross(sat.orbit.r.value, sat.orbit.v.value))
    # Create the transformation matrix T
    T = jnp.stack([vVec.T, wVec.T, rVec.T])
    # Rotate the satellite into Sensor frame:
    sat_pos = jnp.array(sat.orbit.r.value)
    x_sat_sens, y_sat_sens, z_sat_sens = T @ sat_pos
    # Rotate the measurement into the Sensor frame:
    x, y, z = meas_ECI
    meas_ECI_sym = jnp.array([x, y, z])
    x_targ_sens, y_targ_sens, z_targ_sens = T @ meas_ECI_sym
    # Create a line from satellite to the center of Earth:
    satVec = jnp.array([x_sat_sens, y_sat_sens, z_sat_sens])  # sat - earth
    # Now get the in-track component:
    targVec_inTrack = satVec - jnp.array([x_targ_sens, 0, z_targ_sens])  # sat - target
    in_track_angle = jnp.arctan2(jnp.linalg.norm(jnp.cross(targVec_inTrack, satVec)), jnp.dot(targVec_inTrack, satVec))
    
    
    # If targVec_inTrack is negative, switch
    if x_targ_sens < 0:
        in_track_angle = -in_track_angle
    # Now get the cross-track component:
    targVec_crossTrack = satVec - jnp.array([0, y_targ_sens, z_targ_sens])  # sat - target
    cross_track_angle = jnp.arctan2(jnp.linalg.norm(jnp.cross(targVec_crossTrack, satVec)), jnp.dot(targVec_crossTrack, satVec))
    # If targVec_crossTrack is negative, switch
    if y_targ_sens > 0:
        cross_track_angle = -cross_track_angle
    # Convert to degrees:
    in_track_angle_deg = in_track_angle * 180 / jnp.pi
    cross_track_angle_deg = cross_track_angle * 180 / jnp.pi
    
    return jnp.array([in_track_angle_deg, cross_track_angle_deg])

def transform_eci_to_bearings2(self, sat, meas_ECI):
    # Transform the ECI measurement into the Sensor frame
    
    # Get the current sensor frame vectors
    rVec = jnp.array(sat.orbit.r.value)/jnp.linalg.norm(jnp.array(sat.orbit.r.value)) # ECI position of satellite (radial)
    vVec = jnp.array(sat.orbit.v.value)/jnp.linalg.norm(jnp.array(sat.orbit.v.value)) # ECI velocity of satellite (in-track)
    wVec = jnp.cross(sat.orbit.r.value, sat.orbit.v.value)/jnp.linalg.norm(jnp.cross(sat.orbit.r.value, sat.orbit.v.value)) # ECI cross-track vector
    # Create the transformation matrix T = in-track x cross-track x radial
    T = jnp.stack([vVec.T, wVec.T, rVec.T]) 
    # Rotate the satellite position into Sensor frame:
    sat_pos = jnp.array(sat.orbit.r.value)
    x_sat_sens, y_sat_sens, z_sat_sens = T @ sat_pos
    # Rotate the measurement into the Sensor frame:
    x, y, z = meas_ECI
    meas_ECI_sym = jnp.array([x, y, z]) # incase the measurement is a vector
    x_targ_sens, y_targ_sens, z_targ_sens = T @ meas_ECI_sym
    # Create a line from satellite to the center of Earth:
    satVec = jnp.array([x_sat_sens, y_sat_sens, z_sat_sens])  # sat - earth
    # Now get the in-track component:
    targVec_inTrack = satVec - jnp.array([x_targ_sens, 0, z_targ_sens])  # sat - target
    targVec_crossTrack = satVec - jnp.array([0, y_targ_sens, z_targ_sens])  # sat - target
    
    # Normalize all the vectors for numerical stability
    # satVec = satVec/jnp.linalg.norm(satVec)
    # targVec_inTrack = targVec_inTrack/jnp.linalg.norm(targVec_inTrack)
    # targVec_crossTrack = targVec_crossTrack/jnp.linalg.norm(targVec_crossTrack)
    # Use dot product to get the angle between the vectors
    cos_inTrack = jnp.dot(targVec_inTrack, satVec) / (jnp.linalg.norm(targVec_inTrack) * jnp.linalg.norm(satVec))
    cos_crossTrack = jnp.dot(targVec_crossTrack, satVec) / (jnp.linalg.norm(targVec_crossTrack) * jnp.linalg.norm(satVec))
    
    # Clip for numerical issues
    cos_inTrack = jnp.clip(cos_inTrack, -1.0, 1.0)
    cos_crossTrack = jnp.clip(cos_crossTrack, -1.0, 1.0)
    
    in_track_angle = jnp.arccos(cos_inTrack)
    cross_track_angle = jnp.arccos(cos_crossTrack)
    
    # Convert to degrees:
    in_track_angle_deg = in_track_angle * 180 / jnp.pi
    cross_track_angle_deg = cross_track_angle * 180 / jnp.pi
    
    in_track_angle2 = jnp.arctan2(jnp.linalg.norm(jnp.cross(targVec_inTrack, satVec)), jnp.dot(targVec_inTrack, satVec)) * 180 / jnp.pi
    cross_track_angle2 = jnp.arctan2(jnp.linalg.norm(jnp.cross(targVec_crossTrack, satVec)), jnp.dot(targVec_crossTrack, satVec)) * 180 / jnp.pi
    
    return jnp.array([in_track_angle_deg, cross_track_angle_deg])