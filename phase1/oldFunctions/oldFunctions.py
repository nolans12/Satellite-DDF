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