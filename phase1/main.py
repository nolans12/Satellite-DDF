# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, localEstimator
from sensorClass import sensor
from commClass import comms

if __name__ == "__main__":

# DEFINE THE SATELLITE OBJECTS:

    # Define a sensor model:
    sens1 = sensor(name = 'Sensor 1', fov = 115, sensorError = np.array([0, 0]), detectError= 0.05, resolution = 720)
    sens2 = sensor(name = 'Sensor 2', fov = 100, sensorError = np.array([2, 2]), detectError= 0.05, resolution = 720)

    # Define targets for the satellites to track:
    # TODO: should we just make the satellite track any target it can see?
    targetIDs = [1]

    # Define local estimators:
    local1 = localEstimator(targetIDs = targetIDs)
    local2 = localEstimator(targetIDs = targetIDs)

    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = sens1, targetIDs=targetIDs, estimator = local1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 45, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = sens2, targetIDs=targetIDs, estimator = local2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 30, nu = 0, color='r')
    
    sats = [sat1, sat2]

# DEFINE THE TARGET OBJECTS:
    targ1 = target(name = 'Targ1', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
     
    targs = [targ1]
    
# Create an estimator instance with the satellites and targets:
    central = centralEstimator(sats, targs)

# Define the communication network:
    comms = comms(sats, range = 5000 * u.km, displayStruct = True)

# Create an environment instance:
    env = environment(sats, targs, comms)

# Simulate the satellites through a vector of time:
    time_vec = np.linspace(0, 30, 31) * u.minute
    env.simulate(time_vec, display = True)

# Save the gif:
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)


### CLUSTER SIM BELOW:

# # DEFINE THE SATELLITE OBJECTS:

#     # Define a sensor model:
#     sens1_1 = sensor(name = 'Sensor 1.1', fov = 115, sensorError = np.array([0, 0]), detectError= 0.05, resolution = 720)
#     sens1_2 = sensor(name = 'Sensor 1.2', fov = 115, sensorError = np.array([0, 0]), detectError= 0.05, resolution = 720)
#     sens1_3 = sensor(name = 'Sensor 1.3', fov = 115, sensorError = np.array([0, 0]), detectError= 0.05, resolution = 720)
#     sens1_4 = sensor(name = 'Sensor 1.4', fov = 115, sensorError = np.array([0, 0]), detectError= 0.05, resolution = 720)

#     sens2_1 = sensor(name = 'Sensor 2.1', fov = 100, sensorError = np.array([2, 2]), detectError= 0.05, resolution = 720)
#     sens2_2 = sensor(name = 'Sensor 2.2', fov = 100, sensorError = np.array([2, 2]), detectError= 0.05, resolution = 720)
#     sens2_3 = sensor(name = 'Sensor 2.3', fov = 100, sensorError = np.array([2, 2]), detectError= 0.05, resolution = 720)
#     sens2_4 = sensor(name = 'Sensor 2.4', fov = 100, sensorError = np.array([2, 2]), detectError= 0.05, resolution = 720)
    
#     # Define targets for the satellites to track:
#     # TODO: should we just make the satellite track any target it can see?
#     targetIDs = [1]


#     # Define local estimators:
#     local1_1 = localEstimator(targetIDs = targetIDs)
#     local1_2 = localEstimator(targetIDs = targetIDs)
#     local1_3 = localEstimator(targetIDs = targetIDs)
#     local1_4 = localEstimator(targetIDs = targetIDs)
    
#     local2_1 = localEstimator(targetIDs = targetIDs)
#     local2_2 = localEstimator(targetIDs = targetIDs)
#     local2_3 = localEstimator(targetIDs = targetIDs)
#     local2_4 = localEstimator(targetIDs = targetIDs)
    
#     ## Cluster 1: Polar orbits:
#     sat1_1 = satellite(name = 'Sat1', sensor = sens1_1, targetIDs=targetIDs, estimator = local1_1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -90, nu = 0, color='b')
#     sat1_2 = satellite(name = 'Sat2', sensor = sens1_2, targetIDs=targetIDs, estimator = local1_2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -70, nu = 0, color='b')
#     sat1_3 = satellite(name = 'Sat3', sensor = sens1_3, targetIDs=targetIDs, estimator = local1_3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -50, nu = 0, color='b')
#     sat1_4 = satellite(name = 'Sat4', sensor = sens1_4, targetIDs=targetIDs, estimator = local1_4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -30, nu = 0, color='b')

#     ## Cluster 2: Equatorial orbits:
#     sat2_1 = satellite(name = 'Sat5', sensor = sens2_1, targetIDs=targetIDs, estimator = local2_1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -180, nu = 0, color='r')
#     sat2_2 = satellite(name = 'Sat6', sensor = sens2_2, targetIDs=targetIDs, estimator = local2_2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -160, nu = 0, color='r')
#     sat2_3 = satellite(name = 'Sat7', sensor = sens2_3, targetIDs=targetIDs, estimator = local2_3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -140, nu = 0, color='r')
#     sat2_4 = satellite(name = 'Sat8', sensor = sens2_4, targetIDs=targetIDs, estimator = local2_4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -120, nu = 0, color='r')
    
#     sats = [sat1_1, sat1_2, sat1_3, sat1_4, sat2_1, sat2_2, sat2_3, sat2_4]

# # DEFINE THE TARGET OBJECTS:
#     targ1 = target(name = 'Targ1', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
     
#     targs = [targ1]
    
# # Create an estimator instance with the satellites and targets:
#     central = centralEstimator(sats, targs)

# # Define the communication network:
#     comms = comms(sats, range = 5000 * u.km, displayStruct = True)

# # Create an environment instance:
#     env = environment(sats, targs, comms)

# # Simulate the satellites through a vector of time:
#     time_vec = np.linspace(0, 50, 51) * u.minute
#     env.simulate(time_vec, display = True)

# # Save the gif:
#     env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)
