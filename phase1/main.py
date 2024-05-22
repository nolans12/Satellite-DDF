# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, localEstimator
from sensorClass import sensor

if __name__ == "__main__":

# Define a sensor model:
    sens1 = sensor(fov = 115, sensorError = np.array([1, 1]), detectError= 0.05, resolution = 720, name = 'Sensor 1')

# Define targets for the satellites to track:
    targetIDs1 = np.array([1])

# Define local estimators:
    local1 = localEstimator(targetIDs = targetIDs1)
    
# Define some polar orbits at 1000 km altitude:
    sat1 = satellite(name = 'Sat1', sensor = sens1, targetIDs=targetIDs1, estimator = local1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 70, nu = 0, color='b')
    
    sats = [sat1]

# Define some targets:
    targ1 = target(name = 'Targ1', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
   
    targs = [targ1]
    
# Create an estimator instance with the satellites and targets:
    central = centralEstimator(sats, targs)

# Create an environment instance:
    env = environment(sats, targs)

# Simulate the satellites through a vector of time:
    time_vec = np.linspace(0, 10, 21) * u.minute
    env.simulate(time_vec, display = True)

# Save the gif:
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)