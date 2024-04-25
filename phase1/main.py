# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import estimator

if __name__ == "__main__":

# Define some polar orbits at 1000 km altitude
    sat1 = satellite(name = 'Sat 1', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 45, nu = 0, fovNarrow = 50, fovWide = 50, sensorDetectError = 0.1, sensorError = 0.1, color='b')
    sat2 = satellite(name = 'Sat 2', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, fovNarrow = 50, fovWide = 50, sensorDetectError = 0.1, sensorError = 0.1, color='r')
    sat3 = satellite(name = 'Sat 3', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -45, nu = 0, fovNarrow = 50, fovWide = 50, sensorDetectError = 0.1, sensorError = 0.1, color='g')
    sat4 = satellite(name = 'Sat 4', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -90, nu = 0, fovNarrow = 50, fovWide = 50, sensorDetectError = 0.1, sensorError = 0.1, color='m')

# Define some targets
    targ1 = target(name = 'Targ 1', targetID=1, pos = np.array([0, 0 , 6378]), vel = np.array([4000, 0, 0]), color = 'k')
    targ2 = target(name = 'Targ 2', targetID=2, pos = np.array([0, 0, 6378]), vel=np.array([0, 4000, 0]), color = 'y')

    

# Create an estimator instance with the satellites and targets
    est = estimator([sat1, sat2, sat3, sat4], [targ1, targ2])

# Create an environment instance 
    env = environment([sat1, sat2, sat3, sat4], [targ1, targ2], est)

# Simulate the satellites through a vector of time
    time_vec = np.linspace(0, 50, 26) * u.minute
    env.simulate(time_vec, display = True)

# Save the gif
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)