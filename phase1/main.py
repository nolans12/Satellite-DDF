# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import estimator

if __name__ == "__main__":

# Define some polar orbits at 1000 km altitude
    sat1 = satellite(name = 'Sat 1', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 0, fov = 100, sensorError = 5, color='b')
    # sat2 = satellite(name = 'Sat 2', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, fov = 100, sensorError = 5, color='r')
    # sat3 = satellite(name = 'Sat 3', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -45, nu = 0, fov = 100, sensorError = 5, color='g')
    # sat4 = satellite(name = 'Sat 4', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -90, nu = 0, fov = 100, sensorError = 5, color='m')

# Define some targets
    targ1 = target(name = 'Targ 1', targetID=1, pos = np.array([0, 0 , 6378]), vel = np.array([4000, 0, 0]), r = np.array([6378, 0, 0, 0, 0,0]),color = 'k')
    targ2 = target(name = 'Targ 2', targetID=2, pos = np.array([0, 0, 6378]), vel = np.array([0, 4000, 0]), r = np.array([0, 0, 6378, 0, 0,0]), color = 'y')

    

# Create an estimator instance with the satellites and targets
    est = estimator([sat1], [targ1, targ2])
    # est = estimator([sat1, sat2, sat3, sat4], [targ1, targ2])

# Create an environment instance 
    env = environment([sat1], [targ1, targ2], est)
    # env = environment([sat1, sat2, sat3, sat4], [targ1, targ2], est)
    
# Simulate the satellites through a vector of time
    time_vec = np.linspace(0, 20, 21) * u.minute
    env.simulate(time_vec, display = True)

# Plot the results:
    # env.plotResults()

# Save the gif
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)

# at 120 should see whole earth

# # Define some polar orbits/sats
#     sat1 = satellite(name = 'Sat 1', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 70, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 5, color='b')
#     sat2 = satellite(name = 'Sat 2', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 60, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 2.5, color='r')
#     sat3 = satellite(name = 'Sat 3', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 50, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 10, color='g')
#     sat4 = satellite(name = 'Sat 4', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 40, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 15, color='m')
#     sat5 = satellite(name = 'Sat 5', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 30, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 5, color='springgreen')
#     sat6 = satellite(name = 'Sat 6', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 20, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 2.5, color='steelblue')
#     sat7 = satellite(name = 'Sat 7', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 10, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 5, color='thistle')
#     sat8 = satellite(name = 'Sat 8', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, fovNarrow = 120, fovWide = 120, sensorError = 15, color='moccasin')

# # Define some targets
#     targ1 = target(name = 'Targ 1', targetID=1, pos = np.array([0, 0, 6378]), vel=np.array([0, -5, -5]), color = 'k')
#     targ2 = target(name = 'Targ 2', targetID=2, pos = np.array([0, 0, 6378]), vel=np.array([0, 5, 5]), color = 'y')
#     targ3 = target(name = 'Targ 3', targetID=3, pos = np.array([0, 0, 6378]), vel=np.array([-10, -10, 10]), color = 'c')
#     targ4 = target(name = 'Targ 4', targetID=4, pos = np.array([0, 0, 6378]), vel=np.array([10, 10, 10]), color = 'm')

# # Create an estimator instance with the satellites and targets
#     est = estimator([sat1, sat2, sat3, sat4, sat5, sat6, sat7, sat8], [targ1, targ2, targ3, targ4])

# # Create an environment instance 
#     env = environment([sat1, sat2, sat3, sat4, sat5, sat6, sat7, sat8], [targ1, targ2, targ3, targ4], est)
    
# # Simulate the satellites through a vector of time
#     time_vec = np.linspace(0, 30, 61) * u.minute
#     env.simulate(time_vec, display = True)

# # Plot the results:
#     env.plotResults()

# # Save the gif
#     env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)