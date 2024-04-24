# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from environmentClass import environment

if __name__ == "__main__":


# Define two satellites instances, polar orbits at around 1000 km altitude above
    sat1 = satellite(name = 'Sat 1', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = -90, nu = 0, fov = 50, sensorDetectError = 0.1, sensorError = 0.1, color='b')
    sat2 = satellite(name = 'Sat 2', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = -45, nu = 0, fov = 50, sensorDetectError = 0.1, sensorError = 0.1, color='r')
    sat3 = satellite(name = 'Sat 3', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 0, fov = 50, sensorDetectError = 0.1, sensorError = 0.1, color='g')
    sat4 = satellite(name = 'Sat 4', a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 45, nu = 0, fov = 50, sensorDetectError = 0.1, sensorError = 0.1, color='m')

# Create an environment instance with the two satellites
    env = environment([sat1, sat2, sat3, sat4])

# Simulate the satellites through a vector of time
    time_vec = np.linspace(0, 1, 50) * u.hour
    env.simulate(time_vec, display=True)

# Save the gif
    env.render_gif(fileName = 'satellite_simulation.gif')