# Import pre-definied libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from environmentClass import environment

if __name__ == "__main__":

# Define two satellites instances
    sat1 = satellite(name = 'Sat 1', a = 7000, ecc = 0.1, inc = 0, raan = 0, argp = 0, nu = 0, fov = 45, sensorDetectError = 0.1, sensorError = 0.1, color='b')
    sat2 = satellite(name = 'Sat 2', a = 7000, ecc = 0.2, inc = 0, raan = 0, argp = 0, nu = 0, fov = 45, sensorDetectError = 0.1, sensorError = 0.1, color='r')

# Create an environment instance with the two satellites
    env = environment([sat1, sat2])

# Simulate the satellites through a vector of time
    time_vec = np.linspace(0, 10, 100) * u.hour
    env.simulate(time_vec, display=True)

# Save the gif
    env.render_gif(fileName = 'satellite_simulation.gif')