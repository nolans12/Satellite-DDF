## THIS IS A SANDBOX ENVIRONMENT FOR TESTING THE OPTIMIZATION OF LAZY ET-DDF ##
## Assume that no filters are running, only satellites and targets existing ##

from import_libraries import *

from environmentClass import environment
from satelliteClass import satellite
from sensorClass import sensor
from targetClass import target
from commClass import comms

if __name__ == "__main__":

    # Will declare 8 satellites in the sky
    sens = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.01, 115 * 0.01])) 
    # sat1a = satellite(name = 'Sat1', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 45, nu = 0, color='#ff0000')
    # sat1b = satellite(name = 'Sat1b', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 30, nu = 0, color='#ff0000')
    # sat1c = satellite(name = 'Sat1c', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 15, nu = 0, color='#ff0000')
    # sat1d = satellite(name = 'Sat1d', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 0, nu = 0, color='#ff0000')
    # sat2a = satellite(name = 'Sat2a', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 45 + 7, nu = 0, color='#0000ff')
    # sat2b = satellite(name = 'Sat2b', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 30 + 7, nu = 0, color='#0000ff')
    # sat2c = satellite(name = 'Sat2c', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 15 + 7, nu = 0, color='#0000ff')
    # sat2d = satellite(name = 'Sat2d', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 0 + 7, nu = 0, color='#0000ff')

    # sats = [sat1a, sat1b, sat1c, sat1d, sat2a, sat2b, sat2c, sat2d]

    sat1a = satellite(name = 'Sat1a', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 45, nu = 0, color='#ff0000')
    sat1b = satellite(name = 'Sat1b', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 30, nu = 0, color='#ff0000')
    sat2a = satellite(name = 'Sat2a', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 45, nu = 0, color='#0000ff')
    sat2b = satellite(name = 'Sat2b', sensor = deepcopy(sens), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 30, nu = 0, color='#0000ff')

    sats = [sat1a, sat1b, sat2a, sat2b]


    # Define 10 targets on the ground
    targ1_color = '#e71714'
    targ2_color = '#eea62a'
    targ3_color = '#58b428'
    targ4_color = '#2879b4'
    targ5_color = '#b228b4'
    targ6_color = '#b4285f'
    targ7_color = '#b428b4'
    targ8_color = '#b4a228'
    targ9_color = '#28b4a4'
    targ10_color = '#b4a428'

    # targ1 = target(name = 'Targ1', targetID=1, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 3, 0, 90, 0.1]), color = targ1_color)
    # targ2 = target(name = 'Targ2', targetID=2, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ2_color)
    # targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ3_color)
    # targ4 = target(name = 'Targ4', targetID=4, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ4_color)
    # targ5 = target(name = 'Targ5', targetID=5, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ5_color)
    # targ6 = target(name = 'Targ6', targetID=6, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ6_color)
    # targ7 = target(name = 'Targ7', targetID=7, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ7_color)
    # targ8 = target(name = 'Targ8', targetID=8, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ8_color)
    # targ9 = target(name = 'Targ9', targetID=9, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ9_color)
    # targ10 = target(name = 'Targ10', targetID=10, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([10, 10, 0, 90, 0.1]), color = targ10_color)

    # targs = [targ1, targ2, targ3, targ4, targ5, targ6, targ7, targ8, targ9, targ10]

    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([0, 0, 0, 90, 0.1]), color = targ1_color)
    
    targs = [targ1]

    # Define the communication network
    comms_network = comms(sats, maxBandwidth = 99999999, maxNeighbors = 2, maxRange = 10000*u.km, minRange = 500*u.km, displayStruct = True)

    # Define the environment
    env = environment(sats, targs, comms_network, [])

    # Simulate the environment
    time_vec = np.linspace(6, 10, 10*12 + 1) * u.minute
    env.simulate(time_vec = time_vec, pause_step = 0.1)

    env.render_gif(fps = 5)