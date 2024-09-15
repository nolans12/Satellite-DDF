# Import pre-defined libraries
from import_libraries import *

# Import classes
from environmentClass import environment
from satelliteClass import satellite
from sensorClass import sensor
from targetClass import target
from estimatorClass import indeptEstimator, centralEstimator, ciEstimator, etEstimator, gsEstimator
from commClass import comms
from groundStationClass import groundStation

### This environment is used for the base case, with 12 satellites, all with different track qualitys being tracked by 4 satellites from 2 different constellations
def create_environment():

    # Define the targets for the satellites to track:
    targ1_color = '#e71714'
    targ2_color = '#eea62a'
    targ3_color = '#58b428'
    targ4_color = '#2879b4'
    targ5_color = '#b228b4'
    
    # Define the targets
    targ1 = target(name = 'Targ1', tqReq = 1, targetID=1, coords = np.array([45,0,0]), heading=0, speed= 80, uncertainty=np.array([3, 3, 0, 90, 0.1]), color = targ1_color)
    targ2 = target(name = 'Targ2', tqReq = 2, targetID=2, coords = np.array([45,0,0]), heading=0, speed= 50, uncertainty=np.array([3, 3, 0, 90, 0.1]), color = targ2_color)
    targ3 = target(name = 'Targ3', tqReq = 3, targetID=3, coords = np.array([45,0,0]), heading=0, speed= 40, uncertainty=np.array([3, 3, 0, 90, 0.1]), color = targ3_color)
    targ4 = target(name = 'Targ4', tqReq = 4, targetID=4, coords = np.array([45,0,0]), heading=0, speed= 30, uncertainty=np.array([3, 3, 0, 90, 0.1]), color = targ4_color)
    targ5 = target(name = 'Targ5', tqReq = 5, targetID=5, coords = np.array([45,0,0]), heading=0, speed= 20, uncertainty=np.array([3, 3, 0, 90, 0.1]), color = targ5_color)
   
    targs = [targ1, targ2, targ3, targ4, targ5]


    # Define the satellite structure:
    sens_good = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.01, 115 * 0.01])) # 1% error on FOV bearings
    sens_bad = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.1, 115 * 0.1])) # 10% error on FOV bearings

    sat1a = satellite(name = 'Sat1a', sensor = deepcopy(sens_good), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 45, nu = 0, color='#669900')
    sat1b = satellite(name = 'Sat1b', sensor = deepcopy(sens_good), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 30, nu = 0, color='#66a3ff')
    sat2a = satellite(name = 'Sat2a', sensor = deepcopy(sens_bad), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 45 + 7, nu = 0, color='#9966ff')
    sat2b = satellite(name = 'Sat2b', sensor = deepcopy(sens_bad), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 30 + 7, nu = 0, color='#ffff33')

    sats = [sat1a, sat1b, sat2a, sat2b]
    # sats = [sat1a]


    # Define the goal of the system:
    commandersIntent = NestedDict()
    
    # For minute 0+: We want the following tracks:
    commandersIntent[0] = {sat1a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300}, 
                           sat1b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},  
                           sat2a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300}, 
                           sat2b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300}}

    # commandersIntent[0] = {sat1a: {1: 100}}
    

    # Define the ground stations
    # gs1 = groundStation(lat = 55, long = 10, fov = 90, commRange = 2500, estimator = gsEstimator(commandersIntent[0][sat1a]), name = 'G1', color = 'black')
    # gs2 = groundStation(lat = 35, long = -10, fov = 90, commRange = 2500, estimator = gsEstimator(commandersIntent[0][sat1a]), name = 'G2', color = 'gray')

    gs1 = groundStation(lat = 60, long = 10, fov = 80, commRange = 5000, estimator = gsEstimator(commandersIntent[0][sat1a]), name = 'G1', color = 'black')
    gs2 = groundStation(lat = 35, long = -15, fov = 80, commRange = 5000, estimator = gsEstimator(commandersIntent[0][sat1a]), name = 'G2', color = 'gray')

    groundStations = [gs1, gs2]
    # groundStations = [gs1]

    # Define the communication network: 
    comms_network = comms(sats, maxBandwidth = 30, maxNeighbors = 3, maxRange = 10000*u.km, minRange = 500*u.km, displayStruct = True)


    # Define the estimators used:
    central = False
    local = True
    ci = True 
    et = False

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, groundStations, commandersIntent, localEstimatorBool=local, centralEstimatorBool=central, ciEstimatorBool=ci, etEstimatorBool=et)


### Main code to run the simulation
if __name__ == "__main__":

    # Fix random seed for reproducibility
    np.random.seed(0)

    # Vector of time for simulation:
    time_vec = np.linspace(0, 10, 10*24 + 1) * u.minute

    # Header name for the plots, gifs, and data
    fileName = "Best Sat Sends CI, Network Used Optimal CI"

    # Create the environment
    env = create_environment()

    # Simulate the satellites through the vector of time:
    env.simulate(time_vec, saveName = fileName, show_env = False, plot_communication_results = True, plot_groundStation_results = True)

    # Save gifs:
    # env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 5)
    # env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 5)
    # env.render_gif(fileType='dynamic_comms', saveName=fileName, fps = 1)

