# Import pre-defined libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, indeptEstimator, ddfEstimator
from sensorClass import sensor
from commClass import comms

#####################
# Environment # 1:
#####################
def create_environment():
    # Define a sensor model:
    sens = sensor(name = 'Sensor 1', fov = 115, bearingsError = np.array([0.05, 0.05]))
   
    # Define targets for the satellites to track:
    targetIDs = [1]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ddfEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs) 
    
    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 85, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 60, nu = 0, color='r')
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 30, nu = 0, color='y')

    sats = [sat1, sat2, sat3]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = 'k')
    targ2 = target(name = 'Targ2', targetID=2, cords = np.array([60,-45,200]), heading=90, speed=10, climbrate = 1, color = 'r')
    targs = [targ1]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 5000*u.km, minRange = 500*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)


# Plot the NEES and NIS results:
def plot_NEES_NIS(simData):

    # Now that the simulations are done, we can plot the results for NEES and NIS:
    def nested_dict():
        return defaultdict(list)

    # Now that the simulations are done, we can plot the results for NEES and NIS:
    nees_net = defaultdict(lambda: defaultdict(nested_dict))
    nis_net = defaultdict(lambda: defaultdict(nested_dict))
    # Just loop trough everything and append the data to the dictionaries:
    # Loop through all sims
    # Define a satellite vector we can loop through, want to add
    for i in range(numSims):
        # Loop through all the targets
        for targ in simData[i].keys():
            # Loop through all sats
            for sat in simData[i][targ].keys():
                # Loop through all times:
                for time in time_vec.to_value():
                    nees_data = simData[i][targ][sat]['NEES'][time]
                    nis_data = simData[i][targ][sat]['NIS'][time]
                    # If not empty, append to the dictionary
                    if nees_data and nis_data:
                        # Append to the data at that time:
                        if time not in nees_net[targ][sat].keys():
                            nees_net[targ][sat][time] = []
                        if time not in nis_net[targ][sat].keys():
                            nis_net[targ][sat][time] = []
                        nees_net[targ][sat][time].append(nees_data)
                        nis_net[targ][sat][time].append(nis_data)
                    
    # Now we can finally plot the NEES and NIS plots:
    # Goal is to make one plot for each target.
    # Each plot will have 2 subplots, one for NEES and one for NIS plots
    # The data on the plots will be the average NEES and NIS values for each satellite at each time step
    for targ in nees_net.keys():
        fig, axs = plt.subplots(1,2, figsize=(15, 8))
        fig.suptitle(f'Target {targ} NEES and NIS plots over {numSims} simulations', fontsize=16)
        axs[0].set_title('Average NEES vs Time')
        axs[0].set_xlabel('Time [min]')
        axs[0].set_ylabel('NEES')
        axs[1].set_title('Average NIS vs Time')
        axs[1].set_xlabel('Time [min]')
        axs[1].set_ylabel('NIS')
        for sat in nees_net[targ].keys():
            # Calculate the average NEES and NIS values for each time step
            nees_avg = np.array([np.mean(nees_net[targ][sat][time]) for time in nees_net[targ][sat].keys()])
            nis_avg = np.array([np.mean(nis_net[targ][sat][time]) for time in nis_net[targ][sat].keys()])
            # Plot the data
            axs[0].plot(nees_net[targ][sat].keys(), nees_avg, label=f'{sat}')
            axs[1].plot(nis_net[targ][sat].keys(), nis_avg, label=f'{sat}')

                        
        axs[0].legend()
        axs[1].legend()
        # Save the plots
        filePath = os.path.dirname(os.path.realpath(__file__))
        plotPath = os.path.join(filePath, 'plots')
        os.makedirs(plotPath, exist_ok=True)
        plt.savefig(os.path.join(plotPath,"NEES_NIS_results.png"), dpi=300)



def testCase_environment():
# Test Case Consists of:
#   Four Constellations in Polar Orbits at different right ascension of ascending node
#   Each Constellation has 5 satellites separated by 5 degrees in true anomaly
#       Each Satellite one sensors type a,b,c,d,e 
#       Each satellite tracks all targets
#       Each Satellite has a local estimator
#       Each Satellite has a data fusion algorithm
#
#   Five Targets at different locations
#       Target 1: 90 degrees latitude, 0 degrees longitude, 0 altitude, speed 5 m/s, heading 0 degrees, constant altitude 
#       Target 2: 0 degrees latitude, 0 degrees longitude, 200 altitude, speed 100 m/s, heading 90 degrees, constant altitude
#       Target 3: 45 degrees latitude, 45 degrees longitude, 100 altitude, speed 50 m/s, heading 45 degrees, constant altitude
#       Target 4: -45 degrees latitude, 45 degrees longitude, 0 altitude, speed 20 m/s, heading -45 degrees, constant altitude
#       Target 5: -90 degrees latitude, 45 degrees longitude, 0 altitude, speed 30 m/s, heading 90 degrees, constant altitude
#
#   Communication Network:
#       Each satellite can communicate with 3 nearest neighbors
#       Maximum range of communication is 5000 km
#       Minimum range of communication is 500 km
    
    
    # Define 5 sensor models with constant FOV and different bearings error matrixes:
    bearings_Error_matrix = np.array([[0.001,0.001], [0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [1, 1] ])
    
    sensor1 = sensor(name = 'Sensor A', fov = 115, bearingsError = bearings_Error_matrix[0])
    sensor2 = sensor(name = 'Sensor B', fov = 115, bearingsError = bearings_Error_matrix[1])
    sensor3 = sensor(name = 'Sensor C', fov = 115, bearingsError = bearings_Error_matrix[2])
    sensor4 = sensor(name = 'Sensor D', fov = 115, bearingsError = bearings_Error_matrix[3])
    sensor5 = sensor(name = 'Sensor E', fov = 115, bearingsError = bearings_Error_matrix[4])
    
    sensor6 = sensor(name = 'Sensor A', fov = 115, bearingsError = bearings_Error_matrix[0])
    sensor7 = sensor(name = 'Sensor B', fov = 115, bearingsError = bearings_Error_matrix[1])
    sensor8 = sensor(name = 'Sensor C', fov = 115, bearingsError = bearings_Error_matrix[2])
    sensor9 = sensor(name = 'Sensor D', fov = 115, bearingsError = bearings_Error_matrix[3])
    sensor10 = sensor(name = 'Sensor E', fov = 115, bearingsError = bearings_Error_matrix[4])
    
    sensor11 = sensor(name = 'Sensor A', fov = 115, bearingsError = bearings_Error_matrix[0])
    sensor12 = sensor(name = 'Sensor B', fov = 115, bearingsError = bearings_Error_matrix[1])
    sensor13 = sensor(name = 'Sensor C', fov = 115, bearingsError = bearings_Error_matrix[2])
    sensor14 = sensor(name = 'Sensor D', fov = 115, bearingsError = bearings_Error_matrix[3])
    sensor15 = sensor(name = 'Sensor E', fov = 115, bearingsError = bearings_Error_matrix[4])
    
    sensor16 = sensor(name = 'Sensor A', fov = 115, bearingsError = bearings_Error_matrix[0])
    sensor17 = sensor(name = 'Sensor B', fov = 115, bearingsError = bearings_Error_matrix[1])
    sensor18 = sensor(name = 'Sensor C', fov = 115, bearingsError = bearings_Error_matrix[2])
    sensor19 = sensor(name = 'Sensor D', fov = 115, bearingsError = bearings_Error_matrix[3])
    sensor20 = sensor(name = 'Sensor E', fov = 115, bearingsError = bearings_Error_matrix[4])

    # Define targets for the satellites to track:
    targetIDs = [1,2,3,4,5]

    # Define local estimators:
    local1 = indeptEstimator(targetIDs = targetIDs)
    local2 = indeptEstimator(targetIDs = targetIDs)
    local3 = indeptEstimator(targetIDs = targetIDs)
    local4 = indeptEstimator(targetIDs = targetIDs)
    local5 = indeptEstimator(targetIDs = targetIDs)
    
    local6 = indeptEstimator(targetIDs = targetIDs)
    local7 = indeptEstimator(targetIDs = targetIDs)
    local8 = indeptEstimator(targetIDs = targetIDs)
    local9 = indeptEstimator(targetIDs = targetIDs)
    local10 = indeptEstimator(targetIDs = targetIDs)
    
    # Define local estimators:
    local11 = indeptEstimator(targetIDs = targetIDs)
    local12 = indeptEstimator(targetIDs = targetIDs)
    local13 = indeptEstimator(targetIDs = targetIDs)
    local14 = indeptEstimator(targetIDs = targetIDs)
    local15 = indeptEstimator(targetIDs = targetIDs)
    
    local16 = indeptEstimator(targetIDs = targetIDs)
    local17 = indeptEstimator(targetIDs = targetIDs)
    local18 = indeptEstimator(targetIDs = targetIDs)
    local19 = indeptEstimator(targetIDs = targetIDs)
    local20 = indeptEstimator(targetIDs = targetIDs)
    
    
    central = centralEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm
    ddfEstimatorAlg1 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg2 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg3 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg4 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg5 = ddfEstimator(targetIDs = targetIDs)
    
    ddfEstimatorAlg6 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg7 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg8 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg9 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg10 = ddfEstimator(targetIDs = targetIDs)
    
    ddfEstimatorAlg11 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg12 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg13 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg14 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg15 = ddfEstimator(targetIDs = targetIDs)
    
    ddfEstimatorAlg16 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg17 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg18 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg19 = ddfEstimator(targetIDs = targetIDs)
    ddfEstimatorAlg20 = ddfEstimator(targetIDs = targetIDs)
    
    # Define the satellites:
    # Constellation 1: Polar Orbit at 0 degrees right ascension of ascending node
    sat1 = satellite(name = 'Sat1', sensor = sensor1, targetIDs=targetIDs, indeptEstimator = local1, ddfEstimator=ddfEstimatorAlg1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = sensor2, targetIDs=targetIDs, indeptEstimator = local2, ddfEstimator=ddfEstimatorAlg2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 10, nu = 0, color='b')
    sat3 = satellite(name = 'Sat3', sensor = sensor3, targetIDs=targetIDs, indeptEstimator = local3, ddfEstimator=ddfEstimatorAlg3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 20, nu = 0, color='b')
    sat4 = satellite(name = 'Sat4', sensor = sensor4, targetIDs=targetIDs, indeptEstimator = local4, ddfEstimator=ddfEstimatorAlg4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 30, nu = 0, color='b')
    sat5 = satellite(name = 'Sat5', sensor = sensor5, targetIDs=targetIDs, indeptEstimator = local5, ddfEstimator=ddfEstimatorAlg5, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 40, nu = 0, color='b')
    
    # Constellation 2: Polar Orbit at 30 degrees right ascension of ascending node
    sat6 = satellite(name = 'Sat6', sensor = sensor6, targetIDs=targetIDs, indeptEstimator = local6, ddfEstimator=ddfEstimatorAlg6, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 0, color='r')
    sat7 = satellite(name = 'Sat7', sensor = sensor7, targetIDs=targetIDs, indeptEstimator = local7, ddfEstimator=ddfEstimatorAlg7, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 10, nu = 0, color='r')
    sat8 = satellite(name = 'Sat8', sensor = sensor8, targetIDs=targetIDs, indeptEstimator = local8, ddfEstimator=ddfEstimatorAlg8, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 20, nu = 0, color='r')
    sat9 = satellite(name = 'Sat9', sensor = sensor9, targetIDs=targetIDs, indeptEstimator = local9, ddfEstimator=ddfEstimatorAlg9, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 30, nu = 0, color='r')
    sat10 = satellite(name = 'Sat10', sensor = sensor10, targetIDs=targetIDs, indeptEstimator = local10, ddfEstimator=ddfEstimatorAlg10, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 40, nu = 0, color='r')
    
    # Constellation 3: Polar Orbit at 60 degrees right ascension of ascending node
    sat11 = satellite(name = 'Sat11', sensor = sensor11, targetIDs=targetIDs, indeptEstimator = local11, ddfEstimator=ddfEstimatorAlg11, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = -30, color='g')
    sat12 = satellite(name = 'Sat12', sensor = sensor12, targetIDs=targetIDs, indeptEstimator = local12, ddfEstimator=ddfEstimatorAlg12, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 10, nu = -30, color='g')
    sat13 = satellite(name = 'Sat13', sensor = sensor13, targetIDs=targetIDs, indeptEstimator = local13, ddfEstimator=ddfEstimatorAlg13, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 20, nu = -30, color='g')
    sat14 = satellite(name = 'Sat14', sensor = sensor14, targetIDs=targetIDs, indeptEstimator = local14, ddfEstimator=ddfEstimatorAlg14, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 30, nu = -30, color='g')
    sat15 = satellite(name = 'Sat15', sensor = sensor15, targetIDs=targetIDs, indeptEstimator = local15, ddfEstimator=ddfEstimatorAlg15, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 40, nu = -30, color='g')
    
    # Constellation 4: Polar Orbit at 90 degrees right ascension of ascending node
    sat16 = satellite(name = 'Sat16', sensor = sensor16, targetIDs=targetIDs, indeptEstimator = local16, ddfEstimator=ddfEstimatorAlg16, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 10, color='c')
    sat17 = satellite(name = 'Sat17', sensor = sensor17, targetIDs=targetIDs, indeptEstimator = local17, ddfEstimator=ddfEstimatorAlg17, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 10, nu = 10, color='c')
    sat18 = satellite(name = 'Sat18', sensor = sensor18, targetIDs=targetIDs, indeptEstimator = local18, ddfEstimator=ddfEstimatorAlg18, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 20, nu = 10, color='c')
    sat19 = satellite(name = 'Sat19', sensor = sensor19, targetIDs=targetIDs, indeptEstimator = local19, ddfEstimator=ddfEstimatorAlg19, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 30, nu = 10, color='c')
    sat20 = satellite(name = 'Sat20', sensor = sensor20, targetIDs=targetIDs, indeptEstimator = local20, ddfEstimator=ddfEstimatorAlg20, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 40, nu = 10, color='c')
    
    sats = [sat1, sat2, sat3, sat4, sat5, sat6, sat7, sat8, sat9, sat10, sat11, sat12, sat13, sat14, sat15, sat16, sat17, sat18, sat19, sat20]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = 'k')
    targ2 = target(name = 'Targ2', targetID=2, cords = np.array([45,15,200]), heading=0, speed=5, climbrate = 0, color = 'k')
    targ3 = target(name = 'Targ3', targetID=3, cords = np.array([0,30,100]), heading=0, speed=5, climbrate = 0, color = 'k')
    targ4 = target(name = 'Targ4', targetID=4, cords = np.array([-45,60,50]), heading=0, speed=5, climbrate = 0, color = 'k')
    targ5 = target(name = 'Targ5', targetID=5, cords = np.array([-90,90,0]), heading=0, speed=5, climbrate = 0, color = 'k')
    
    
    targs = [targ1, targ2, targ3, targ4, targ5]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 5000*u.km, minRange = 500*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

if __name__ == "__main__":
    # Vector of time for simulation:
    time_vec = np.linspace(0, 10, 61) * u.minute

    # Number of simulations:
    numSims = 1
    simData = defaultdict(dict)
    for i in range(numSims):
        print(f'Simulation {i + 1} out of {numSims}')
        # Create a new environment instance for each simulation run:
        env = create_environment()
        # Simulate the satellites through the vector of time:
        simData[i] = env.simulate(time_vec, savePlot = True, saveName = "CI", showSim = True)

    
    # env = testCase_environment()
    # env.simulate(time_vec, savePlot = True, saveName = str(1), showSim = True)
        
    # Plot the NEES and NIS results:
    # plot_NEES_NIS(simData)
    
    # Save the gif:
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)