# Import pre-defined libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, localEstimator, dataFusion
from sensorClass import sensor
from commClass import comms

#####################
# Environment # 1:
#####################
def create_environment():
    # Define a sensor model:
    sens1 = sensor(name = 'Sensor 1', fov = 115, bearingsError = np.array([0.05, 0.05]))
    sens2 = sensor(name = 'Sensor 2', fov = 115, bearingsError = np.array([0.1, 0.1]))

    # Define targets for the satellites to track:
    targetIDs = [1]

    # Define local estimators:
    local1 = localEstimator(targetIDs = targetIDs)
    local2 = localEstimator(targetIDs = targetIDs)
    central = centralEstimator(targetIDs = targetIDs) # TODO: why not just make centralized always do all targets? since it is the baseline?

    # Define the Data Fusion Algorithm
    dataFusionAlg = dataFusion(targetIDs = targetIDs)
    
    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = sens1, targetIDs=targetIDs, estimator = local1, dataFusion=dataFusionAlg, a = Earth.R + 1000 * u.km, ecc = 0, inc = 45, raan = 0, argp = 80, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = sens2, targetIDs=targetIDs, estimator = local2, dataFusion=dataFusionAlg, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 80, nu = 0, color='r')

    sats = [sat1, sat2]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = 'k')
    #targ2 = target(name = 'Targ2', targetID=2, cords = np.array([0,0,200]), heading=90, speed=100, climbrate = 1, color = 'r')
    targs = [targ1]#, targ2]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 5000*u.km, minRange = 500*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)



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
    local1 = localEstimator(targetIDs = targetIDs)
    local2 = localEstimator(targetIDs = targetIDs)
    local3 = localEstimator(targetIDs = targetIDs)
    local4 = localEstimator(targetIDs = targetIDs)
    local5 = localEstimator(targetIDs = targetIDs)
    
    local6 = localEstimator(targetIDs = targetIDs)
    local7 = localEstimator(targetIDs = targetIDs)
    local8 = localEstimator(targetIDs = targetIDs)
    local9 = localEstimator(targetIDs = targetIDs)
    local10 = localEstimator(targetIDs = targetIDs)
    
    # Define local estimators:
    local11 = localEstimator(targetIDs = targetIDs)
    local12 = localEstimator(targetIDs = targetIDs)
    local13 = localEstimator(targetIDs = targetIDs)
    local14 = localEstimator(targetIDs = targetIDs)
    local15 = localEstimator(targetIDs = targetIDs)
    
    local16 = localEstimator(targetIDs = targetIDs)
    local17 = localEstimator(targetIDs = targetIDs)
    local18 = localEstimator(targetIDs = targetIDs)
    local19 = localEstimator(targetIDs = targetIDs)
    local20 = localEstimator(targetIDs = targetIDs)
    
    
    central = centralEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm
    dataFusionAlg1 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg2 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg3 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg4 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg5 = dataFusion(targetIDs = targetIDs)
    
    dataFusionAlg6 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg7 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg8 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg9 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg10 = dataFusion(targetIDs = targetIDs)
    
    dataFusionAlg11 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg12 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg13 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg14 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg15 = dataFusion(targetIDs = targetIDs)
    
    dataFusionAlg16 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg17 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg18 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg19 = dataFusion(targetIDs = targetIDs)
    dataFusionAlg20 = dataFusion(targetIDs = targetIDs)
    
    # Define the satellites:
    # Constellation 1: Polar Orbit at 0 degrees right ascension of ascending node
    sat1 = satellite(name = 'Sat1', sensor = sensor1, targetIDs=targetIDs, estimator = local1, dataFusion=dataFusionAlg1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = sensor2, targetIDs=targetIDs, estimator = local2, dataFusion=dataFusionAlg2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 10, nu = 0, color='r')
    sat3 = satellite(name = 'Sat3', sensor = sensor3, targetIDs=targetIDs, estimator = local3, dataFusion=dataFusionAlg3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 20, nu = 0, color='g')
    sat4 = satellite(name = 'Sat4', sensor = sensor4, targetIDs=targetIDs, estimator = local4, dataFusion=dataFusionAlg4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 30, nu = 0, color='c')
    sat5 = satellite(name = 'Sat5', sensor = sensor5, targetIDs=targetIDs, estimator = local5, dataFusion=dataFusionAlg5, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 40, nu = 0, color='m')
    
    # Constellation 2: Polar Orbit at 30 degrees right ascension of ascending node
    sat6 = satellite(name = 'Sat6', sensor = sensor6, targetIDs=targetIDs, estimator = local6, dataFusion=dataFusionAlg6, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 0, color='b')
    sat7 = satellite(name = 'Sat7', sensor = sensor7, targetIDs=targetIDs, estimator = local7, dataFusion=dataFusionAlg7, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 10, nu = 0, color='r')
    sat8 = satellite(name = 'Sat8', sensor = sensor8, targetIDs=targetIDs, estimator = local8, dataFusion=dataFusionAlg8, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 20, nu = 0, color='g')
    sat9 = satellite(name = 'Sat9', sensor = sensor9, targetIDs=targetIDs, estimator = local9, dataFusion=dataFusionAlg9, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 30, nu = 0, color='c')
    sat10 = satellite(name = 'Sat10', sensor = sensor10, targetIDs=targetIDs, estimator = local10, dataFusion=dataFusionAlg10, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 40, nu = 0, color='m')
    
    # Constellation 3: Polar Orbit at 60 degrees right ascension of ascending node
    sat11 = satellite(name = 'Sat11', sensor = sensor11, targetIDs=targetIDs, estimator = local11, dataFusion=dataFusionAlg11, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = -30, color='b')
    sat12 = satellite(name = 'Sat12', sensor = sensor12, targetIDs=targetIDs, estimator = local12, dataFusion=dataFusionAlg12, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 10, nu = -30, color='r')
    sat13 = satellite(name = 'Sat13', sensor = sensor13, targetIDs=targetIDs, estimator = local13, dataFusion=dataFusionAlg13, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 20, nu = -30, color='g')
    sat14 = satellite(name = 'Sat14', sensor = sensor14, targetIDs=targetIDs, estimator = local14, dataFusion=dataFusionAlg14, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 30, nu = -30, color='c')
    sat15 = satellite(name = 'Sat15', sensor = sensor15, targetIDs=targetIDs, estimator = local15, dataFusion=dataFusionAlg15, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 40, nu = -30, color='m')
    
    # Constellation 4: Polar Orbit at 90 degrees right ascension of ascending node
    sat16 = satellite(name = 'Sat16', sensor = sensor16, targetIDs=targetIDs, estimator = local16, dataFusion=dataFusionAlg16, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 10, color='b')
    sat17 = satellite(name = 'Sat17', sensor = sensor17, targetIDs=targetIDs, estimator = local17, dataFusion=dataFusionAlg17, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 10, nu = 10, color='r')
    sat18 = satellite(name = 'Sat18', sensor = sensor18, targetIDs=targetIDs, estimator = local18, dataFusion=dataFusionAlg18, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 20, nu = 10, color='g')
    sat19 = satellite(name = 'Sat19', sensor = sensor19, targetIDs=targetIDs, estimator = local19, dataFusion=dataFusionAlg19, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 30, nu = 10, color='c')
    sat20 = satellite(name = 'Sat20', sensor = sensor20, targetIDs=targetIDs, estimator = local20, dataFusion=dataFusionAlg20, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 40, nu = 10, color='m')
    
    sats = [sat1, sat2, sat3, sat4, sat5, sat6, sat7, sat8, sat9, sat10, sat11, sat12, sat13, sat14, sat15, sat16, sat17, sat18, sat19, sat20]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([90,0,0]), heading=0, speed=50, climbrate = 0, color = 'k')
    targ2 = target(name = 'Targ2', targetID=2, cords = np.array([45,15,200]), heading=0, speed=50, climbrate = 0, color = 'r')
    targ3 = target(name = 'Targ3', targetID=3, cords = np.array([0,30,100]), heading=0, speed=50, climbrate = 0, color = 'g')
    targ4 = target(name = 'Targ4', targetID=4, cords = np.array([-45,60,50]), heading=0, speed=50, climbrate = 0, color = 'c')
    targ5 = target(name = 'Targ5', targetID=5, cords = np.array([-90,90,0]), heading=0, speed=50, climbrate = 0, color = 'm')
    
    
    targs = [targ1, targ2, targ3, targ4, targ5]

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

if __name__ == "__main__":
    # Vector of time for simulation:
    time_vec = np.linspace(0, 240, 241) * u.minute

    env = testCase_environment()
    env.simulate(time_vec, savePlot = True, saveName = str(1), showSim = True)
    
    # Number of simulations:
    # numSims = 1
    # simData = defaultdict(dict)
    # for i in range(numSims):
    #     print(f'Simulation {i + 1} out of {numSims}')
    #     # Create a new environment instance for each simulation run:
    #     env = create_environment()
    #     # Simulate the satellites through the vector of time:
    #     simData[i] = env.simulate(time_vec, savePlot = True, saveName = str(i + 1), showSim = True)

        
    # Plot the NEES and NIS results:
    #plot_NEES_NIS(simData)
    
    env.render_gif("sim.gif", fps=5)
    
