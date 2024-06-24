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
    sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.15, 0.05]))
    #sens1 = sensor(name = 'Sensor 1', fov = 20, bearingsError = np.array([0.15, 0.05]))
    #sens2 = sensor(name = 'Sensor 2', fov = 20, bearingsError = np.array([0.15, 0.05]))
   
    # Define targets for the satellites to track:
    targetIDs = [1]#,2]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ddfEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs) 

    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = -35, argp = -55, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 0, raan = -35, argp = -55, nu = 0, color='c')
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 0, argp = -90, nu = 0, color='y')
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 35, argp = -125, nu = 0, color='r')
    # sat5 = satellite(name = 'Sat5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -60, nu = 0, color='g')
    # sat6 = satellite(name = 'Sat6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -90, nu = 0, color='m')

    sats = [sat1, sat2, sat3, sat4]#, sat3, sat4, sat5, sat6]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([0,-90,0]), heading=90, speed=95, climbrate = 0, color = 'k')
    #targ2 = target(name = 'Targ2', targetID=2, cords = np.array([0,-90,0]), heading=90, speed=100, climbrate = 0, color = 'k')
    
    targs = [targ1]#, targ2]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 50000*u.km, minRange = 1*u.km, displayStruct = True)

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
    
    numSims = len(simData)
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
    bearings_Error_matrix = np.array([ [0.1, 0.1], [0.15, 0.15], [0.2, 0.2], [0.25, 0.25], [0.3, 0.3] ])
    
    sensorA = sensor(name = 'Sensor A', fov = 115, bearingsError = bearings_Error_matrix[0])
    sensorB = sensor(name = 'Sensor B', fov = 115, bearingsError = bearings_Error_matrix[1])
    sensorC = sensor(name = 'Sensor C', fov = 115, bearingsError = bearings_Error_matrix[2])
    sensorD = sensor(name = 'Sensor D', fov = 115, bearingsError = bearings_Error_matrix[3])
    sensorE = sensor(name = 'Sensor E', fov = 115, bearingsError = bearings_Error_matrix[4])
    
    # Define TargetIDs
    targetIDs = [1,2,3,4,5]
    
    # Define the indepent Estimator
    local = indeptEstimator(targetIDs = targetIDs)
    
    # Define Data Fusion Algorithm
    ddf = ddfEstimator(targetIDs = targetIDs)
    
    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs)
        
    # Define the satellites:
    # Constellation 1: Polar Orbit at 0 degrees right ascension of ascending node
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sensorA), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sensorB), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 15, color='b')
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sensorC), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 30, color='b')
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sensorD), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 45, color='b')
    sat5 = satellite(name = 'Sat5', sensor = deepcopy(sensorE), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = 60, color='b')
    
    # Constellation 2: Polar Orbit at 30 degrees right ascension of ascending node
    sat6 = satellite(name = 'Sat6', sensor = deepcopy(sensorA), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 0, color='y')
    sat7 = satellite(name = 'Sat7', sensor = deepcopy(sensorB), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 15, color='y')
    sat8 = satellite(name = 'Sat8', sensor = deepcopy(sensorC), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 30, color='y')
    sat9 = satellite(name = 'Sat9', sensor = deepcopy(sensorD), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 45, color='y')
    sat10 = satellite(name = 'Sat10', sensor = deepcopy(sensorE), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 30, argp = 0, nu = 60, color='y')
    
    # Constellation 3: Polar Orbit at 60 degrees right ascension of ascending node
    # sat11 = satellite(name = 'Sat11', sensor = deepcopy(sensorA), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = 0, color='r')
    # sat12 = satellite(name = 'Sat12', sensor = deepcopy(sensorB), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = 5, color='r')
    # sat13 = satellite(name = 'Sat13', sensor = deepcopy(sensorC), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = 10, color='r')
    # sat14 = satellite(name = 'Sat14', sensor = deepcopy(sensorD), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = 15, color='r')
    # sat15 = satellite(name = 'Sat15', sensor = deepcopy(sensorE), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 60, argp = 0, nu = 20, color='r')
   
    # # Constellation 4: Polar Orbit at 90 degrees right ascension of ascending node
    # sat16 = satellite(name = 'Sat16', sensor = deepcopy(sensorA), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 0, color='g')
    # sat17 = satellite(name = 'Sat17', sensor = deepcopy(sensorB), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 5, color='g')
    # sat18 = satellite(name = 'Sat18', sensor = deepcopy(sensorC), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 10, color='g')
    # sat19 = satellite(name = 'Sat19', sensor = deepcopy(sensorD), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 15, color='g')
    # sat20 = satellite(name = 'Sat20', sensor = deepcopy(sensorE), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 90, argp = 0, nu = 20, color='g')
    
    sats = [sat1, sat2, sat3, sat4, sat5, sat6, sat7, sat8, sat9, sat10]#, sat11, sat12, sat13, sat14, sat15, sat16, sat17, sat18, sat19, sat20]
    
    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = '#DC143C')
    targ2 = target(name = 'Targ2', targetID=2, cords = np.array([45,30,200]), heading=0, speed=5, climbrate = 0, color = '#FF8C00')
    targ3 = target(name = 'Targ3', targetID=3, cords = np.array([0,0,100]), heading=90, speed=5, climbrate = 0, color = '#FFD700')
    targ4 = target(name = 'Targ4', targetID=4, cords = np.array([-45,-150,50]), heading=0, speed=5, climbrate = 0, color = '#228B22')
    targ5 = target(name = 'Targ5', targetID=5, cords = np.array([-90,0,0]), heading=0, speed=5, climbrate = 0, color = '#4B0082')
    
    
    targs = [targ1, targ2, targ3, targ4, targ5]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 5000*u.km, minRange = 500*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

if __name__ == "__main__":
    # Vector of time for simulation:
    time_vec = np.linspace(50, 180, 130*4 + 1) * u.minute
    
    env = create_environment()
    env.simulate(time_vec, savePlot = True, saveName = "new", showSim = False)
        
    # Plot the NEES and NIS results:
    # plot_NEES_NIS(simData)

    # Number of simulations:
    # numSims = 1
    # simData = defaultdict(dict)
    # for i in range(numSims):
    #     print(f'Simulation {i + 1} out of {numSims}')
    #     # Create a new environment instance for each simulation run:
    #     env = create_environment()
    #     # Simulate the satellites through the vector of time:
    #     simData[i] = env.simulate(time_vec, savePlot = True, saveName = "CI", showSim = False)

    # Save the gif:
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 15)