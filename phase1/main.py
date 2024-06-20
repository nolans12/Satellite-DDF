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
    sat1 = satellite(name = 'Sat1', sensor = sens1, targetIDs=targetIDs, estimator = local1, dataFusion=dataFusionAlg, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 80, nu = 0, color='b')
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
    time_vec = np.linspace(0, 10, 11) * u.minute

    # Number of simulations:
    numSims = 1
    simData = defaultdict(dict)
    for i in range(numSims):
        print(f'Simulation {i + 1} out of {numSims}')
        # Create a new environment instance for each simulation run:
        env = create_environment()
        # Simulate the satellites through the vector of time:
        simData[i] = env.simulate(time_vec, savePlot = True, saveName = str(i + 1), showSim = True)

        
    # Plot the NEES and NIS results:
    #plot_NEES_NIS(simData)
    

# ### CLUSTER SIM BELOW:

# # DEFINE THE SATELLITE OBJECTS:

#     # Define a sensor model:
#     sens1_1 = sensor(name = 'Sensor 1.1', fov = 115, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens1_2 = sensor(name = 'Sensor 1.2', fov = 115, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens1_3 = sensor(name = 'Sensor 1.3', fov = 115, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens1_4 = sensor(name = 'Sensor 1.4', fov = 115, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens2_1 = sensor(name = 'Sensor 2.1', fov = 100, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens2_2 = sensor(name = 'Sensor 2.2', fov = 100, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens2_3 = sensor(name = 'Sensor 2.3', fov = 100, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
#     sens2_4 = sensor(name = 'Sensor 2.4', fov = 100, bearingsError = np.array([0.5, 0.5]), rangeError = 0.5, detectChance= 0.05, resolution = 720)
    
#     # Define targets for the satellites to track:
#     # TODO: should we just make the satellite track any target it can see?
#     targetIDs = [1]


#     # Define local estimators:
#     local1_1 = localEstimator(targetIDs = targetIDs)
#     local1_2 = localEstimator(targetIDs = targetIDs)
#     local1_3 = localEstimator(targetIDs = targetIDs)
#     local1_4 = localEstimator(targetIDs = targetIDs)
    
#     local2_1 = localEstimator(targetIDs = targetIDs)
#     local2_2 = localEstimator(targetIDs = targetIDs)
#     local2_3 = localEstimator(targetIDs = targetIDs)
#     local2_4 = localEstimator(targetIDs = targetIDs)

#     # Central Estimator:
#     central = centralEstimator(targetIDs = targetIDs) 
    
#     ## Cluster 1: Polar orbits:
#     sat1_1 = satellite(name = 'Sat1', sensor = sens1_1, targetIDs=targetIDs, estimator = local1_1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, color='b')
#     sat1_2 = satellite(name = 'Sat2', sensor = sens1_2, targetIDs=targetIDs, estimator = local1_2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 20, nu = 0, color='b')
#     sat1_3 = satellite(name = 'Sat3', sensor = sens1_3, targetIDs=targetIDs, estimator = local1_3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 40, nu = 0, color='b')
#     sat1_4 = satellite(name = 'Sat4', sensor = sens1_4, targetIDs=targetIDs, estimator = local1_4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 60, nu = 0, color='b')

#     ## Cluster 2: Equatorial orbits:
#     sat2_1 = satellite(name = 'Sat5', sensor = sens2_1, targetIDs=targetIDs, estimator = local2_1, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -180, nu = 0, color='r')
#     sat2_2 = satellite(name = 'Sat6', sensor = sens2_2, targetIDs=targetIDs, estimator = local2_2, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -160, nu = 0, color='r')
#     sat2_3 = satellite(name = 'Sat7', sensor = sens2_3, targetIDs=targetIDs, estimator = local2_3, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -140, nu = 0, color='r')
#     sat2_4 = satellite(name = 'Sat8', sensor = sens2_4, targetIDs=targetIDs, estimator = local2_4, a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = 0, argp = -120, nu = 0, color='r')
    
#     sats = [sat1_1, sat1_2, sat1_3, sat1_4, sat2_1, sat2_2, sat2_3, sat2_4]

# # DEFINE THE TARGET OBJECTS:
#     targ1 = target(name = 'Targ1', targetID=1, r = np.array([6378, 0, 0, 0, 0, 0]),color = 'k')
     
#     targs = [targ1]

# # Define the communication network:
#     comms = comms(sats, range = 5000 * u.km, displayStruct = True)

# # Create an environment instance:
#     env = environment(sats, targs, comms, central)

# # Simulate the satellites through a vector of time:
#     time_vec = np.linspace(0, 50, 51) * u.minute
#     env.simulate(time_vec, display = True)

# # Save the gif:
#     env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)
