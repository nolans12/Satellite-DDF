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
    sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.25, 0.1]))
    #sens1 = sensor(name = 'Sensor 1', fov = 20, bearingsError = np.array([0.15, 0.05]))
    #sens2 = sensor(name = 'Sensor 2', fov = 20, bearingsError = np.array([0.15, 0.05]))
   
    # Define targets for the satellites to track:
    targetIDs = [1,2,3]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ddfEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ddfEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs) 

    # Colors: Purple, Pink, Light Purple, Light Pink, orange || Blue, dark cyan, 
    hex_colors = ['#9467BD','#E377C2', '#D7BDE2', '#F8BBD0', '#FF7F0E', '#1F77B4', '#008B8B', '#1B4F72']
    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 0, raan = -35, argp = -55, nu = 0, color=hex_colors[0])
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = -35, argp = -55, nu = 0, color=hex_colors[1])
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 0, argp = -90, nu = 0, color=hex_colors[2])
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 35, argp = -125, nu = 0, color=hex_colors[3])
    sat5 = satellite(name = 'Sat5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 12000 * u.km, ecc = 0, inc = 45, raan = -35, argp = -55, nu = 0, color=hex_colors[4])
    # sat6 = satellite(name = 'Sat6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -90, nu = 0, color='m')

    sats = [sat1, sat2, sat3, sat4, sat5] # , sat3, sat4]#, sat3, sat4, sat5, sat6]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, cords = np.array([0,-90,0]), heading=90, speed=95, climbrate = 0, color = hex_colors[5])
    targ2 = target(name = 'Targ2', targetID=2, cords = np.array([0,0,0]), heading=0, speed=0, climbrate = 0, color = hex_colors[6])
    targ3 = target(name = 'Targ3', targetID=3, cords = np.array([0,-25,0]), heading=0, speed=35, climbrate = 0, color = hex_colors[7])
    
    
    targs = [targ1, targ2, targ3]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 15000*u.km, minRange = 1*u.km, displayStruct = True)

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


if __name__ == "__main__":
    # Vector of time for simulation:
    time_vec = np.linspace(0, 250, 250*6 + 1) * u.minute
    
    env = create_environment()
    savePlotName = 'dt_10s_'
    env.simulate(time_vec, savePlot = True, saveName = savePlotName, showSim = False)
        
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
    env.render_gif(fileName = savePlotName + 'satellite_simulation.gif', fps = 30)