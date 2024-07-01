# Import pre-defined libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, indeptEstimator, ddfEstimator
from sensorClass import sensor
from commClass import comms

def create_environment():
    # Define a sensor model:
    sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([20 * 0.1, 20 * 0.1]))

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

    purple_shades = ['#9467BD']
    blue_shades = ['#87CEEB', '#6495ED', '#0000FF']
    yellow_shades = ['#FFBF00', '#E49B0F', '#FDDA0D']
    red_shades = ['#EE4B2B', '#800020', '#DE3163']

    # MONO TRACK SATELLITE
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 0, nu = 0, color=purple_shades[0])

    # POLAR ORBIT SATELLITES
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = -45, color=blue_shades[0])
    # sat2_2 = satellite(name = 'Sat2.2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 180, argp = 0, nu = 150 - 60, color=blue_shades[1])
    # sat2_3 = satellite(name = 'Sat2.3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 180, argp = 0, nu = 150 - 60*2, color=blue_shades[2])
   
    # INCLINATION 50 SATELLITES
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 50, raan = -135, argp = 0, nu = 90, color=yellow_shades[0])
    # sat3_2 = satellite(name = 'Sat3.2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 50, raan = -135, argp = 0, nu = 90 + 60, color=yellow_shades[1])
    # sat3_1 = satellite(name = 'Sat3.3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 50, raan = -135, argp = 0, nu = 90 + 60*2 - 180, color=yellow_shades[2])

    # sats = [sat1, sat2_1, sat2_2, sat2_3, sat3_1, sat3_2, sat3_3]
    sats = [sat1, sat2, sat3]

    # Define the target objects:
    # At M = 4.7, hypersonic
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([0,-45,0]), heading=90, speed= 1.61538*60, color = red_shades[0])
    # At M = 0.7, transonic speed
    targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, color = red_shades[1])
    # At 50 mph 
    targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, color = red_shades[2])
    
    targs = [targ1, targ2, targ3]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 15000*u.km, minRange = 1*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

def simple_environment():
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
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -25, nu = 0, color='c')
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -50, nu = 0, color='y')
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -30, nu = 0, color='r')
    sat5 = satellite(name = 'Sat5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -60, nu = 0, color='g')
    sat6 = satellite(name = 'Sat6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -90, nu = 0, color='m')

    sats = [sat1, sat2, sat3, sat4, sat5, sat6]

    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = 'k')
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
    time_vec = np.linspace(0, 250, 250*4 + 1) * u.minute
    
    # Number of simulations:
    numSims = 1
    simData = defaultdict(dict)
    for i in range(numSims):
        print(f'Simulation {i + 1} out of {numSims}')
        # Create a new environment instance for each simulation run:
        env = create_environment()
        # Simulate the satellites through the vector of time:
        simData[i] = env.simulate(time_vec, savePlot = True, saveName = "nolanCase", showSim = False)

    # Plot the NEES and NIS results:
    plot_NEES_NIS(simData)

    # Save the gif:
    env.render_gif(fileName = 'satellite_simulation.gif', fps = 5)