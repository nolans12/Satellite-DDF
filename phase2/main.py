# Import pre-defined libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, indeptEstimator, ciEstimator, etEstimator
from sensorClass import sensor
from commClass import comms

### This environment is used for the base case, with 12 satellites, all with different track qualitys being tracked by 4 satellites from 2 different constellations
def create_environment():

    # Define the targets for the satellites to track:
    # We will use the Reds color map for the targets
    reds = plt.get_cmap('Reds', 8)

    # inverse the order so that 0 is most intense, 11 is least intense
    reds = reds.reversed()
    targ1_color = '#e71714'
    targ2_color = '#eea62a'
    targ3_color = '#58b428'
    targ4_color = '#2879b4'
    targ5_color = '#b228b4'
    

    # Define a random target:
    targ1 = target(name = 'Targ1', tqReq = 1, targetID=1, coords = np.array([45,0,0]), heading=0, speed= 80,  uncertainty=np.array([5, 7.5, 0, 90, 0.1]), color = targ1_color)
    targ2 = target(name = 'Targ2', tqReq = 2, targetID=2, coords = np.array([45,0,0]), heading=0, speed= 50, uncertainty=np.array([5, 7.5, 0, 90, 0.1]), color =  targ2_color)
    targ3 = target(name = 'Targ3', tqReq = 3, targetID=3, coords = np.array([45,0,0]), heading=0, speed= 40,  uncertainty=np.array([5, 7.5, 0, 90, 0.1]), color = targ3_color)
    targ4 = target(name = 'Targ4', tqReq = 4, targetID=4, coords = np.array([45,0,0]), heading=0, speed= 30,  uncertainty=np.array([5, 7.5, 0, 90, 0.1]), color = targ4_color)
    targ5 = target(name = 'Targ5', tqReq = 5, targetID=5, coords = np.array([45,0,0]), heading=0, speed= 20,  uncertainty=np.array([5, 7.5, 0, 90, 0.1]), color = targ5_color)
    # targ6 = target(name = 'Targ6', tqReq = 6, targetID=6, coords = np.array([45,0,0]), heading=0, speed= 10,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(5))
    # targ7 = target(name = 'Targ7', tqReq = 7, targetID=7, coords = np.array([45,0,0]), heading=0, speed= 5,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(6))
    # targ8 = target(name = 'Targ8', tqReq = 8, targetID=8, coords = np.array([45,0,0]), heading=0, speed= 2,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(7))
    # targ9 = target(name = 'Targ9', tqReq = 9, targetID=9, coords = np.array([45,0,0]), heading=0, speed= 2,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(8))
    # targ10 = target(name = 'Targ10', tqReq = 10, targetID=10, coords = np.array([45,0,0]), heading=0, speed= 2,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(9))
    # targ11 = target(name = 'Targ11', tqReq = 11, targetID=11, coords = np.array([45,0,0]), heading=0, speed= 1,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(10))
    # targ12 = target(name = 'Targ12', tqReq = 12, targetID=12, coords = np.array([45,0,0]), heading=0, speed= 1,  uncertainty=np.array([5, 10, 0, 90, 0.1]), color = reds(11))
    
    targs = [targ1, targ2, targ3, targ4, targ5]

    # Define the satellite structure:

    sens1 = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.001, 115 * 0.001])) # 0.1% error on FOV bearings
    sens2 = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.01, 115 * 0.01])) # 1% error on FOV bearings
    sens3 = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.05, 115 * 0.05])) # 5% error on FOV bearings
    sens4 = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.1, 115 * 0.1])) # 10% error on FOV bearings

    targetIDs = [1,2,3,4,5,6,7,8,9,10,11,12]

    local = indeptEstimator(targetIDs = targetIDs)

    ci = ciEstimator(targetIDs = targetIDs)
    
    et = etEstimator(targetIDs = targetIDs, targets=None, sat=None, neighbors=None)

    central = centralEstimator(targetIDs = targetIDs)

    # Define the colors for the sats:
    # Make two of them green, two of them yellow
    green_shades = ['#8ff881', '#82f8e6']
    yellow_shades = ['#FDDA0D', '#FFA500']

    # Define the satellites:
    sat1a = satellite(name = 'Sat1a', sensor = deepcopy(sens2), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 45, nu = 0, color=green_shades[0])
    sat1b = satellite(name = 'Sat1b', sensor = deepcopy(sens2), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 30, nu = 0, color=green_shades[1])
    sat2a = satellite(name = 'Sat2a', sensor = deepcopy(sens4), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 45 + 7, nu = 0, color=yellow_shades[0])
    sat2b = satellite(name = 'Sat2b', sensor = deepcopy(sens4), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 1000 * u.km, ecc = 0, inc = 120, raan = 45, argp = 30 + 7, nu = 0, color=yellow_shades[1])

    sats = [sat1a, sat1b, sat2a, sat2b]

    # Define the communication network: 
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 10000*u.km, minRange = 500*u.km, displayStruct = True)
    
    for sat in sats:
        neighbors = [neighbor for neighbor in sats if neighbor != sat]
        et = etEstimator(targets = targs, targetIDs=targetIDs, sat=sat, neighbors=neighbors)
        sat.update_et_estimator(et)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central, ci, et)

### This environment is used for sampling mono tracks and other intresting edge cases, only 3 sats at 12000 km ####
def create_environment_mono():

    # Define a sensor model:
    sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.02, 0.02])) # .1% error
    #sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.2, 0.2])) # 1% error

    # Define targets for the satellites to track:
    targetIDs = [1]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ci = ciEstimator(targetIDs = targetIDs)
    
    # Define the ET Fusion Algorithm
    et = etEstimator(targetIDs = targetIDs, targets=None, sat=None, neighbors=None)
    
    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs) 

    # Define the colors for the sats:
    sat1_color = '#28B463'  # Green
    sat2_color = '#ef10d2'  # Purple
    sat3_color = '#3498DB'  # Blue
    
    red_shades = ['#EE4B2B', '#800020', '#DE3163']
    
    # MONO TRACK SATELLITE
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et),a = Earth.R + 12000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 0, nu = 0, color=sat1_color)

    # POLAR ORBIT SATELLITE
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = -45, color=sat2_color)

    # INCLINATION 50 SATELLITE=90
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ci), etEstimator=deepcopy(et), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 35, argp = 0, nu = -80, color=sat3_color)
    
    sats = [sat1, sat2, sat3]

    # Make the targets have some uncertainty in their initial state
    # At M = 4.7, hypersonic
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([0,-45,0]), heading=90, speed= 1.61538*60,  uncertainty=np.array([0, 0, 0, 0, 0]), color = red_shades[0])
    # # At M = 0.7, transonic speed
    # targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, uncertainty=np.array([0, 0, 0, 0, 0]), color = red_shades[1])
    # # At 50 mph
    # targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, uncertainty=np.array([0.1, 0.1, 0, 0.1, 0.1]), color = red_shades[2])

    targs = [targ1]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 15000*u.km, minRange = 1*u.km, displayStruct = True)
    
    for sat in sats:
        neighbors = [neighbor for neighbor in sats if neighbor != sat]
        et = etEstimator(targets = targs, targetIDs=targetIDs, sat=sat, neighbors=neighbors)
        sat.update_et_estimator(et)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central, ci, et)


### This environment is used for standard testing, 6 sats at 1000 km ####
def simple_environment():
   # Define a sensor model:
    sens = sensor(name = 'Sensor 1', fov = 115, bearingsError = np.array([0.05, 0.05]))
   
    # Define targets for the satellites to track:
    targetIDs = [1]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ciEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs) 

    # Define the satellites:
    sat1 = satellite(name = 'Sat1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = 0, nu = 0, color='b')
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -25, nu = 0, color='c')
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 90, raan = -45, argp = -50, nu = 0, color='y')
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 0, nu = 0, color='r')
    sat5 = satellite(name = 'Sat5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -60, nu = 0, color='g')
    sat6 = satellite(name = 'Sat6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ciEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -90, nu = 0, color='m')

    #sats = [sat1, sat2, sat3, sat4, sat5, sat6]
    sats = [sat1,sat4]
    
    # Define the target objects:
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([90,0,0]), heading=0, speed=5, climbrate = 0, color = 'k')
    targs = [targ1]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 5000*u.km, minRange = 500*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

### Plot the NEES and NIS results:
def plot_NEES_NIS(simData, fileName):

    # Now that the simulations are done, we can plot the results for NEES and NIS:
    def nested_dict():
        return defaultdict(list)
    
    # For each estimator, reset the nees and nis dictionaries
    nees_net = defaultdict(lambda: defaultdict(nested_dict))
    nis_net = defaultdict(lambda: defaultdict(nested_dict))

    numSims = len(simData)
    # Just loop trough everything and append the data to the dictionaries:
    # Loop through all sims
    # Define a satellite vector we can loop through, want to add
    for i in range(numSims):
        # Loop through all the targets
        for targ in simData[i].keys():
            # Loop through all of the different estimators we are using
            for estimator in simData[i][targ].keys():

                # if estimator == "Sat1" or estimator == "Sat2" or estimator == "Sat3":
                #     continue

                # For each estimator, store the NEES and NIS data for that in the nees_net and nis_net dictionaries
                for time in time_vec.to_value():
                    nees_data = simData[i][targ][estimator]['NEES'][time]
                    nis_data = simData[i][targ][estimator]['NIS'][time]
                    # If not empty, append to the dictionary
                    if nees_data and nis_data:
                        # Append to the data at that time:
                        if time not in nees_net[targ][estimator].keys():
                            nees_net[targ][estimator][time] = []
                        if time not in nis_net[targ][estimator].keys():
                            nis_net[targ][estimator][time] = []
                        nees_net[targ][estimator][time].append(nees_data)
                        nis_net[targ][estimator][time].append(nis_data)

    # Now, clean up the data
    # Loop through nees net and nis net and if any of the target entries are 0, remove that key
    for targ in nees_net.keys():
        if not nees_net[targ]:
            nees_net.pop(targ)
    for targ in nis_net.keys():
        if not nis_net[targ]:
            nis_net.pop(targ)

    # Now we can finally plot the NEES and NIS plots:
    # Goal is to make one plot for each target.
    # Each plot will have 2 subplots, one for NEES and one for NIS plots
    # The data on the plots will be averages for each estimator at each time step
    for targ in nees_net.keys():
        fig, axs = plt.subplots(2,2, figsize=(15, 8))
        fig.suptitle(f'Target {targ} NEES and NIS plots over {numSims} simulations', fontsize=16)
        axs[0, 0].set_title('Average NEES vs Time')
        axs[0, 0].set_xlabel('Time [min]')
        axs[0, 0].set_ylabel('NEES')
        axs[0, 0].set_ylim([0, 12])

        axs[0, 1].set_title('Average NIS vs Time')
        axs[0, 1].set_xlabel('Time [min]')
        axs[0, 1].set_ylabel('NIS')
        axs[0, 1].set_ylim([0, 6])

        # axs[1, 0].set_title('Average NEES vs Time')
        axs[1, 0].set_xlabel('Time [min]')
        axs[1, 0].set_ylabel('NEES')

        # axs[1, 1].set_title('Average NIS vs Time')
        axs[1, 1].set_xlabel('Time [min]')
        axs[1, 1].set_ylabel('NIS')

        for estimator in nees_net[targ].keys():
            # Calculate the average NEES and NIS values for each time step
            nees_avg = np.array([np.mean(nees_net[targ][estimator][time]) for time in nees_net[targ][estimator].keys()])
            nis_avg = np.array([np.mean(nis_net[targ][estimator][time]) for time in nis_net[targ][estimator].keys()])
            # Plot the data
            axs[0, 0].scatter(list(nees_net[targ][estimator].keys()), nees_avg, label=f'{estimator}')
            axs[1, 0].scatter(list(nees_net[targ][estimator].keys()), nees_avg, label=f'{estimator}')
            axs[0, 1].scatter(list(nis_net[targ][estimator].keys()), nis_avg, label=f'{estimator}')
            axs[1, 1].scatter(list(nis_net[targ][estimator].keys()), nis_avg, label=f'{estimator}')
 
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()

        # Save the plots, one for each target
        filePath = os.path.dirname(os.path.realpath(__file__))
        plotPath = os.path.join(filePath, 'plots')
        os.makedirs(plotPath, exist_ok=True)
        saveName = f'{fileName}_Target_{targ}_NEES_NIS'
        plt.savefig(os.path.join(plotPath,saveName), dpi=300)

### Main code to run the simulation
if __name__ == "__main__":

    # Vector of time for simulation:
    time_vec = np.linspace(0, 10, 20 + 1) * u.minute

    # Header name for the plots, gifs, and data
    fileName = "final_threshold_again_haha"

    env = create_environment()
    # Simulate the satellites through the vector of time:
    env.simulate(time_vec, showSim = False, savePlot = False, saveGif= False, saveData = False, saveComms = True, plot_dynamic_comms = False, saveName = fileName)

    # Save the gif:
    env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 5)
    #env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 5)
    #env.render_gif(fileType='dynamic_comms', saveName=fileName, fps = 1)

    # ### Do formal NEES and NIS test:
    # time_vec = np.linspace(36, 51, 15 + 1) * u.minute
    # fileName = "example2"
    # numSims = 1
    # simData = defaultdict(dict)
    # for i in range(numSims):
    #     print(f'Simulation {i + 1} out of {numSims}')
    #     # Create a new environment instance for each simulation run:
    #     env = create_environment_mono()
    #     # Simulate the satellites through the vector of time:
    #     simData[i] = env.simulate(time_vec, pause_step=0.1, savePlot=True, saveGif=True, saveData=True, saveName=fileName, showSim=False)
        
    #     env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 1)
    #     env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 1)

    # plot_NEES_NIS(simData, fileName)

