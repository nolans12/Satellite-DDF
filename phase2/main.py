# Import pre-defined libraries
from import_libraries import *

# Import classes
from satelliteClass import satellite
from targetClass import target
from environmentClass import environment
from estimatorClass import centralEstimator, indeptEstimator, ddfEstimator
from sensorClass import sensor
from commClass import comms

### This environment is used for sampling mono tracks and other intresting edge cases, only 3 sats at 12000 km ####
def create_environment_edge():

    # Define a sensor model:
    # sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.002, 0.002])) # 1/100% error
    sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.2, 0.2])) # 1% error

    # Define targets for the satellites to track:
    targetIDs = [1,2,3]

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

    # POLAR ORBIT SATELLITE
    sat2 = satellite(name = 'Sat2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 90, raan = 0, argp = 0, nu = -45, color=blue_shades[0])

    # INCLINATION 50 SATELLITE
    sat3 = satellite(name = 'Sat3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 12000 * u.km, ecc = 0, inc = 50, raan = -135, argp = 0, nu = 90, color=yellow_shades[0])
    
    sats = [sat1, sat2, sat3]

    # # Define the target objects:
    # # At M = 4.7, hypersonic
    # targ1 = target(name = 'Targ1', targetID=1, coords = np.array([0,-45,0]), heading=90, speed= 1.61538*60, color = red_shades[0])
    # # At M = 0.7, transonic speed
    # targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, color = red_shades[1])
    # # At 50 mph 
    # targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, color = red_shades[2])
    
    # Make the targets have some uncertainty in their initial state
    # At M = 4.7, hypersonic
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([0,-45,0]), heading=90, speed= 1.61538*60,  uncertainty=np.array([0.1, 0.1, 0, 0.1, 0.1]), color = red_shades[0])
    # At M = 0.7, transonic speed
    targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, uncertainty=np.array([0.1, 0.1, 0, 0.1, 0.1]), color = red_shades[1])
    # At 50 mph
    targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, uncertainty=np.array([0.1, 0.1, 0, 0.1, 0.1]), color = red_shades[2])

    targs = [targ1, targ2, targ3]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 15000*u.km, minRange = 1*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

#### This environment is used for a standard LEO constellation sim, 20 satellites at 1000 km ####
def create_environment_standard():

    # Define the sensor model:
    sens = sensor(name = 'Sensor', fov = 115, bearingsError = np.array([115 * 0.1, 115 * 0.1]))

    # Define targets for the satellites to track:
    targetIDs = [1,2,3]

    # Define local estimators:
    local = indeptEstimator(targetIDs = targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ddf = ddfEstimator(targetIDs = targetIDs)

    # Define the centralized estimator
    central = centralEstimator(targetIDs = targetIDs)

    # Define the colors for the targets:
    red_shades = ['#EE4B2B', '#800020', '#DE3163']

    # For the equatorial orbits, use a bunch of purples:
    purple_shades = ['#9467BD', '#8A2BE2', '#800080', '#4B0082', '#483D8B', '#0000FF', '#0000CD', '#00008B', '#000080', '#191970']
    yellow_shades = ['#FFBF00', '#E49B0F', '#FDDA0D', '#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500', '#FF0000']
    green_shades = ['#00FF00', '#32CD32', '#008000', '#006400', '#9ACD32', '#6B8E23', '#556B2F', '#808000', '#556B2F', '#2E8B57']

    # Constellation of 10 satellites at RAAN = -20, i = 60
    sat1_1 = satellite(name = 'Sat1.1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*0, nu = 0, color=purple_shades[0])
    sat1_2 = satellite(name = 'Sat1.2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*1, nu = 0, color=purple_shades[1])
    sat1_3 = satellite(name = 'Sat1.3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*2, nu = 0, color=purple_shades[2])
    sat1_4 = satellite(name = 'Sat1.4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*3, nu = 0, color=purple_shades[3])
    sat1_5 = satellite(name = 'Sat1.5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*4, nu = 0, color=purple_shades[4])
    sat1_6 = satellite(name = 'Sat1.6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*5, nu = 0, color=purple_shades[5])
    sat1_7 = satellite(name = 'Sat1.7', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*6, nu = 0, color=purple_shades[6])
    sat1_8 = satellite(name = 'Sat1.8', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*7, nu = 0, color=purple_shades[7])
    sat1_9 = satellite(name = 'Sat1.9', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*8, nu = 0, color=purple_shades[8])
    sat1_10 = satellite(name = 'Sat1.10', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 60, raan = -45, argp = 36*9, nu = 0, color=purple_shades[9])

    # Constellation of 10 satellites at RAAN = -20 + 180, i = 60
    sat2_1 = satellite(name = 'Sat2.1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*0, nu = 0, color=yellow_shades[0])
    sat2_2 = satellite(name = 'Sat2.2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*1, nu = 0, color=yellow_shades[1])
    sat2_3 = satellite(name = 'Sat2.3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*2, nu = 0, color=yellow_shades[2])
    sat2_4 = satellite(name = 'Sat2.4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*3, nu = 0, color=yellow_shades[3])
    sat2_5 = satellite(name = 'Sat2.5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*4, nu = 0, color=yellow_shades[4])
    sat2_6 = satellite(name = 'Sat2.6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*5, nu = 0, color=yellow_shades[5])
    sat2_7 = satellite(name = 'Sat2.7', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*6, nu = 0, color=yellow_shades[6])
    sat2_8 = satellite(name = 'Sat2.8', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*7, nu = 0, color=yellow_shades[7])
    sat2_9 = satellite(name = 'Sat2.9', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*8, nu = 0, color=yellow_shades[8])
    sat2_10 = satellite(name = 'Sat2.10', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf), a = Earth.R + 1000 * u.km, ecc = 0, inc = 180 - 60, raan = 45, argp = 36*9, nu = 0, color=yellow_shades[9])

    # Constellation of 10 sats doing equotorial orbits
    sat3_1 = satellite(name = 'Sat3.1', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*0, nu = 0, color=green_shades[0])
    sat3_2 = satellite(name = 'Sat3.2', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*1, nu = 0, color=green_shades[1])
    sat3_3 = satellite(name = 'Sat3.3', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*2, nu = 0, color=green_shades[2])
    sat3_4 = satellite(name = 'Sat3.4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*3, nu = 0, color=green_shades[3])
    sat3_5 = satellite(name = 'Sat3.5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*4, nu = 0, color=green_shades[4])
    sat3_6 = satellite(name = 'Sat3.6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*5, nu = 0, color=green_shades[5])
    sat3_7 = satellite(name = 'Sat3.7', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*6, nu = 0, color=green_shades[6])
    sat3_8 = satellite(name = 'Sat3.8', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*7, nu = 0, color=green_shades[7])
    sat3_9 = satellite(name = 'Sat3.9', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*8, nu = 0, color=green_shades[8])
    sat3_10 = satellite(name = 'Sat3.10', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 36*9, nu = 0, color=green_shades[9])

    sats = [sat1_1, sat1_2, sat1_3, sat1_4, sat1_5, sat1_6, sat1_7, sat1_8, sat1_9, sat1_10, sat2_1, sat2_2, sat2_3, sat2_4, sat2_5, sat2_6, sat2_7, sat2_8, sat2_9, sat2_10, sat3_1, sat3_2, sat3_3, sat3_4, sat3_5, sat3_6, sat3_7, sat3_8, sat3_9, sat3_10]

    # Define the targets:
    # At M = 4.7, hypersonic
    targ1 = target(name = 'Targ1', targetID=1, coords = np.array([0,-45,0]), heading=90, speed= 1.61538*60, color = red_shades[0])
    # At M = 0.7, transonic speed
    targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, color = red_shades[1])
    # At 50 mph 
    targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, color = red_shades[2])
    
    targs = [targ1, targ2, targ3]

    # Define the communication network:
    comms_network = comms(sats, maxNeighbors = 3, maxRange = 7500*u.km, minRange = 5*u.km, displayStruct = True)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central)

### This environment is used for standard testing, 6 sats at 1000 km ####
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
    sat4 = satellite(name = 'Sat4', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = 0, nu = 0, color='r')
    sat5 = satellite(name = 'Sat5', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -60, nu = 0, color='g')
    sat6 = satellite(name = 'Sat6', sensor = deepcopy(sens), targetIDs=targetIDs, indeptEstimator=deepcopy(local), ddfEstimator=deepcopy(ddf),  a = Earth.R + 1000 * u.km, ecc = 0, inc = 0, raan = -45, argp = -90, nu = 0, color='m')

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

    # # Vector of time for simulation:
    # time_vec = np.linspace(0, 40, 40 + 1) * u.minute

    # # Header name for the plots, gifs, and data
    # fileName = "example"

    # env = create_environment_edge()
    # # Simulate the satellites through the vector of time:
    # env.simulate(time_vec, savePlot = True, showSim = True, saveName = fileName)

    # # Save the gif:
    # env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 5)
    # env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 5)

    ### Do formal NEES and NIS test:
    time_vec = np.linspace(40, 80, 40*1 + 1) * u.minute
    fileName = "testingPlots" 
    numSims = 1
    simData = defaultdict(dict)
    for i in range(numSims):
        print(f'Simulation {i + 1} out of {numSims}')
        # Create a new environment instance for each simulation run:
        env = create_environment_edge()
        # Simulate the satellites through the vector of time:
        simData[i] = env.simulate(time_vec, savePlot = True, saveName = fileName)
        env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 5)
        # simData[i] = env.simulate(time_vec)

    plot_NEES_NIS(simData, fileName)

