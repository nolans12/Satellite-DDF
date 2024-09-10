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
    reds = plt.get_cmap('Reds', 7)

    # inverse the order so that 0 is most intense, 11 is least intense
    reds = reds.reversed()
    targ1_color = '#e71714'
    targ2_color = '#eea62a'
    targ3_color = '#58b428'
    targ4_color = '#2879b4'
    targ5_color = '#b228b4'

    # Define a random target:
    targ1 = target(
        name='Targ1',
        tqReq=1,
        targetID=1,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=80,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=reds(1),
    )
    targ2 = target(
        name='Targ2',
        tqReq=2,
        targetID=2,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=50,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=reds(2),
    )
    targ3 = target(
        name='Targ3',
        tqReq=3,
        targetID=3,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=40,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=reds(3),
    )
    targ4 = target(
        name='Targ4',
        tqReq=4,
        targetID=4,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=30,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=reds(4),
    )
    targ5 = target(
        name='Targ5',
        tqReq=5,
        targetID=5,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=20,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=reds(5),
    )

    targs = [targ1, targ2, targ3, targ4, targ5]

    # Define the satellite structure:

    sens_good = sensor(
        name='Sensor', fov=115, bearingsError=np.array([115 * 0.01, 115 * 0.01])
    )  # 1% error on FOV bearings
    sens_bad = sensor(
        name='Sensor', fov=115, bearingsError=np.array([115 * 0.1, 115 * 0.1])
    )  # 10% error on FOV bearings

    # Define the satellites:
    commandersIntent = NestedDict()

    sat1a = satellite(
        name='Sat1a',
        sensor=deepcopy(sens_good),
        a=Earth.R + 1000 * u.km,
        ecc=0,
        inc=60,
        raan=-45,
        argp=45,
        nu=0,
        color='#669900',
    )
    sat1b = satellite(
        name='Sat1b',
        sensor=deepcopy(sens_good),
        a=Earth.R + 1000 * u.km,
        ecc=0,
        inc=60,
        raan=-45,
        argp=30,
        nu=0,
        color='#66a3ff',
    )
    sat2a = satellite(
        name='Sat2a',
        sensor=deepcopy(sens_bad),
        a=Earth.R + 1000 * u.km,
        ecc=0,
        inc=120,
        raan=45,
        argp=45 + 7,
        nu=0,
        color='#9966ff',
    )
    sat2b = satellite(
        name='Sat2b',
        sensor=deepcopy(sens_bad),
        a=Earth.R + 1000 * u.km,
        ecc=0,
        inc=120,
        raan=45,
        argp=30 + 7,
        nu=0,
        color='#ffff33',
    )

    sats = [sat1a, sat1b, sat2a, sat2b]

    # For minute 0+: We want the following tracks:
    commandersIntent[0] = {
        sat1a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat1b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat2a: {2: 100, 3: 300},
        sat2b: {1: 100, 2: 220, 3: 240, 4: 300, 5: 280},
    }

    commandersIntent[4] = {
        sat1a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat1b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat2a: {2: 300, 3: 100},
        sat2b: {1: 300, 2: 220, 3: 240, 4: 100, 5: 280},
    }

    local = True

    central = False

    ci = True

    et = False

    # Define the communication network:
    comms_network = comms(
        sats,
        maxBandwidth=60,
        maxNeighbors=3,
        maxRange=10000 * u.km,
        minRange=500 * u.km,
        displayStruct=True,
    )

    # Create and return an environment instance:
    return environment(
        sats,
        targs,
        comms_network,
        commandersIntent,
        localEstimatorBool=local,
        centralEstimatorBool=central,
        ciEstimatorBool=ci,
        etEstimatorBool=et,
    )


### This environment is used for sampling mono tracks and other intresting edge cases, only 3 sats at 12000 km ####
def create_environment_mono():

    # Define a sensor model:
    sens = sensor(
        name='Sensor', fov=20, bearingsError=np.array([0.02, 0.02])
    )  # .1% error
    # sens = sensor(name = 'Sensor', fov = 20, bearingsError = np.array([0.2, 0.2])) # 1% error

    # Define targets for the satellites to track:
    targetIDs = [1]

    # Define local estimators:
    local = indeptEstimator(targetIDs=targetIDs)

    # Define the Data Fusion Algorithm, use the covariance intersection estimator:
    ci = ciEstimator(targetIDs=targetIDs)

    # Define the ET Fusion Algorithm
    et = etEstimator(targetIDs=targetIDs, targets=None, sat=None, neighbors=None)

    # Define the centralized estimator
    central = centralEstimator(targetIDs=targetIDs)

    # Define the colors for the sats:
    sat1_color = '#28B463'  # Green
    sat2_color = '#ef10d2'  # Purple
    sat3_color = '#3498DB'  # Blue

    red_shades = ['#EE4B2B', '#800020', '#DE3163']

    # MONO TRACK SATELLITE
    sat1 = satellite(
        name='Sat1',
        sensor=deepcopy(sens),
        targetIDs=targetIDs,
        indeptEstimator=deepcopy(local),
        ciEstimator=deepcopy(ci),
        etEstimator=deepcopy(et),
        a=Earth.R + 12000 * u.km,
        ecc=0,
        inc=0,
        raan=-45,
        argp=0,
        nu=0,
        color=sat1_color,
    )

    # POLAR ORBIT SATELLITE
    sat2 = satellite(
        name='Sat2',
        sensor=deepcopy(sens),
        targetIDs=targetIDs,
        indeptEstimator=deepcopy(local),
        ciEstimator=deepcopy(ci),
        etEstimator=deepcopy(et),
        a=Earth.R + 12000 * u.km,
        ecc=0,
        inc=90,
        raan=0,
        argp=0,
        nu=-45,
        color=sat2_color,
    )

    # INCLINATION 50 SATELLITE=90
    sat3 = satellite(
        name='Sat3',
        sensor=deepcopy(sens),
        targetIDs=targetIDs,
        indeptEstimator=deepcopy(local),
        ciEstimator=deepcopy(ci),
        etEstimator=deepcopy(et),
        a=Earth.R + 12000 * u.km,
        ecc=0,
        inc=90,
        raan=35,
        argp=0,
        nu=-80,
        color=sat3_color,
    )

    sats = [sat1, sat2, sat3]

    # Make the targets have some uncertainty in their initial state
    # At M = 4.7, hypersonic
    targ1 = target(
        name='Targ1',
        targetID=1,
        coords=np.array([0, -45, 0]),
        heading=90,
        speed=1.61538 * 60,
        uncertainty=np.array([0, 0, 0, 0, 0]),
        color=red_shades[0],
    )
    # # At M = 0.7, transonic speed
    # targ2 = target(name = 'Targ2', targetID=2, coords = np.array([100,-5,0]), heading=180, speed= 0.2401*60, uncertainty=np.array([0, 0, 0, 0, 0]), color = red_shades[1])
    # # At 50 mph
    # targ3 = target(name = 'Targ3', targetID=3, coords = np.array([45,0,0]), heading=180 + 45, speed= 0.022352*60, uncertainty=np.array([0.1, 0.1, 0, 0.1, 0.1]), color = red_shades[2])

    targs = [targ1]

    # Define the communication network:
    comms_network = comms(
        sats,
        maxNeighbors=3,
        maxRange=15000 * u.km,
        minRange=1 * u.km,
        displayStruct=True,
    )

    for sat in sats:
        neighbors = [neighbor for neighbor in sats if neighbor != sat]
        et = etEstimator(
            targets=targs, targetIDs=targetIDs, sat=sat, neighbors=neighbors
        )
        sat.populate_pairwise_et_estimator(et)

    # Create and return an environment instance:
    return environment(sats, targs, comms_network, central, ci, et)


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
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        fig.suptitle(
            f'Target {targ} NEES and NIS plots over {numSims} simulations', fontsize=16
        )
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
            nees_avg = np.array(
                [
                    np.mean(nees_net[targ][estimator][time])
                    for time in nees_net[targ][estimator].keys()
                ]
            )
            nis_avg = np.array(
                [
                    np.mean(nis_net[targ][estimator][time])
                    for time in nis_net[targ][estimator].keys()
                ]
            )
            # Plot the data
            axs[0, 0].scatter(
                list(nees_net[targ][estimator].keys()), nees_avg, label=f'{estimator}'
            )
            axs[1, 0].scatter(
                list(nees_net[targ][estimator].keys()), nees_avg, label=f'{estimator}'
            )
            axs[0, 1].scatter(
                list(nis_net[targ][estimator].keys()), nis_avg, label=f'{estimator}'
            )
            axs[1, 1].scatter(
                list(nis_net[targ][estimator].keys()), nis_avg, label=f'{estimator}'
            )

        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()

        # Save the plots, one for each target
        filePath = os.path.dirname(os.path.realpath(__file__))
        plotPath = os.path.join(filePath, 'plots')
        os.makedirs(plotPath, exist_ok=True)
        saveName = f'{fileName}_Target_{targ}_NEES_NIS'
        plt.savefig(os.path.join(plotPath, saveName), dpi=300)


### Main code to run the simulation
if __name__ == "__main__":

    # Vector of time for simulation:
    time_vec = np.linspace(0, 5, 5 * 12 + 1) * u.minute

    # Header name for the plots, gifs, and data
    fileName = "testing"

    # Create the environment
    env = create_environment()

    # Simulate the satellites through the vector of time:
    env.simulate(
        time_vec,
        saveName=fileName,
        show_env=True,
        plot_estimation_results=True,
        plot_communication_results=True,
    )

    # Save gifs:
    # env.render_gif(fileType='satellite_simulation', saveName=fileName, fps = 5)
    # env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 5)
    # env.render_gif(fileType='dynamic_comms', saveName=fileName, fps = 1)
