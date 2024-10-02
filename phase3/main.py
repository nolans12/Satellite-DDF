# Import pre-defined libraries
import copy

import numpy as np
from astropy import units as u
from matplotlib import cm
from poliastro import bodies

# Import classes
from phase3 import comm
from phase3 import environment
from phase3 import satellite
from phase3 import sensor
from phase3 import target
from phase3 import util


### This environment is used for the base case, with 12 satellites, all with different track qualitys being tracked by 4 satellites from 2 different constellations
def create_environment():

    # Define the targets for the satellites to track:
    # We will use the Reds color map for the targets
    reds = cm.get_cmap('Reds', 7)
    reds = (
        reds.reversed()
    )  # inverse the order so that 0 is most intense, 7 is least intense

    targ1_color = '#e71714'
    targ2_color = '#eea62a'
    targ3_color = '#58b428'
    targ4_color = '#2879b4'
    targ5_color = '#b228b4'

    # Define the targets
    targ1 = target.Target(
        name='Targ1',
        tqReq=1,
        targetID=1,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=80,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=targ1_color,
    )
    targ2 = target.Target(
        name='Targ2',
        tqReq=2,
        targetID=2,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=50,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=targ2_color,
    )
    targ3 = target.Target(
        name='Targ3',
        tqReq=3,
        targetID=3,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=40,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=targ3_color,
    )
    targ4 = target.Target(
        name='Targ4',
        tqReq=4,
        targetID=4,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=30,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=targ4_color,
    )
    targ5 = target.Target(
        name='Targ5',
        tqReq=5,
        targetID=5,
        coords=np.array([45, 0, 0]),
        heading=0,
        speed=20,
        uncertainty=np.array([3, 7.5, 0, 90, 0.1]),
        color=targ5_color,
    )

    targs = [targ1, targ2, targ3, targ4, targ5]

    # Define the satellite structure:
    sens_good = sensor.Sensor(
        name='Sensor', fov=115, bearingsError=np.array([115 * 0.01, 115 * 0.01])
    )  # 1% error on FOV bearings
    sens_bad = sensor.Sensor(
        name='Sensor', fov=115, bearingsError=np.array([115 * 0.1, 115 * 0.1])
    )  # 10% error on FOV bearings

    sat1a = satellite.Satellite(
        name='Sat1a',
        sensor=copy.deepcopy(sens_good),
        a=bodies.Earth.R + 1000 * u.km,
        ecc=0,
        inc=60,
        raan=-45,
        argp=45,
        nu=0,
        color='#669900',
    )
    sat1b = satellite.Satellite(
        name='Sat1b',
        sensor=copy.deepcopy(sens_good),
        a=bodies.Earth.R + 1000 * u.km,
        ecc=0,
        inc=60,
        raan=-45,
        argp=30,
        nu=0,
        color='#66a3ff',
    )
    sat2a = satellite.Satellite(
        name='Sat2a',
        sensor=copy.deepcopy(sens_bad),
        a=bodies.Earth.R + 1000 * u.km,
        ecc=0,
        inc=120,
        raan=45,
        argp=45 + 7,
        nu=0,
        color='#9966ff',
    )
    sat2b = satellite.Satellite(
        name='Sat2b',
        sensor=copy.deepcopy(sens_bad),
        a=bodies.Earth.R + 1000 * u.km,
        ecc=0,
        inc=120,
        raan=45,
        argp=30 + 7,
        nu=0,
        color='#ffff33',
    )

    sats = [sat1a, sat1b, sat2a, sat2b]

    # Define the goal of the system:
    commandersIntent: util.CommandersIndent = {}

    # For minute 0+: We want the following tracks:
    commandersIntent[0] = {
        sat1a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat1b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat2a: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
        sat2b: {1: 100, 2: 150, 3: 200, 4: 250, 5: 300},
    }

    commandersIntent[4] = {
        sat1a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        sat1b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        sat2a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
        sat2b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    }

    # Define the estimators used:
    central = True
    local = True
    ci = True
    et = False

    # Define the communication network:
    comms_network = comm.Comms(
        sats,
        maxBandwidth=60,
        maxNeighbors=3,
        maxRange=10000 * u.km,
        minRange=500 * u.km,
        displayStruct=True,
    )

    # Create and return an environment instance:
    return environment.Environment(
        sats,
        targs,
        comms_network,
        commandersIntent,
        localEstimatorBool=local,
        centralEstimatorBool=central,
        ciEstimatorBool=ci,
        etEstimatorBool=et,
    )


### Main code to run the simulation
if __name__ == "__main__":

    # Vector of time for simulation:
    time_vec = np.linspace(0, 10, 10 * 6 + 1) * u.minute

    # Header name for the plots, gifs, and data
    fileName = "test"

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
    env.render_gif(fileType='satellite_simulation', saveName=fileName, fps=5)
    # env.render_gif(fileType='uncertainty_ellipse', saveName=fileName, fps = 5)
    # env.render_gif(fileType='dynamic_comms', saveName=fileName, fps = 1)
