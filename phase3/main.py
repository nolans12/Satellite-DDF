# Import pre-defined libraries
import copy
import pathlib
from typing import cast

import numpy as np
import rich_click as click
import yaml
from astropy import units as u
from matplotlib import cm
from poliastro import bodies

from common import click_utils
from common import path_utils
from phase3 import comms
from phase3 import environment
from phase3 import estimator
from phase3 import groundStation
from phase3 import satellite
from phase3 import sensor
from phase3 import sim_config
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

    # commandersIntent[4] = {
    #     sat1a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat1b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat2a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat2b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    # }

    # Define the ground stations:
    gs1 = groundStation.GroundStation(
        estimator=estimator.GsEstimator(commandersIntent[0][sat1a]),
        lat=60,
        lon=10,
        fov=80,
        commRange=10000,
        name='GS1',
        color='black',
    )

    groundStations = [gs1]

    # Define the communication network:
    comms_network = comms.Comms(
        sats,
        maxBandwidth=60,
        maxNeighbors=3,
        maxRange=10000 * u.km,
        minRange=500 * u.km,
        displayStruct=True,
    )

    # Define the estimators used:
    central = False
    local = True
    ci = False
    et = True

    # Create and return an environment instance:
    return environment.Environment(
        sats,
        targs,
        comms_network,
        groundStations,
        commandersIntent,
        localEstimatorBool=local,
        centralEstimatorBool=central,
        ciEstimatorBool=ci,
        etEstimatorBool=et,
    )


def _load_sim_config(file: pathlib.Path) -> sim_config.SimConfig:
    schema = sim_config.SimConfigSchema()
    return cast(sim_config.SimConfig, schema.load(yaml.safe_load(file.read_text())))


@click.command()
@click.option(
    '--config',
    default=path_utils.SCENARIOS / 'default.yaml',
    help='Simulation configuration file',
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--output-prefix',
    help='Prefix of the output files',
)
@click.option(
    '--time',
    help='Duration of the simulation in minutes',
    type=int,
)
@click.option(
    '--show',
    help='Show the environment plot',
    type=bool,
)
@click.option(
    '--plot-estimation',
    help='Plot the estimation results',
    type=bool,
)
@click.option(
    '--plot-communication',
    help='Plot the communication results',
    type=bool,
)
@click.option(
    '--plot-et-network',
    help='Plot the ET network',
    type=bool,
)
@click.option(
    '--plot-uncertainty-ellipses',
    help='Plot the uncertainty ellipses',
    type=bool,
)
@click.option(
    '--plot-groundStation-results',
    help='Plot the ground station results',
    type=bool,
)
@click.option(
    '--gifs',
    help='Gifs to generate',
    multiple=True,
    type=click_utils.EnumChoice(sim_config.GifType),
)
def main(
    output_prefix: str,
    config: pathlib.Path,
    time: int,
    show: bool,
    plot_estimation: bool,
    plot_communication: bool,
    plot_et_network: bool,
    plot_uncertainty_ellipses: bool,
    plot_groundStation_results: bool,
    gifs: list[sim_config.GifType],
) -> None:
    cfg = _load_sim_config(config)
    cfg.merge_overrides(
        sim_duration_m=time,
        output_prefix=output_prefix,
        show_env=show,
        plot_estimation=plot_estimation,
        plot_communication=plot_communication,
        plot_et_network=plot_et_network,
        plot_uncertainty_ellipses=plot_uncertainty_ellipses,
        plot_groundStation_results=plot_groundStation_results,
        gifs=gifs,
    )

    # Vector of time for simulation:
    time_vec = np.linspace(0, cfg.sim_duration_m, cfg.sim_duration_m * 6 + 1) * u.minute

    # Create the environment
    env = create_environment()

    # Simulate the satellites through the vector of time:
    env.simulate(
        time_vec,
        plot_config=cfg.plot,
    )

    # Save gifs:
    env.render_gifs(plot_config=cfg.plot, save_name=cfg.plot.output_prefix)


if __name__ == '__main__':
    main()
