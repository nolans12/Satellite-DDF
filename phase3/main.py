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
def create_environment(cfg: sim_config.SimConfig) -> environment.Environment:

    # Define the targets for the satellites to track:
    # We will use the Reds color map for the targets
    reds = cm.get_cmap('Reds', 7)
    reds = (
        reds.reversed()
    )  # inverse the order so that 0 is most intense, 7 is least intense

    targs = [
        target.Target(
            name=name,
            tqReq=t.tq_req,
            targetID=t.target_id,
            coords=np.array(t.coords),
            heading=t.heading,
            speed=t.speed,
            uncertainty=np.array(t.uncertainty),
            color=t.color,
        )
        for name, t in cfg.targets.items()
    ]

    sats = {
        name: satellite.Satellite(
            name=name,
            sensor=sensor.Sensor(
                name=s.sensor,
                fov=cfg.sensors[s.sensor].fov,
                bearingsError=np.array(cfg.sensors[s.sensor].bearings_error),
            ),
            a=bodies.Earth.R + s.altitude * u.km,
            ecc=s.ecc,
            inc=s.inc,
            raan=s.raan,
            argp=s.argp,
            nu=s.nu,
            color=s.color,
        )
        for name, s in cfg.satellites.items()
    }

    # Define the goal of the system:
    commandersIntent: util.CommandersIndent = {
        time: {sat: intent for sat, intent in sat_intents.items()}
        for time, sat_intents in cfg.commanders_intent.items()
    }

    first_intent = next(iter(next(iter(commandersIntent.values())).values()))

    # commandersIntent[4] = {
    #     sat1a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat1b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat2a: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    #     sat2b: {1: 175, 2: 225, 3: 350, 4: 110, 5: 125},
    # }

    # Define the ground stations:
    groundStations = [
        groundStation.GroundStation(
            estimator=estimator.GsEstimator(first_intent),
            lat=gs.lat,
            lon=gs.lon,
            fov=gs.fov,
            commRange=gs.comms_range,
            name=name,
            color=gs.color,
        )
        for name, gs in cfg.ground_stations.items()
    ]

    # Define the communication network:
    comms_network = comms.Comms(
        list(sats.values()),
        maxBandwidth=cfg.comms.max_bandwidth,
        maxNeighbors=cfg.comms.max_neighbors,
        maxRange=cfg.comms.max_range * u.km,
        minRange=cfg.comms.min_range * u.km,
        displayStruct=cfg.comms.display_struct,
    )

    # Create and return an environment instance:
    return environment.Environment(
        list(sats.values()),
        targs,
        comms_network,
        groundStations,
        commandersIntent,
        cfg.estimators,
    )


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
    plot_groundstation_results: bool,
    gifs: list[sim_config.GifType],
) -> None:
    """Run the DDF simulation."""
    cfg = sim_config.load_sim_config(config)
    cfg.merge_overrides(
        sim_duration_m=time,
        output_prefix=output_prefix,
        show_env=show,
        plot_estimation=plot_estimation,
        plot_communication=plot_communication,
        plot_et_network=plot_et_network,
        plot_uncertainty_ellipses=plot_uncertainty_ellipses,
        plot_groundStation_results=plot_groundstation_results,
        gifs=gifs,
    )

    time_steps = cfg.sim_duration_m / cfg.sim_time_step_m

    # round to the nearest int
    time_steps = round(time_steps)

    # Vector of time for simulation:
    time_vec = (np.linspace(0, cfg.sim_duration_m, time_steps + 1)) * u.minute

    # Create the environment
    env = create_environment(cfg)

    # Simulate the satellites through the vector of time:
    env.simulate(
        time_vec,
        plot_config=cfg.plot,
    )

    # Save gifs:
    env.render_gifs(plot_config=cfg.plot, save_name=cfg.plot.output_prefix)


if __name__ == '__main__':
    main()
