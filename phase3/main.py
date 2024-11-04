import pathlib

import numpy as np
import rich_click as click
from astropy import units as u

from common import click_utils
from common import path_utils
from phase3 import environment
from phase3 import sim_config


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
    time_vec = u.Quantity(np.linspace(0, cfg.sim_duration_m, time_steps + 1), u.minute)

    # Create the environment
    env = environment.Environment.from_config(cfg)

    # Simulate the satellites through the vector of time:
    env.simulate(
        time_vec,
        plot_config=cfg.plot,
    )

    # Save gifs:
    # env.render_gifs(plot_config=cfg.plot, save_name=cfg.plot.output_prefix)


if __name__ == '__main__':
    # main()

    scenario = sim_config.load_sim_config(path_utils.SCENARIOS / 'wd_45_30_6_1.yaml')

    # # Create the environment
    env = environment.Environment.from_config(scenario)
