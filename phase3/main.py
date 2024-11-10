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
    '--plots',
    help='Plots to generate',
    multiple=True,
    type=click_utils.EnumChoice(sim_config.PlotType),
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
    plots: list[sim_config.PlotType],
    gifs: list[sim_config.GifType],
) -> None:
    """Run the DDF simulation."""
    cfg = sim_config.load_sim_config(config)
    cfg.merge_overrides(
        sim_duration_m=time,
        output_prefix=output_prefix,
        show_live=show,
        plots=plots,
        gifs=gifs,
    )
    # Create the environment
    env = environment.Environment.from_config(cfg)

    env.simulate()

    env.post_process()


if __name__ == '__main__':

    scenario = sim_config.load_sim_config(path_utils.SCENARIOS / 'new.yaml')

    env = environment.Environment.from_config(scenario)

    env.simulate()

    env.post_process()
