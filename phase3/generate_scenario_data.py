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
def main(
    config: pathlib.Path,
) -> None:
    """Run the DDF simulation."""
    cfg = sim_config.load_sim_config(config)
    # Create the environment
    env = environment.Environment.from_config(cfg)

    env.simulate_for_db()


if __name__ == '__main__':
    main()
