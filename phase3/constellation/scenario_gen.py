import dataclasses
import enum
import pathlib

import rich_click as click
from InquirerPy.base import control
from InquirerPy.prompts import filepath
from InquirerPy.prompts import list as list_prompt
from InquirerPy.prompts import number

from common import path_utils
from common import prompt_utils
from phase3 import sim_config
from phase3.constellation import walker_delta


class Constellelation(enum.Enum):
    WALKER_DELTA = 'walker_delta'


def _prompt_for_constellation() -> Constellelation:
    return list_prompt.ListPrompt(
        message='Select a constellation type',
        choices=[control.Choice(const, const.name) for const in Constellelation],
        default=None,
    ).execute()


def _prompt_for_walker_delta() -> tuple[int, int, float, float, float]:
    total_sats = number.NumberPrompt(
        message='Enter the total number of satellites', default=24, min_allowed=1
    ).execute()

    num_planes = number.NumberPrompt(
        message='Enter the number of planes',
        default=6,
        min_allowed=1,
    ).execute()

    inc_deg = number.NumberPrompt(
        message='Enter the inclination of the planes in degrees',
        default=45,
        float_allowed=True,
        min_allowed=0.0,
        max_allowed=180.0,
    ).execute()

    altitude_km = number.NumberPrompt(
        message='Enter the altitude of the satellites in kilometers',
        default=550,  # LEO
        float_allowed=True,
        min_allowed=400.0,  # VLEO
    ).execute()

    phasing_deg = number.NumberPrompt(
        message='Enter the phasing between planes in degrees',
        default=1,
        float_allowed=True,
        min_allowed=1.0,
        max_allowed=359.0,
    ).execute()

    return (
        int(total_sats),
        int(num_planes),
        float(inc_deg),
        float(altitude_km),
        float(phasing_deg),
    )


def _prompt_for_sensor(scenario: sim_config.SimConfig) -> str:
    return list_prompt.ListPrompt(
        message='Select a sensor',
        choices=[sensor for sensor in scenario.sensors],
        default=None,
    ).execute()


def _prompt_for_output() -> pathlib.Path:
    return pathlib.Path(
        filepath.FilePathPrompt(
            message='Enter the output file path',
            default=str(path_utils.SCENARIOS),
            validate=prompt_utils.PathValidator(
                message='Pick a `.yaml` path',
                is_file=True,
                must_exist=False,
            ),
            only_files=True,
        ).execute()
    )


@click.command()
def main() -> None:
    constellation = _prompt_for_constellation()

    if constellation == Constellelation.WALKER_DELTA:
        total_sats, num_planes, inc_deg, altitude_km, phasing_deg = (
            _prompt_for_walker_delta()
        )
        orbits = walker_delta.walker_delta(
            total_sats, num_planes, inc_deg, altitude_km, phasing_deg
        )
    else:
        orbits = []

    scenario = sim_config.load_default_sim_config()

    sensor = _prompt_for_sensor(scenario)

    # Update the scenario with the new orbits
    satellites = {
        f'Sat{i}': sim_config.Satellite(
            sensor=sensor,
            altitude=orbit.altitude.value,
            ecc=orbit.ecc.value,
            inc=orbit.inc.value,
            raan=orbit.raan.value,
            argp=orbit.argp.value,
            nu=orbit.nu.value,
            color='green',
        )
        for i, orbit in enumerate(orbits)
    }

    time, intent = next(iter(scenario.commanders_intent.keys())), next(
        iter(next(iter(scenario.commanders_intent.values())).values())
    )
    commanders_intent = {time: {sat: intent for sat in satellites}}

    new_scenario = dataclasses.replace(
        scenario, satellites=satellites, commanders_intent=commanders_intent
    )

    sim_config.save_sim_config(new_scenario, _prompt_for_output())


if __name__ == '__main__':
    main()
