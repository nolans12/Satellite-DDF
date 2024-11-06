from matplotlib import pyplot as plt

from common import path_utils
from phase3 import environment
from phase3 import sim_config
from phase3.constellation import walker_delta
from phase3.plotting import orbit


def _sample_walker_delta(save_animation: bool) -> None:
    # Generate a Walker Delta constellation
    total_sats = 24
    num_planes = 6
    inc_deg = 45
    altitude_km = 550
    phasing_deg = 1
    orbits = walker_delta.walker_delta(
        total_sats, num_planes, inc_deg, altitude_km, phasing_deg
    )

    # Plot the orbits
    fig, ani = orbit.plot_orbits(orbits)
    plt.show()

    if save_animation:
        # Save a gif of the animation
        ani.save(
            str(path_utils.REPO_ROOT / 'phase3/gifs/satellite_orbits.gif'),
            writer='imagemagick',
        )


def _sample_comm(save_animation: bool) -> None:
    scenario = sim_config.load_sim_config(path_utils.SCENARIOS / 'wd_45_30_3_1.yaml')

    # Create the environment
    env = environment.Environment.from_config(scenario)

    # Plot the communication links
    fig, ani = orbit.plot_comms(env.comms, frames=90)
    plt.show()

    if save_animation:
        # Save a gif of the animation
        ani.save(
            str(path_utils.REPO_ROOT / 'phase3/gifs/satellite_comms.gif'),
            writer='imagemagick',
        )


if __name__ == '__main__':
    # Change the backend
    plt.switch_backend('qtagg')

    # _sample_walker_delta(save_animation=False)
    _sample_comm(save_animation=True)
