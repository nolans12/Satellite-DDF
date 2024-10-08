import dataclasses
from typing import cast

import numpy as np
from astropy import units as u
from matplotlib import animation
from matplotlib import figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import axes3d
from poliastro import bodies

from phase3 import orbit


def plot_orbits(
    orbits: list[orbit.Orbit],
) -> tuple[figure.Figure, animation.FuncAnimation]:
    """Create an interactive 3D plot of the satellite orbits.

    Args:
        orbits: List of Orbit objects to plot.
    """
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax: axes3d.Axes3D = cast(
        axes3d.Axes3D, fig.add_subplot(111, projection="3d", computed_zorder=False)
    )

    # Plot the Earth
    u_, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = bodies.Earth.R.to(u.km).value * np.cos(u_) * np.sin(v)
    y = bodies.Earth.R.to(u.km).value * np.sin(u_) * np.sin(v)
    z = bodies.Earth.R.to(u.km).value * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.6)

    # Plot the orbit lines
    for orb in orbits:
        r = orb.to_poliastro().sample()
        ax.plot(
            r.x.to(u.km).value,
            r.y.to(u.km).value,
            r.z.to(u.km).value,
            color='red',
        )

    # Initialize points for each orbit
    (points,) = ax.plot([], [], [], 'go', markersize=5)

    frames = 360

    # Function to update the positions of the satellites
    def update(
        num: int, orbits_: list[orbit.Orbit], points_: art3d.Line3D
    ) -> tuple[art3d.Line3D]:
        count = 0
        x_s = []
        y_s = []
        z_s = []
        for orb in orbits_:
            updated_orbit = dataclasses.replace(
                orb, nu=orb.nu + (num * (360 / frames)) * u.deg
            ).to_poliastro()
            x, y, z = updated_orbit.r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
            count += 1
        points_.set_data_3d(x_s, y_s, z_s)
        return (points_,)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=frames, fargs=(orbits, points), interval=50, blit=False
    )

    # Set the title
    ax.set_title("Satellite Orbits")

    # Set the labels
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    return fig, ani


if __name__ == '__main__':
    from phase3.constellation import walker_delta

    # Change the backend
    plt.switch_backend('qtagg')

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
    fig, ani = plot_orbits(orbits)
    plt.show()

    # Save a gif of the animation
    # from common import path_utils

    # ani.save(
    #     str(path_utils.REPO_ROOT / 'satellite_orbits.gif'),
    #     writer='imagemagick',
    # )
