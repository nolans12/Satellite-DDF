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

from canonical import comms
from canonical import orbit
from canonical import satellite


def _create_earth_plot() -> tuple[figure.Figure, axes3d.Axes3D]:
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

    return fig, ax


def plot_comms(
    comm: comms.Comms,
    frames: int = 360,
) -> tuple[figure.Figure, animation.FuncAnimation]:
    """Create an interactive 3D plot of the satellite communication links.

    Args:
        comm: Comms object to plot.
    """
    fig, ax = _create_earth_plot()

    # Plot the orbit lines
    for sat in comm.G.nodes:
        sat: satellite.Satellite
        r_s = sat.orbit.sample()
        ax.plot(
            r_s.xyz[0].to(u.km).value,
            r_s.xyz[1].to(u.km).value,
            r_s.xyz[2].to(u.km).value,
            color='red',
        )

    # Initialize points for each link
    (points,) = ax.plot([], [], [], 'go', markersize=5)
    # Initialize lines for each possible link
    lines = [
        ax.plot([], [], [], 'c--')[0]
        for _ in range(len(comm.G.nodes) * comm.max_neighbors)
    ]

    # Function to update the positions of the satellites and links
    def update(
        num: int, points_: art3d.Line3D, lines_: list[art3d.Line3D]
    ) -> list[art3d.Line3D]:
        x_s = []
        y_s = []
        z_s = []
        for sat in comm.G.nodes:
            sat: satellite.Satellite
            updated_orbit = dataclasses.replace(
                sat._orbit_params,
                nu=sat.orbit.nu + u.Quantity(1, u.deg),
            ).to_poliastro()
            sat.orbit = updated_orbit
            x, y, z = updated_orbit.r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
        points_.set_data_3d(x_s, y_s, z_s)

        # Update the network graph
        comm.update_edges()

        for i, (u_, v) in enumerate(comm.G.edges):
            u_: satellite.Satellite
            v: satellite.Satellite
            x_u, y_u, z_u = u_.orbit.r.value
            x_v, y_v, z_v = v.orbit.r.value
            lines_[i].set_data_3d(
                [x_u, x_v],
                [y_u, y_v],
                [z_u, z_v],
            )

        # Reset the lines that are not used
        for i in range(len(comm.G.edges), len(lines_)):
            lines_[i].set_data_3d([], [], [])

        return [points_] + lines_

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        fargs=(points, lines),
        interval=50,
        blit=False,
    )

    # Set the title
    ax.set_title("Satellite Communication Links")

    # Set the labels
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    return fig, ani


def plot_orbits(
    orbits: list[orbit.Orbit],
    frames: int = 360,
) -> tuple[figure.Figure, animation.FuncAnimation]:
    """Create an interactive 3D plot of the satellite orbits.

    Args:
        orbits: List of Orbit objects to plot.
    """
    fig, ax = _create_earth_plot()

    # Plot the orbit lines
    for orb in orbits:
        r_s = orb.to_poliastro().sample()
        ax.plot(
            r_s.xyz[0].to(u.km).value,
            r_s.xyz[1].to(u.km).value,
            r_s.xyz[2].to(u.km).value,
            color='red',
        )

    # Initialize points for each orbit
    (points,) = ax.plot([], [], [], 'go', markersize=5)

    # Function to update the positions of the satellites
    def update(num: int, points_: art3d.Line3D) -> tuple[art3d.Line3D]:
        x_s = []
        y_s = []
        z_s = []
        for orb in orbits:
            orb = dataclasses.replace(orb, nu=orb.nu + u.Quantity(num * 1, u.deg))
            x, y, z = orb.to_poliastro().r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
        points_.set_data_3d(x_s, y_s, z_s)
        return (points_,)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=frames, fargs=(points,), interval=50, blit=False
    )

    # Set the title
    ax.set_title("Satellite Orbits")

    # Set the labels
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    return fig, ani
