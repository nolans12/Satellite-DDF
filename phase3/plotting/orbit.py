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

from phase3 import comms
from phase3 import orbit
from phase3 import satellite


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
    for sat_name in comm.G.nodes:
        sat_name: str
        sat = comm._nodes[sat_name]
        if sat_name in comm._sensing_sats:
            color = 'red'
        else:
            color = 'yellow'
        r_s = sat.orbit.sample()
        ax.plot(
            r_s.xyz[0].to(u.km).value,
            r_s.xyz[1].to(u.km).value,
            r_s.xyz[2].to(u.km).value,
            color=color,
        )

    # Initialize points for each link
    (sensing_points,) = ax.plot([], [], [], 'go', markersize=5)
    (fusion_points,) = ax.plot([], [], [], 'co', markersize=5)
    # Initialize lines for each possible link
    lines = [
        ax.plot([], [], [], 'k--')[0]
        for _ in range(len(comm.G.nodes) * comm._config.max_neighbors)
    ]

    # Function to update the positions of the satellites and links
    def update(
        num: int,
        sensing: art3d.Line3D,
        fusion: art3d.Line3D,
        lines_: list[art3d.Line3D],
    ) -> list[art3d.Line3D]:
        s_x_s = []
        s_y_s = []
        s_z_s = []
        f_x_s = []
        f_y_s = []
        f_z_s = []
        for sat_name in comm.G.nodes:
            sat_name: str
            sat: satellite.Satellite = comm._nodes[sat_name]
            updated_orbit = dataclasses.replace(
                sat._orbit_params,
                nu=sat.orbit.nu + u.Quantity(1, u.deg),
            ).to_poliastro()
            sat.orbit = updated_orbit
            x, y, z = updated_orbit.r.value
            if sat_name in comm._sensing_sats:
                s_x_s.append(x)
                s_y_s.append(y)
                s_z_s.append(z)
            elif sat_name in comm._fusion_sats:
                f_x_s.append(x)
                f_y_s.append(y)
                f_z_s.append(z)
        sensing.set_data_3d(s_x_s, s_y_s, s_z_s)
        fusion.set_data_3d(f_x_s, f_y_s, f_z_s)

        # Update the network graph
        comm.update_edges()

        for i, (u_, v) in enumerate(comm.G.edges):
            u_: str
            v: str
            sat_u: satellite.Satellite = comm._nodes[u_]
            sat_v: satellite.Satellite = comm._nodes[v]
            if not u_ in comm._sensing_sats and not v in comm._sensing_sats:
                continue
            x_u, y_u, z_u = sat_u.orbit.r.value
            x_v, y_v, z_v = sat_v.orbit.r.value
            lines_[i].set_data_3d(
                [x_u, x_v],
                [y_u, y_v],
                [z_u, z_v],
            )

        # Reset the lines that are not used
        for i in range(len(comm.G.edges), len(lines_)):
            lines_[i].set_data_3d([], [], [])

        return [sensing, fusion] + lines_

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        fargs=(sensing_points, fusion_points, lines),
        interval=50,
        blit=False,
    )

    # Set the title
    ax.set_title('Satellite Communication Links')

    # Set the labels
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')

    return fig, ani


def plot_orbits(
    sensing_orbits: list[orbit.Orbit],
    fusion_orbits: list[orbit.Orbit],
    frames: int = 360,
) -> tuple[figure.Figure, animation.FuncAnimation]:
    """Create an interactive 3D plot of the satellite orbits.

    Args:
        orbits: List of Orbit objects to plot.
    """
    fig, ax = _create_earth_plot()

    # Plot the orbit lines
    for orb in sensing_orbits:
        r_s = orb.to_poliastro().sample()
        ax.plot(
            r_s.xyz[0].to(u.km).value,
            r_s.xyz[1].to(u.km).value,
            r_s.xyz[2].to(u.km).value,
            color='red',
        )
    for orb in fusion_orbits:
        r_s = orb.to_poliastro().sample()
        ax.plot(
            r_s.xyz[0].to(u.km).value,
            r_s.xyz[1].to(u.km).value,
            r_s.xyz[2].to(u.km).value,
            color='yellow',
        )

    # Initialize points for each orbit
    (sensing_points,) = ax.plot([], [], [], 'go', markersize=5)
    (fusion_points,) = ax.plot([], [], [], 'co', markersize=5)

    # Function to update the positions of the satellites
    def update(
        num: int, sensing: art3d.Line3D, fusion: art3d.Line3D
    ) -> tuple[art3d.Line3D, ...]:
        x_s = []
        y_s = []
        z_s = []
        for orb in sensing_orbits:
            orb = dataclasses.replace(orb, nu=orb.nu + u.Quantity(num * 1, u.deg))
            x, y, z = orb.to_poliastro().r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
        sensing.set_data_3d(x_s, y_s, z_s)
        x_s = []
        y_s = []
        z_s = []
        for orb in fusion_orbits:
            orb = dataclasses.replace(orb, nu=orb.nu + u.Quantity(num * 1, u.deg))
            x, y, z = orb.to_poliastro().r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
        fusion.set_data_3d(x_s, y_s, z_s)
        return (sensing, fusion)

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        fargs=(sensing_points, fusion_points),
        interval=50,
        blit=False,
    )

    # Set the title
    ax.set_title('Satellite Orbits')

    # Set the labels
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')

    return fig, ani
