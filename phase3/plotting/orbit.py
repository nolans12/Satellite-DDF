import numpy as np
from astropy import units as u
from matplotlib import animation
from matplotlib import figure
from matplotlib import pyplot as plt
from poliastro import bodies
from poliastro import twobody

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
    ax = fig.add_subplot(111, projection="3d")

    # Plot the Earth
    u_, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = bodies.Earth.R.to(u.km).value * np.cos(u_) * np.sin(v)
    y = bodies.Earth.R.to(u.km).value * np.sin(u_) * np.sin(v)
    z = bodies.Earth.R.to(u.km).value * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.4, zorder=1)

    # Initialize points for each orbit
    (points,) = ax.plot([], [], [], 'go', markersize=5, zorder=3)

    frames = 360

    # Function to update the positions of the satellites
    def update(num, orbits_, points_):
        count = 0
        x_s = []
        y_s = []
        z_s = []
        for orb in orbits_:
            a, e, inc, raan, argp, nu = (
                bodies.Earth.R.to(u.km) + orb.altitude,
                orb.ecc,
                orb.inc,
                orb.raan,
                orb.argp,
                # Simple true anomaly update for circular orbits
                orb.nu + (num * (360 / frames)) * u.deg,
            )
            orbit = twobody.Orbit.from_classical(
                bodies.Earth, a, e, inc, raan, argp, nu
            )
            x, y, z = orbit.r.value
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
            count += 1
        points_.set_data(x_s, y_s)
        points_.set_3d_properties(z_s)
        return points_

    # Plot the orbit lines
    for orb in orbits:
        a, e, inc, raan, argp, nu = (
            bodies.Earth.R.to(u.km) + orb.altitude,
            orb.ecc,
            orb.inc,
            orb.raan,
            orb.argp,
            orb.nu,
        )
        orbit = twobody.Orbit.from_classical(bodies.Earth, a, e, inc, raan, argp, nu)
        r = orbit.sample()
        ax.plot(
            r.x.to(u.km).value,
            r.y.to(u.km).value,
            r.z.to(u.km).value,
            color='red',
            zorder=2,
        )

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
