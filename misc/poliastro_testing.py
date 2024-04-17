# Main script for phase 1 of the Satellite-DDF project
# To start, generate a polar satellite orbit using poliastro library
# Import the necessary libraries
import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
# Need 3.7 version! pip install matplotlib==3.7
from mpl_toolkits.mplot3d import Axes3D
from poliastro.plotting import OrbitPlotter3D

# pio.renderers.default = "plotly_mimetype+notebook_connected"


# Create a satellite orbit using poliastro
def satellite_orbit(a, ecc, inc, raan, argp, nu):
    """
    Computes the orbital elements for a satellite orbiting Earth using poliastro library.

    Parameters:
        a : Semi-major axis (u.km)
        ecc : Eccentricity (u.dimensionless_unscaled)
        inc : Inclination (u.deg)
        raan : Right Ascension of the Ascending Node (u.deg)
        argp : Argument of Periapsis (u.deg)
        nu : True Anomaly (u.deg)

    Returns:
        Orbit: Poliastro Orbit object representing the satellite's orbit around Earth.
    """
    # Define the satellite's orbit
    orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)

    return orbit

def plot_orbit(orbit):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6378 * np.outer(np.cos(u), np.sin(v))
    y = 6378 * np.outer(np.sin(u), np.sin(v))
    z = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot orbit
    orbit.plot(label='Satellite Orbit')

    plt.legend()
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Satellite Orbit Visualization')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define orbital elements (example values)
    semi_major_axis = 7000 * u.km  # in kilometers
    eccentricity = 0.1 * u.dimensionless_unscaled
    inclination = 30 * u.deg  # degrees
    raan = 45 * u.deg  # degrees
    arg_perigee = 60 * u.deg  # degrees
    true_anomaly = 90 * u.deg  # degrees

    # Calculate satellite's orbit
    orb = satellite_orbit(semi_major_axis, eccentricity, inclination, raan, arg_perigee, true_anomaly)

    # Print the orbital elements
    # print("Semi-major Axis:", orb.a)
    # print("Eccentricity:", orb.ecc)
    # print("Inclination (degrees):", orb.inc.to(u.deg))
    # print("RAAN (degrees):", orb.raan.to(u.deg))
    # print("Argument of Perigee (degrees):", orb.argp.to(u.deg))
    # print("True Anomaly (degrees):", orb.nu.to(u.deg))
    # print("Period:", orb.period.to(u.day))
    print("Epoch:", orb.epoch)

    # Now try propagating the orbit
    # Define a time, just a scalar value for time of flight
    time_of_flight = 2 * u.day
    # Propagate the orbit
    orb_propagated = orb.propagate(time_of_flight)
    print("Epoch:", orb_propagated.epoch)

    # Visualize the orbit
    plot_orbit(orb)
    # plot_orbit(orb_propagated)

    # Now try and get a simulation of the orbit
    # Define a time range
    time_range = np.linspace(0 * u.day, 2 * u.day, num=10)

    # Propagate the orbit over the time range
    orb_simulated = [orb.propagate(t) for t in time_range]

    # Create the OrbitPlotter3D object
    plotter = OrbitPlotter3D()

    # Plot the orbit
    plotter.plot(orb)

    # Show the plot
    # plotter.show()
