# Main script for phase 1 of the Satellite-DDF project
# To start, generate a polar satellite orbit using poliastro library
# Import the necessary libraries
import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
from mpl_toolkits.mplot3d import Axes3D

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

def render_gif(imgs, file, frame_duration=0.25):
    with imageio.get_writer(file, mode='I', duration=frame_duration) as writer:
        for img in imgs:
            writer.append_data(img)

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

    # Now try and get a simulation of the orbit
    # Define a time range
    time_range = np.linspace(0 * u.day, 1/50 * u.day, num=50)

    # Propagate the orbit over the time range, saving the img plots
    imgs = []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = 6378 * np.outer(np.cos(u), np.sin(v))
    y_earth = 6378 * np.outer(np.sin(u), np.sin(v))
    z_earth = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x, y, z, color='b', alpha=0.1)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Satellite Orbit Visualization')
    for i, time in enumerate(time_range):
        orb_propagated = orb.propagate(time)

        # Extract the XYZ coordinates of the orbit
        x, y, z = orb_propagated.r.value
        
        # Clear the previous scatter point
        ax.cla()
        
        # Plot Earth
        ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.1)
        
        # Plot the current position of the satellite
        ax.scatter(x, y, z, s=40, label='Satellite Orbit')

        # Annotate with the current time
        ax.text(x, y, z, f"Time: {time_range[i]:.2f}", color='red')

        plt.legend()
        plt.pause(0.1)  # Pause to display the plot
        plt.draw()  # Update the plot

        ios = io.BytesIO()
        fig.savefig(ios, format='raw')  # RGBA
        ios.seek(0)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(np.frombuffer(ios.getvalue(), dtype=np.uint8), (int(h), int(w), 4))[:, :, 0:4]
        imgs.append(img)

    plt.show()

    # Save the images as a GIF
    render_gif(imgs, 'satellite_orbit.gif', frame_duration=0.10)