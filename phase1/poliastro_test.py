import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import numpy as np

def satellite_orbit(a, ecc, inc, raan, argp, nu):
    """
    Computes the orbital elements for a satellite orbiting Earth using poliastro library.

    Parameters:
        a (float): Semi-major axis (km)
        ecc (float): Eccentricity
        inc (float): Inclination (radians)
        raan (float): Right Ascension of the Ascending Node (radians)
        argp (float): Argument of Periapsis (radians)
        nu (float): True Anomaly (radians)

    Returns:
        Orbit: Poliastro Orbit object representing the satellite's orbit around Earth.
    """
    # Define the satellite's orbit
    orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)

    return orbit

# Example usage
if __name__ == "__main__":
    # Define orbital elements (example values)
    semi_major_axis = 7000  # in kilometers
    eccentricity = 0.1
    inclination = np.radians(30)  # convert to radians
    raan = np.radians(45)  # convert to radians
    arg_perigee = np.radians(60)  # convert to radians
    true_anomaly = np.radians(90)  # convert to radians

    # Calculate satellite's orbit
    satellite_orbit = satellite_orbit(semi_major_axis, eccentricity, inclination, raan, arg_perigee, true_anomaly)

    # Print the orbital elements
    print("Semi-major Axis:", satellite_orbit.a)
    print("Eccentricity:", satellite_orbit.ecc)
    print("Inclination (degrees):", np.degrees(satellite_orbit.inc))
    print("RAAN (degrees):", np.degrees(satellite_orbit.raan))
    print("Argument of Perigee (degrees):", np.degrees(satellite_orbit.argp))
    print("True Anomaly (degrees):", np.degrees(satellite_orbit.nu))
