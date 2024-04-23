from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, fov, sensorDetectError, sensorError, name, color):
    # Define orbital elements for poliastro orbit
        self.a = a * u.km
        self.ecc = ecc * u.dimensionless_unscaled
        self.inc = inc * u.deg
        self.raan = raan * u.deg
        self.argp = argp * u.deg
        self.nu = nu * u.deg
        
        # Create the poliastro orbit
        self.orbit = Orbit.from_classical(Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)
        
    # Define sensor settings
        self.fov = fov
        self.sensorDetectError = sensorDetectError
        self.sensorError = sensorError

    # Other parameters
        self.name = name
        self.color = color