from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, sensor, a, ecc, inc, raan, argp, nu, name, color):
        """Initialize a Satellite object.

        Args:
            a (float or int): Semi-major axis of the satellite's orbit.
            ecc (float or int): Eccentricity of the satellite's orbit.
            inc (float or int): Inclination of the satellite's orbit in degrees.
            raan (float or int): Right ascension of the ascending node in degrees.
            argp (float or int): Argument of periapsis in degrees.
            nu (float or int): True anomaly in degrees.
            sensor (object): Sensor used by the satellite.
            target_ids (list): List of target IDs to track.
            indept_estimator (object): Independent estimator for benchmarking.
            name (str): Name of the satellite.
            color (str): Color of the satellite for visualization.
            ddf_estimator (object, optional): DDF estimator to test. Defaults to None.
        """

        self.sensor = sensor
        self.name = name
        self.color = color
        
    # Create the orbit
        # Check if already in units, if not convert
        if type(a) == int:
            a = a * u.km
        self.a = a
        if type(ecc) == int:
            ecc = ecc * u.dimensionless_unscaled
        self.ecc = ecc 
        if type(inc) == int:
            inc = inc * u.deg
        self.inc = inc 
        if type(raan) == int:
            raan = raan * u.deg
        self.raan = raan 
        if type(argp) == int:
            argp = argp * u.deg
        self.argp = argp 
        if type(nu) == int:
            nu = nu * u.deg
        self.nu = nu 
        
        # Create the poliastro orbit
        self.orbit = Orbit.from_classical(Earth, self.a, self.ecc, self.inc, self.raan, self.argp, self.nu)
        self.time = 0