from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, fov, sensorDetectError, sensorError, name, color):
    # Define orbital elements for poliastro orbit

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
        self.orbitHist = [];

    # Define sensor settings
        self.fov = fov
        self.sensorDetectError = sensorDetectError
        self.sensorError = sensorError

    # Other parameters
        self.name = name
        self.color = color