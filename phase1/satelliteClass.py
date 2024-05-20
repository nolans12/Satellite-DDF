from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, name, color):
    
    # Sensor to use
        self.sensor = sensor
        self.measurementHist = [] # contains xyz and t of measurements
        
    # Targets to track:
        self.targetIDs = targetIDs

    # Other parameters
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
        self.orbitHist = []; # contains xyz and time of orbit history
        self.orbitHistPlot = []; # contains xyz of history, fo
        self.time = 0

    def collect_measurements(self, targs):
        for i, targ in enumerate(targs):
            measurement = self.sensor.get_measurement(self, targ)
            if measurement != 0:
                self.measurementHist.append(measurement)
        self.orbitHistPlot.append(self.orbit.r.value)
        self.orbitHist.append([self.orbit.r.value[0], self.orbit.r.value[1], self.orbit.r.value[2], self.time])
