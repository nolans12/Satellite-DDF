from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, name, color):
    
    # Sensor to use
        self.sensor = sensor
        self.measurementHist = {targetID: [] for targetID in targetIDs} # Initialize as a dictionary of lists
        
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
        # Loop through all targets
            if targ.targetID in self.targetIDs:
            # Is the current target one of the ones to track?
                # If so, get the measurement
                measurement = self.sensor.get_measurement(self, targ)
                # Make sure its not just a default 0, means target isnt visible
                if not isinstance(measurement, int):
                # If target is visible, save relavent data
                    # Need time, satellite positon, and measurement
                    save = np.array([self.time, self.orbit.r.value[0], self.orbit.r.value[1], self.orbit.r.value[2]])
                    save = np.append(save, measurement)
                    self.measurementHist[targ.targetID].append(save)
                    # print(self.name, "measures", targ.name, "at", measurement)