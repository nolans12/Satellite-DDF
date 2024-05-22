from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, estimator, name, color):
    
    # Sensor to use
        self.sensor = sensor
        self.measurementHist = {targetID: [] for targetID in targetIDs} # Initialize as a dictionary of lists for raw measurements: t, sat ECI pos, sensor measurements
        self.rawEstimateHist = {targetID: [] for targetID in targetIDs} # Initialize as a dictionary of lists for raw estimates: t, targ ECI pos

    # Targets to track:
        self.targetIDs = targetIDs

    # Estimator to use
        self.estimator = estimator

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
        self.orbitHist = []; # contains time and xyz of orbit history
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
                    saveMeas = np.array([self.time, self.orbit.r.value[0], self.orbit.r.value[1], self.orbit.r.value[2]])
                    saveMeas = np.append(saveMeas, measurement)
                    self.measurementHist[targ.targetID].append(saveMeas) # Format is: [time, x, y, z, alpha, beta] in ECI coordinates of satellite

                    # Also save raw Estimate of target in ECI
                    saveEst = np.array([self.time])
                    rawEst = self.sensor.convert_to_ECI(self, measurement)
                    saveEst = np.append(saveEst, rawEst)         
                    self.rawEstimateHist[targ.targetID].append(saveEst) # Format is: [time, x, y, z] in ECI coordinates of target

                    # print rawEst and the truth position of the target
                    print("Raw Estimate of", targ.name, "is", rawEst, "and the truth position is", targ.pos)

                    # Also just print norm distance between the two for now
                    print("Distance between raw estimate and truth position is", np.linalg.norm(rawEst - targ.pos))


        