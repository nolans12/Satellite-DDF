from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, indeptEstimator, name, color, ddfEstimator = None):
    
    # Sensor to use
        self.sensor = sensor
        self.measurementHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Initialize as a dictionary of dictornies for raw measurements. Index with targetID and time: t, sat ECI pos, sensor measurements
    
    # Targets to track:
        self.targetIDs = targetIDs

    # Estimator to use to benchmark against, worst case
        self.indeptEstimator = indeptEstimator
    
    # DDF estimator to test
        if ddfEstimator:
            self.ddfEstimator = ddfEstimator
        else:
            self.ddfEstimator = None

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
        self.orbitHist = defaultdict(dict) # contains time and xyz of orbit history
        self.time = 0

    def collect_measurements_and_filter(self, target):
        collectedFlag = 0
        if target.targetID in self.targetIDs:
        # Is the current target one of the ones to track?
            # If so, get the measurement
            measurement = self.sensor.get_measurement(self, target)
            # Make sure its not just a default 0, means target isnt visible
            if not isinstance(measurement, int):
            # If target is visible, save relavent data
                collectedFlag = 1

                # Save the measurement
                self.measurementHist[target.targetID][self.time] = measurement

                # Update the local filters
                self.update_local_filters(measurement, target, self.time)
                #print('Satellite:', self.name, 'has collected a measurement for target:', target.name, 'at time:', self.time)
                                                            
        return collectedFlag

    def update_local_filters(self, measurement, target, time):
        # Update the local filters
        self.indeptEstimator.EKF(self, measurement, target, time)
        if self.ddfEstimator:
            self.ddfEstimator.EKF(self, measurement, target, time)