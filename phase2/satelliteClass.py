from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, indeptEstimator, name, color, ciEstimator = None, etEstimator = None, neighbors = None):
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

    # Sensor to use
        self.sensor = sensor
        self.measurementHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Initialize as a dictionary of dictornies for raw measurements. Index with targetID and time: t, sat ECI pos, sensor measurements
    
    # Targets to track:
        self.targetIDs = targetIDs
        
    # Topology to communciate with
        self.neighbors = [self, neighbors] # List of neighbors, will be updated by the topology class

    # Estimator to use to benchmark against, worst case
        self.indeptEstimator = indeptEstimator
    
    # DDF estimator to test
        if ciEstimator:
            self.ciEstimator = ciEstimator
        else:
            self.ciEstimator = None

    # ET estimator to test
        if etEstimator:
            self.etEstimator = etEstimator
        else:
            self.etEstimator = None
            
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
        """Collect measurements from the sensor for a specified target and update local filters.
                The satellite will use its sensor class to collect a measurement on the target.
                It then stores the measurement in its measurement history and updates its local filters.
                Updating hte local filters calls the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            target (object): Target object containing targetID and other relevant information.

        Returns:
            int: Flag indicating whether measurements were successfully collected (1) or not (0).
        """
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
                
        return collectedFlag

    def update_local_filters(self, measurement, target, time):
        """Update the local filters for the satellite.

        The satellite will update its local filters using the measurement provided.
        This will call the EKF functions to update the state and covariance estimates based on the measurement.

        Args:
            measurement (object): Measurement data obtained from the sensor.
            target (object): Target object containing targetID and other relevant information.
            time (float): Current time at which the measurement is taken.
        """
        # Update the local filters using the independent estimator
        self.indeptEstimator.EKF([self], [measurement], target, time)
        
        # If a DDF estimator is present, update the DDF filters using a local EKF 
        if self.ciEstimator:
            self.ciEstimator.EKF([self], [measurement], target, time)
            
    def update_et_estimator(self, etEstimator):
        self.etEstimator = etEstimator