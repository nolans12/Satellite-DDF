from import_libraries import *
## Creates the satellite class, will contain the poliastro orbit and all other parameters needed to define the orbit

class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, sensor, targetIDs, estimator, name, color):
    
    # Sensor to use
        self.sensor = sensor
        self.measurementHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Initialize as a dictionary of dictornies for raw measurements. Index with targetID and time: t, sat ECI pos, sensor measurements
        self.raw_ECI_MeasimateHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Initialize as a dictionary of dictonaries for raw estimates. Index with targetID and time:: t, targ ECI pos

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
        self.orbitHist = defaultdict(dict) # contains time and xyz of orbit history
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
                    saveMeas = np.array([self.orbit.r.value[0], self.orbit.r.value[1], self.orbit.r.value[2]])
                    saveMeas = np.append(saveMeas, measurement)
                    self.measurementHist[targ.targetID][self.time] = saveMeas # Index with targetID and time, Format is [x, y, z, alpha, beta] in ECI coordinates of satellite

                    # Also save raw Estimate of target in ECI
                    raw_ECI_Meas = self.sensor.convert_to_ECI(self, measurement)
                    self.raw_ECI_MeasimateHist[targ.targetID][self.time] = raw_ECI_Meas # Index with targetID and time, Format is [x, y, z] in ECI coordinates of target

                    # # Local Kalman Filter on raw Estimate
                    dt = 1
                    estimate = self.estimator.EKF(raw_ECI_Meas, targ.targetID, dt, self.time)  
                    # print("Estimate of", targ.name, "is", estimate)
                    # #print("Distance between estimate and truth position is", np.linalg.norm(estimate - targ.pos))
                    # print("\n")
                    print("=" * 50)
                    print(f"{'SATELLITE AND TARGET INFORMATION':^50}")
                    print("=" * 50)
                    print("Satellite:", self.name, "Target:", targ.name)
                    print(f"{'True Position:':<15} {tuple(round(coord, 2) for coord in targ.pos)}")
                    print(f"{'True Velocity:':<15} {tuple(round(vel, 2) for vel in targ.vel)}")
                    print("=" * 50)
                    print(f"{'Raw Measurement and Kalman Filter Estimate':^100}")
                    print("=" * 100)
                    print(f"{'Raw Measurement (ECI):':<40} {tuple(round(coord, 2) for coord in raw_ECI_Meas)}")
                    print(f"{'Kalman Filter Position Estimate:':<40} {tuple(round(coord, 2) for coord in estimate[::2])}")
                    print(f"{'Kalman Filter Velocity Estimate:':<40} {tuple(round(vel, 2) for vel in estimate[1::2])}")
                    print("=" * 100)
                    print(f"{'Distance (Norm) between Measurement and Truth:':<40} {round(np.linalg.norm(raw_ECI_Meas - targ.pos), 2)}")
                    print(f"{'Distance (Norm) between Estimate and Truth:':<40} {round(np.linalg.norm([estimate[i] - targ.pos[i//2] for i in range(0, 6, 2)]), 2)}")
                    print(f"{'Velocity (Norm) between Estimate and Truth:':<30} {round(np.linalg.norm([estimate[i] - targ.vel[i//2] for i in range(1, 7, 2)]), 2)}")
                    print("\n")
                                        


        