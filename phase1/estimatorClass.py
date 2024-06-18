from import_libraries import *
## Creates the estimator class

class centralEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targetID
    
    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.measHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}

    # Input: All satellites and a single target
    # Output: A dictionary containing the satelite object and the measurement associated with that satellite
    def collectAllMeasurements(self, sats, targetID, envTime):
        satMeasurements = defaultdict(dict)
        for sat in sats: # check if a satellite viewed a target at this time
            if (hasattr(sat, 'raw_ECI_measHist') and 
                targetID in sat.raw_ECI_measHist and envTime in sat.raw_ECI_measHist[targetID]):                
                    
                    # Add the satellite and the measurement to the dictionary
                    satMeasurements[sat] = sat.raw_ECI_measHist[targetID][envTime]
            
        return satMeasurements

# CENTRALIZED EXTENDED KALMAN FILTER
    # Inputs: All satellite objects, targetID to estimate, and the environment time
    def EKF(self, sats, target, envTime):

    # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [x y z] ECI coordinates

        # First get the measurements from the satellites at given time and targetID
        targetID = target.targetID
        satMeasurements = self.collectAllMeasurements(sats, targetID, envTime)
# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
    # If no prior estimates, use the first measurement and assume no velocity
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0]) # start with true position, no velocity
            P_prior = np.array([[10, 0, 0, 0, 0, 0], # initalize positions to be +- 10 km and velocities to be +- 1 km/s
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 10, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 10, 0],
                                [0, 0, 0, 0, 0, 1]])
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.measHist[targetID][envTime] = satMeasurements
            self.innovationHist[targetID][envTime] = np.zeros(3)
            self.innovationCovHist[targetID][envTime] = np.eye(3)
            return est_prior
        
        else:
    # Else, get prior estimate, need to get the last time, which will be the max
            time_prior = max(self.estHist[targetID].keys())
            est_prior = self.estHist[targetID][time_prior]
            P_prior = self.covarianceHist[targetID][time_prior]

        # Now to get dt, use time since last measurement
        dt = envTime - time_prior

# CALCULATE MATRICES:
    # Define the state transition matrix, F.
    # Is a 6x6 matrix representing mapping b/w state at time k and time k+1
        # How does our state: [x, vx, y, vy, z, vz] change over time?
        F = np.array([[1, dt, 0, 0, 0, 0], # Assume no acceleration, just constant velocity over the time step
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])

    # Define the process noise matrix, Q.
    # Is a 6x6 matrix representing the covariance of the process noise
        # Estimate the randomness of the acceleration
        q_x = 0.0001
        q_y = 0.0001
        q_z = 0.00001
        q_mat = np.array([0, q_x, 0, q_y, 0, q_z])
        # TODO: LOOK INTO Van Loan's METHOD FOR TUNING Q
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]]) * q_mat**2

    # Define the obversation matrix, H.
    # STACK MATRICES
    # Is a 3Nx6 matrix representing the mapping b/w state and N measurement.
        # How does our state relate to our measurement? 
        H = np.zeros((3*len(satMeasurements), 6))
        for i, sat in enumerate(satMeasurements):
            H[3*i:3*i+3][0:6] = np.array([[1, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0]])

    # Define the sensor noise matrix, R.
    # STACK MATRICES
    # Is a 3Nx3N matrix representing the covariance of the sensor noise for each satellite's 3x1 measurement.
        # This is the covariance estimate of the sensor error
        # We need to stack this for each satellite
        for i, sat in enumerate(satMeasurements):
            R = self.calculate_R(sat, satMeasurements[sat])
            R = self.calculate_R_range(sat, satMeasurements[sat]) # For range and bearings
            if i == 0:
                R_stack = R
            else:
                R_stack = block_diag(R_stack, R)

        

# EXTRACT THE MEASUREMENTS
        z = np.array([satMeasurements[sat] for sat in satMeasurements]).flatten()

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior)
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# UPDATE:
    # Calculate innovation terms:
        innovation = z - np.dot(H, est_pred) # Difference between the measurement and the predicted measurement
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R_stack

    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

    # Correct prediction
        est = est_pred + np.dot(K, innovation)
        P = P_pred - np.dot(K, np.dot(H, P_pred))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.measHist[targetID][envTime] = satMeasurements
        self.innovationHist[targetID][envTime] = innovation 
        self.innovationCovHist[targetID][envTime] = innovationCov 

        return est


    # FOR BEARINGS            
    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R(self, sat, meas_ECI):
            
            # Convert the ECI measurement into a bearings and range measurement
            in_track_truth, cross_track_truth = sat.sensor.convert_to_bearings(sat, meas_ECI)

            # Get the error from the sensor.
            bearingsError = sat.sensor.bearingsError

            # Run the monte-carlo simulation
            numSims = 1000
            allErrors = np.zeros((numSims, 3))
            for i in range(numSims):
                    
                    # Add noise to the measurement
                    simMeas_bearings = np.array([in_track_truth + np.random.normal(0, bearingsError[0]), 
                                                cross_track_truth + np.random.normal(0, bearingsError[1])])
    
                    # Now calculate what this new bearings range measurement would be in ECI:
                    simMeas_ECI = sat.sensor.convert_from_bearings_to_ECI(sat, simMeas_bearings, meas_ECI)
                    # INPUT MEAS ECI AS THE POINT TO INTERSECT THE BEARINGS LINE WITH

                    # Now calculate the error between the truth and the simulated ECI measurement
                    allErrors[i] = simMeas_ECI - meas_ECI

            # Now calculate the covariance matrix of the error
            R = np.cov(allErrors.T)

            return R
    
    # FOR BEARINGS AND RANGE
    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R_range(self, sat, meas_ECI):

        # Convert the ECI measurement into a bearings and range measurement
        in_track_truth, cross_track_truth, range_truth = sat.sensor.convert_to_range_bearings(sat, meas_ECI) # Will treat this as the truth estimate!

        # Get the error from the sensor.
        bearingsError = sat.sensor.bearingsError
        rangeError = sat.sensor.rangeError

        # Run the monte-carlo simulation
        numSims = 1000
        allErrors = np.zeros((numSims, 3))
        for i in range(numSims):
            
            # Add noise to the measurement
            simMeas_bearings_range = np.array([in_track_truth + np.random.normal(0, bearingsError[0]), 
                                               cross_track_truth + np.random.normal(0, bearingsError[1]), 
                                               range_truth + np.random.normal(0, rangeError)])

            # Now calculate what this new bearings range measurement would be in ECI:
            simMeas_ECI = sat.sensor.convert_from_range_bearings_to_ECI(sat, simMeas_bearings_range)
            # INPUT MEAS ECI AS THE POINT TO INTERSECT THE BEARINGS LINE WITH

            # Now calculate the error between the truth and the simulated ECI measurement
            allErrors[i] = simMeas_ECI - meas_ECI

        # Now calculate the covariance matrix of the error
        R = np.cov(allErrors.T)

        return R
    
    
class localEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targets
        # TODO: add the input being a sensor error, so we can predefine an R

    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.measHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}

# LOCAL EXTENDED KALMAN FILTER
    # Inputs: Satellite with a new bearings and range measurement, targetID, dt since last measurement, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def EKF(self, sat, meas_ECI, target, envTime):

    # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [x y z] ECI coordinates

        targetID = target.targetID

# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
    # If no prior estimates, use the first measurement and assume no velocity
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0]) # start with true position, no velocity
            P_prior = np.array([[10, 0, 0, 0, 0, 0], # initalize positions to be +- 10 km and velocities to be +- 1 km/s
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 10, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 10, 0],
                                [0, 0, 0, 0, 0, 1]])
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.measHist[targetID][envTime] = meas_ECI
            self.innovationHist[targetID][envTime] = np.zeros(3)
            self.innovationCovHist[targetID][envTime] = np.eye(3)
            return est_prior
        
        else:
    # Else, get prior estimate, need to get the last time, which will be the max
            time_prior = max(self.estHist[targetID].keys())
            est_prior = self.estHist[targetID][time_prior]
            P_prior = self.covarianceHist[targetID][time_prior]

        # Now to get dt, use time since last measurement
        dt = envTime - time_prior

        # print("Integrating filter with time step: ", dt)

# CALCULATE MATRICES:
    # Define the state transition matrix, F. 
        # How does our state: [x, vx, y, vy, z, vz] change over time?
        F = np.array([[1, dt, 0, 0, 0, 0], # Assume no acceleration, just constant velocity over the time step
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])
        
    # Define the process noise matrix, Q.
        # Estimate the randomness of the acceleration
        q_x = 0.0001
        q_y = 0.0001
        q_z = 0.00001
        q_mat = np.array([0, q_x, 0, q_y, 0, q_z])
        # TODO: LOOK INTO Van loan's METHOD FOR TUNING Q
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]]) * q_mat**2

    # Define the obversation matrix, H.
        # How does our state relate to our measurement? 
        # Because we alredy converted our measurement to ECI, we can just use the identity matrix
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0]])
    # Define the sensor noise matrix, R.
        # This is the covariance estimate of the sensor error
        # Tuned using monte-carlo estimation at each timestep
        # R = self.calculate_R(sat, meas_ECI)
        R = np.eye(3)

# EXTRACT THE MEASUREMENTS
        z = meas_ECI

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior)
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

        # TODO: jacobian of ECI to bearings measurement
        # Want the size to be 2x6, when we multiply by our measurement, the bearings angles, we get the state
        H_test = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred)
        print("Jacobian: ", H_test)
        # now estimate the measurement
        est_meas = np.dot(H_test, est_pred)
        print("Estimate: ", est_meas)
        # get the true angle measurement
        true_meas = sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]]))
        print("True Measurement: ", true_meas)                                               



# UPDATE:
    # Calculate innovation terms:
        innovation = z - np.dot(H, est_pred) # Difference between the measurement and the predicted measurement
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R

    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

    # Correct prediction
        est = est_pred + np.dot(K, innovation)
        P = P_pred - np.dot(K, np.dot(H, P_pred))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.measHist[targetID][envTime] = z
        self.innovationHist[targetID][envTime] = innovation
        self.innovationCovHist[targetID][envTime] = innovationCov

        return est



    # FOR BEARINGS            
    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R(self, sat, meas_ECI):
            
            # Convert the ECI measurement into a bearings and range measurement
            in_track_truth, cross_track_truth = sat.sensor.convert_to_bearings(sat, meas_ECI)

            # Get the error from the sensor.
            bearingsError = sat.sensor.bearingsError

            # Run the monte-carlo simulation
            numSims = 1000
            allErrors = np.zeros((numSims, 3))
            for i in range(numSims):
                    
                    # Add noise to the measurement
                    simMeas_bearings = np.array([in_track_truth + np.random.normal(0, bearingsError[0]), 
                                                cross_track_truth + np.random.normal(0, bearingsError[1])])
    
                    # Now calculate what this new bearings range measurement would be in ECI:
                    simMeas_ECI = sat.sensor.convert_from_bearings_to_ECI(sat, simMeas_bearings, meas_ECI)
                    # INPUT MEAS ECI AS THE POINT TO INTERSECT THE BEARINGS LINE WITH

                    # Now calculate the error between the truth and the simulated ECI measurement
                    allErrors[i] = simMeas_ECI - meas_ECI

            # Now calculate the covariance matrix of the error
            R = np.cov(allErrors.T)

            return R
    
    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R_range(self, sat, meas_ECI):

        # Convert the ECI measurement into a bearings and range measurement
        in_track_truth, cross_track_truth, range_truth = sat.sensor.convert_to_range_bearings(sat, meas_ECI) # Will treat this as the truth estimate!

        # Get the error from the sensor.
        bearingsError = sat.sensor.bearingsError
        rangeError = sat.sensor.rangeError

        # Run the monte-carlo simulation
        numSims = 1000
        allErrors = np.zeros((numSims, 3))
        for i in range(numSims):
            
            # Add noise to the measurement
            simMeas_bearings_range = np.array([in_track_truth + np.random.normal(0, bearingsError[0]), 
                                               cross_track_truth + np.random.normal(0, bearingsError[1]), 
                                               range_truth + np.random.normal(0, rangeError)])

            # Now calculate what this new bearings range measurement would be in ECI:
            simMeas_ECI = sat.sensor.convert_from_range_bearings_to_ECI(sat, simMeas_bearings_range)

            # Now calculate the error between the truth and the simulated ECI measurement
            allErrors[i] = simMeas_ECI - meas_ECI

        # Now calculate the covariance matrix of the error
        R = np.cov(allErrors.T)

        # # print the maximum value in R
        # print("Max value in R: ", np.max(R))

        return R