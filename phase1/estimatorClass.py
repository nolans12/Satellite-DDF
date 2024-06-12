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
        
        self.time = 0

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
    def EKF(self, sats, targetID, envTime):

    # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [x y z] ECI coordinates

        # First get the measurements from the satellites at given time and targetID
        satMeasurements = self.collectAllMeasurements(sats, targetID, envTime)
    
# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
        # If no prior estimate exists, just use the measurement from the first satellite
            # Find the first satellite that has a measurement
            for sat in satMeasurements:
                if sat in satMeasurements:
                    est_prior = np.array([satMeasurements[sat][0], 0, satMeasurements[sat][1], 0, satMeasurements[sat][2], 0])
                    break
            P_prior = np.eye(6)
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.measHist[targetID][envTime] = satMeasurements
            return est_prior
        else:
            # To get the prior estimate, need to get the last time, which will be the max
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
        # TODO: LOOK INTO Van der Merwe's METHOD FOR TUNING Q
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
            if i == 0:
                R_stack = R
            else:
                R_stack = block_diag(R_stack, R)

        # R_stack = np.eye(3*len(satMeasurements)) * 0.01

# EXTRACT THE MEASUREMENTS
        z = np.array([satMeasurements[sat] for sat in satMeasurements]).flatten()

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior)
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# UPDATE:
    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_pred, H.T)) + R_stack)))

    # Correct prediction
        est = est_pred + np.dot(K, z - np.dot(H, est_pred))
        P = P_pred - np.dot(K, np.dot(H, P_pred))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.measHist[targetID][envTime] = satMeasurements

        return est
                
    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R(self, sat, meas_ECI):

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
            simMeas_ECI = sat.sensor.convert_to_ECI(sat, simMeas_bearings_range)

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



    # EXTENDED KALMAN FILTER
    # Inputs: Satellite with a new bearings and range measurement, targetID, dt since last measurement, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def EKF(self, sat, meas_ECI, targetID, envTime):

    # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [x y z] ECI coordinates

# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
        # If no prior estimates, use the first measurement and assume no velocity
            est_prior = np.array([meas_ECI[0], 0, meas_ECI[1], 0, meas_ECI[2], 0]) 
            P_prior = np.eye(6)
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.measHist[targetID][envTime] = meas_ECI
            return est_prior
        else:
            # To get the prior estimate, need to get the last time, which will be the max
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
        # TODO: LOOK INTO Van der Merwe's METHOD FOR TUNING Q
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]]) * q_mat**2
        
        # print("Max of Orig: ", np.max(Q))

    # Define the obversation matrix, H.
        # How does our state relate to our measurement? 
        # Because we alredy converted our measurement to ECI, we can just use the identity matrix
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0]])

    # Define the sensor noise matrix, R.
        # This is the covariance estimate of the sensor error
        # Tuned using monte-carlo estimation at each timestep
        R = self.calculate_R(sat, meas_ECI) 
        # R = np.eye(3) * 0.01

# EXTRACT THE MEASUREMENTS
        z = meas_ECI

# # ATTEMPT VAN DER MERWES METHOD
#         # Compute the innovation:
#         y = meas_ECI - np.dot(H, est_prior)

#         # Compute the innovation covariance:
#         S = np.dot(H, np.dot(P_prior, H.T)) + R

#         # Compute the expected innovation covariance?

#         # Compute the Kalman gain:
#         S_inv = np.linalg.inv(S)
#         K = np.dot(P_prior, np.dot(H.T, S_inv))

#         # Update the process noise covariance:
#         Q = np.dot(K, np.dot(S, K.T))

#         # print("Max of New: ", np.max(Q))


# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior)
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# UPDATE:
    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_pred, H.T)) + R)))

    # Correct prediction
        est = est_pred + np.dot(K, z - np.dot(H, est_pred))
        P = P_pred - np.dot(K, np.dot(H, P_pred))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.measHist[targetID][envTime] = z

        return est


    # Input: A satellite object, and a ECI measurement
    # Output: The Covariance matrix of measurement noise, R
    # Description: Uses monte-carlo estimation. Treats the ECI measurement as truth and run X nums of sims with sensor noise to estimate R. 
    def calculate_R(self, sat, meas_ECI):

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
            simMeas_ECI = sat.sensor.convert_to_ECI(sat, simMeas_bearings_range)

            # Now calculate the error between the truth and the simulated ECI measurement
            allErrors[i] = simMeas_ECI - meas_ECI

        # Now calculate the covariance matrix of the error
        R = np.cov(allErrors.T)

        # # print the maximum value in R
        # print("Max value in R: ", np.max(R))

        return R