from import_libraries import *
## Creates the estimator class

class centralEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targetID
    
    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates. the kalman estimate
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} 

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}

        self.neesHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.nisHist = {targetID: defaultdict(dict) for targetID in targetIDs}

    # Input: All satellites and a single target
    # Output: A dictionary containing the satelite object and the measurement associated with that satellite
    def collectAllMeasurements(self, sats, targetID, envTime):
        satMeasurements = defaultdict(dict)
        for sat in sats: # check if a satellite viewed a target at this time
            if (hasattr(sat, 'measurementHist') and 
                targetID in sat.measurementHist and envTime in sat.measurementHist[targetID]):                
                    
                    # Add the satellite and the measurement to the dictionary
                    satMeasurements[sat] = sat.measurementHist[targetID][envTime]
            
        return satMeasurements

# CENTRALIZED EXTENDED KALMAN FILTER
    # Inputs: All satellite objects, targetID to estimate, and the environment time
    # Output: New estimate in ECI
    def EKF(self, sats, target, envTime):
    
    # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: [in_track, cross_track] bearings measurement

        # First get the measurements from the satellites at given time and targetID
        targetID = target.targetID
        satMeasurements = self.collectAllMeasurements(sats, targetID, envTime)
        
# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
    # If no prior estimates, use the first measurement and assume no velocity
            # start with true position plus some noise
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0]) + np.random.normal(0, 2, 6) 
            # start with some covariance, about +- 5 km and +- 20 km/min, then plus some noise 
            P_prior = np.array([[5, 0, 0, 0, 0, 0],
                                [0, 20, 0, 0, 0, 0],
                                [0, 0, 5, 0, 0, 0],
                                [0, 0, 0, 20, 0, 0],
                                [0, 0, 0, 0, 5, 0],
                                [0, 0, 0, 0, 0, 20]]) + np.random.normal(0, 1, (6,6))*np.eye(6)
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
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
        Q = self.calculate_Q(dt)

    # Define the sensor nonise matrix, R.
    # Needs to be stacked for each satellite
        for i, sat in enumerate(satMeasurements):
            R_curr = np.eye(2) * sat.sensor.bearingsError**2
            if i == 0:
                R_stack = R_curr
            else:
                R_stack = block_diag(R_stack, R_curr)

# EXTRACT THE MEASUREMENTS
        z = np.array([satMeasurements[sat] for sat in satMeasurements]).flatten()

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior)
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# Use the predicted state to calculate H
    # Need this to be stacked for each satellite
    # Each H will be a 2x6 matrix
        H = np.zeros((2*len(satMeasurements), 6))
        for i, sat in enumerate(satMeasurements):
            H[2*i:2*i+2][0:6] = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred)

# UPDATE:
    # Calculate innovation terms:
        # We need to calculate the innovation for each satellite
        for i, sat in enumerate(satMeasurements):
            innovation_curr = satMeasurements[sat] - sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]])) # Difference between the measurement and the predicted measurement
            if i == 0:
                innovation = innovation_curr
            else:
                innovation = np.append(innovation, innovation_curr)

        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R_stack

    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

    # Correct prediction
        est = est_pred + np.dot(K, innovation)
        P = P_pred - np.dot(K, np.dot(H, P_pred))

    # CALCUATE NEES AND NIS
        # For nees, we need to get the error compared to the truth data:
        # Get the true position
        true = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]])
        # Get the error
        error = est - true
        # Get the covariance of the error
        nees = np.dot(error.T, np.dot(np.linalg.inv(P), error))
        nis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = innovation
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis


    def calculate_Q(self, dt, intensity=np.array([0.001, 5, 5])):
        # Use Van Loan's method to tune Q using the matrix exponential
        
        # Define the state transition matrix, A.
        A = np.array([[0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        # Assume there could be noise impacting the cartesian acceleration
        Gamma = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
        
        # Assing a maximum intensity of the noise --> 0.001 km/min^2 = 1 m/min^2 over the time step
        
        rangeNoise, elevationNoise, azimuthNoise = intensity
        x = rangeNoise * np.cos(elevationNoise) * np.cos(azimuthNoise)
        y = rangeNoise * np.cos(elevationNoise) * np.sin(azimuthNoise)
        z = rangeNoise * np.sin(elevationNoise)
        
        W = np.array([[x, 0, 0],
                      [0, y, 0],
                      [0, 0, z]])
    
        # Form Block Matrix Z
        Z = dt * np.block([ [-A, Gamma @ W @ Gamma.T], [np.zeros([6,6]), A.T]])
        
        # Compute Matrix Exponential
        vanLoan = expm(Z)

        # Extract Q = F.T * VanLoan[0:6, 6:12]
        F = vanLoan[6:12, 0:6].T

        Q = F @ vanLoan[0:6, 6:12]
        
        return Q
    
class indeptEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targets

    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates. the kalman estimate
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} 

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}

        self.neesHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.nisHist = {targetID: defaultdict(dict) for targetID in targetIDs}

# LOCAL EXTENDED KALMAN FILTER
    # Inputs: Satellite with a new bearings measurement, targetID, dt since last measurement, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def EKF(self, sat, measurement, target, envTime):
         
# Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [in_track, cross_track] bearings measurement

        targetID = target.targetID

# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
    # If no prior estimates, use the first measurement and assume no velocity
            # start with true position plus some noise
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0]) + np.random.normal(0, 2, 6) 
            # start with some covariance, about +- 5 km and +- 15 km/min, then plus some noise 
            P_prior = np.array([[5, 0, 0, 0, 0, 0],
                                [0, 20, 0, 0, 0, 0],
                                [0, 0, 5, 0, 0, 0],
                                [0, 0, 0, 20, 0, 0],
                                [0, 0, 0, 0, 5, 0],
                                [0, 0, 0, 0, 0, 20]]) + np.random.normal(0, 1, (6,6))*np.eye(6)
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
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
        # How does our state: [x, vx, y, vy, z, vz] change over time?
        F = np.array([[1, dt, 0, 0, 0, 0], # Assume no acceleration, just constant velocity over the time step
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])
        
    # Define the process noise matrix, Q.
        # Estimate the randomness of the acceleration
        # Use Van Loan's method to tune Q
        Q = self.calculate_Q(dt)

    # Define the sensor noise matrix, R.
        # This is the covariance estimate of the sensor error
        # just use the bearing error for now?
        R = np.eye(2)*sat.sensor.bearingsError**2
        # TODO: could start with tuning this

# EXTRACT THE MEASUREMENTS
        z = measurement # WILL BE BEARINGS ONLY MEASUREMENT

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior) # ECI coordinates
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# Use the predicted state to calculate H
    # Define the obversation matrix, H.
        # How does our state relate to our measurement?
        H = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred)

# UPDATE:
    # Calculate innovation terms:
        # y = z - h(x)
        innovation = z - sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]])) # Difference between the measurement and the predicted measurement
        # then use big H
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R

    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

    # Correct prediction
        est = est_pred + np.dot(K, innovation)
        P = P_pred - np.dot(K, np.dot(H, P_pred))

    # CALCUATE NEES AND NIS
        # For nees, we need to get the error compared to the truth data:
        # Get the true position
        true = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]])
        # Get the error
        error = est - true
        # Get the covariance of the error
        nees = np.dot(error.T, np.dot(np.linalg.inv(P), error))
        nis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = innovation
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis
                               

    def calculate_Q(self, dt, intensity=np.array([0.001, 5, 5])):
        # Use Van Loan's method to tune Q using the matrix exponential
        
        # Define the state transition matrix, A.
        A = np.array([[0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        # Assume there could be noise impacting the cartesian acceleration
        Gamma = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
        
        # Assing a maximum intensity of the noise --> 0.001 km/min^2 = 1 m/min^2 over the time step
        
        rangeNoise, elevationNoise, azimuthNoise = intensity
        x = rangeNoise * np.cos(elevationNoise) * np.cos(azimuthNoise)
        y = rangeNoise * np.cos(elevationNoise) * np.sin(azimuthNoise)
        z = rangeNoise * np.sin(elevationNoise)
        
        W = np.array([[x, 0, 0],
                      [0, y, 0],
                      [0, 0, z]])
    
        # Form Block Matrix Z
        Z = dt * np.block([ [-A, Gamma @ W @ Gamma.T], [np.zeros([6,6]), A.T]])
        
        # Compute Matrix Exponential
        vanLoan = expm(Z)

        # Extract Q = F.T * VanLoan[0:6, 6:12]
        F = vanLoan[6:12, 0:6].T

        Q = F @ vanLoan[0:6, 6:12]
        
        return Q

class ddfEstimator:
    def __init__(self, targetIDs):

    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates. the kalman estimate
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} 

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}

        self.neesHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.nisHist = {targetID: defaultdict(dict) for targetID in targetIDs}


    # Perform covariance intersection with all estimates and covariances sent from satellites
    # Inputs:
    #   - sat: satellite object
    #   - commNode: communication node object, containing potentially queued information
    #   - targetID: target ID
    #   - envTime: environment time
    def CI(self, sat, commNode, targetID):

        # Check if there is any information in the queue:
        if len(commNode['queued_data']) == 0 or targetID not in commNode['queued_data'] or len(commNode['queued_data'][targetID]) == 0:
            return


    ### There is information in the queue

        # If there is new information in the queue, we want to perform covariance intersection on all new time estimates and covariances:
        for sentTime in commNode['queued_data'][targetID].keys():

        # First check, do we have any prior estimate and covariance?
            # Should only ever happen once on first run through
            # If we dont, use the sent estimate and covariance to initalize
            if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
                self.estHist[targetID][sentTime] = commNode['queued_data'][targetID][sentTime]['est'][0]
                self.covarianceHist[targetID][sentTime] = commNode['queued_data'][targetID][sentTime]['cov'][0]

        # Now, we have a prior estimate and covariance, so we need to perform covariance intersection?

            priorTime = max(self.estHist[targetID].keys())
            estPrior = self.estHist[targetID][priorTime]
            covPrior = self.covarianceHist[targetID][priorTime]

        # Check, should we THROW OUT the prior? or do CI with it?
            # If the time b/w prior and new estimate is greater than 5 mins, we should throw out the prior
            if sentTime - priorTime > 5: 
                print(str(sat.name) + " is throwing out prior estimate and covariance from " + str(priorTime) + " for " + str(commNode['queued_data'][targetID][sentTime]['sender']) + "s new update at " + str(sentTime))
                estPrior = commNode['queued_data'][targetID][sentTime]['est'][0]
                covPrior = commNode['queued_data'][targetID][sentTime]['cov'][0]
            # If the time b/w prior and new estimate is less than 5 mins, keep the prior the same, then do CI

        # Now do CI on all new estimates and covariances taken at that time step
            for i in range(len(commNode['queued_data'][targetID][sentTime]['est'])):

                estSent = commNode['queued_data'][targetID][sentTime]['est'][i]
                covSent = commNode['queued_data'][targetID][sentTime]['cov'][i]

                # Minimize the covariance determinant
                omegaOpt = minimize(self.det_of_fused_covariance, [0.5], args=(covPrior, covSent), bounds=[(0, 1)]).x

                # Now compute the fused covariance
                cov1 = covPrior
                cov2 = covSent
                covPrior = np.linalg.inv(omegaOpt * np.linalg.inv(cov1) + (1 - omegaOpt) * np.linalg.inv(cov2))
                estPrior = covPrior @ (omegaOpt * np.linalg.inv(cov1) @ estPrior + (1 - omegaOpt) * np.linalg.inv(cov2) @ estSent)

        # Finally, save the fused estimate and covariance
            self.estHist[targetID][sentTime] = estPrior
            self.covarianceHist[targetID][sentTime] = covPrior

        # Clear the queued data
        commNode['queued_data'][targetID] = {}



    #         # If the time is not in the estimator history, add it:
    #         if sentTime not in self.estHist[targetID]:
    #             self.estHist[targetID][sentTime] = {}
    #             self.covarianceHist[targetID][sentTime] = {}

    #         # Perform covariance intersection on the new information:
    #         self.covariance_intersection(sat, commNode, targetID, sentTime)
        




    #     ciMaxTime = max(commNode['queued_data'][targetID].keys())

    # #### First check does the target have any prior estimates or covariances?
    #     if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
    #     #### If not, just use the send estimate and covariance to initalize filter
    #         self.estHist[targetID][ciMaxTime] = commNode['queued_data'][targetID][ciMaxTime]['est'][0]
    #         self.covarianceHist[targetID][ciMaxTime] = commNode['queued_data'][targetID][ciMaxTime]['cov'][0]

    #     else:
    # #### If the sat has a prior estimate and covariance, check, should we THORW OUT PRIOR OR CI PRIOR?
    #         estMaxTime = max(self.estHist[targetID].keys())


    # #### MERGE NEW ESTIMATE WITH OLD ESTIMATE, DOING CI




    #     if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
    #         covCur = commNode['queued_data'][targetID][ciMaxTime]['cov'][0]
    #         estCur = commNode['queued_data'][targetID][ciMaxTime]['est'][0]
    #     else:
    #         estMaxTime = max(self.estHist[targetID].keys())
    #         covCur = self.covarianceHist[targetID][estMaxTime] 
    #         estCur = self.estHist[targetID][estMaxTime]

    # #### THROW AWAY OLD ESTIMATE AND COVARIANCE, JUST USE NEW ESTIMATE AND COVARIANCE FROM CI

    #     covCur = self.covarianceHist[targetID][ciMaxTime] 
    #     estCur = self.estHist[targetID][ciMaxTime]

    #     # If no prior estimate, we can just use the first estimate and covariance
    #     if len(estCur) == 0 and len(covCur) == 0:
    #         covCur = commNode['queued_data'][targetID][ciMaxTime]['cov'][0]
    #         estCur = commNode['queued_data'][targetID][ciMaxTime]['est'][0]

    #     # If there is CI information through the queue, then fuse the information
    #     for i in range(len(commNode['queued_data'][targetID][ciMaxTime]['cov'])):

    #         # Extract the estimate and covariance
    #         est = commNode['queued_data'][targetID][ciMaxTime]['est'][i]
    #         cov = commNode['queued_data'][targetID][ciMaxTime]['cov'][i]

    #         # Minimize the covariance determinant
    #         omegaOpt = minimize(self.det_of_fused_covariance, [0.5], args=(covCur, cov), bounds=[(0, 1)]).x

    #         # Now compute the fused covariance
    #         cov1 = covCur
    #         cov2 = cov
    #         covCur = np.linalg.inv(omegaOpt * np.linalg.inv(cov1) + (1 - omegaOpt) * np.linalg.inv(cov2))
    #         estCur = covCur @ (omegaOpt * np.linalg.inv(cov1) @ estCur + (1 - omegaOpt) * np.linalg.inv(cov2) @ est)

    #     # estCur = covCur @ (omegaOpt * np.linalg.inv(covCur) @ estCur + (1 - omegaOpt) * np.linalg.inv(cov) @ est)

    #     # Save the fused estimate and covariance
    #     self.estHist[targetID][ciMaxTime] = estCur
    #     self.covarianceHist[targetID][ciMaxTime] = covCur

    #     # Clear the queued data
    #     commNode['queued_data'][targetID] = {}


    # LOCAL EXTENDED KALMAN FILTER
    # Inputs: Satellite with a new bearings measurement, targetID, dt since last measurement, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def local_EKF(self, sat, measurement, target, envTime):
         
# Desired estimate: Xdot = [x, vx, y, vy, z, vz]
    # Measurment: Z = [in_track, cross_track] bearings measurement

        targetID = target.targetID

# GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
    # If no prior estimates, use the first measurement and assume no velocity
            # start with true position plus some noise
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0]) + np.random.normal(0, 2, 6) 
            # start with some covariance, about +- 5 km and +- 15 km/min, then plus some noise 
            P_prior = np.array([[5, 0, 0, 0, 0, 0],
                                [0, 20, 0, 0, 0, 0],
                                [0, 0, 5, 0, 0, 0],
                                [0, 0, 0, 20, 0, 0],
                                [0, 0, 0, 0, 5, 0],
                                [0, 0, 0, 0, 0, 20]]) + np.random.normal(0, 1, (6,6))*np.eye(6)
            
        # Store these and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
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
        # How does our state: [x, vx, y, vy, z, vz] change over time?
        F = np.array([[1, dt, 0, 0, 0, 0], # Assume no acceleration, just constant velocity over the time step
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])
        
    # Define the process noise matrix, Q.
        # Estimate the randomness of the acceleration
        # Use Van Loan's method to tune Q
        Q = self.calculate_Q(dt)

    # Define the sensor noise matrix, R.
        # This is the covariance estimate of the sensor error
        # just use the bearing error for now?
        R = np.eye(2)*sat.sensor.bearingsError**2

# EXTRACT THE MEASUREMENTS
        z = measurement # WILL BE BEARINGS ONLY MEASUREMENT

# PREDICTION:
    # Predict the state and covariance
        est_pred = np.dot(F, est_prior) # ECI coordinates
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q

# Use the predicted state to calculate H
    # Define the obversation matrix, H.
        # How does our state relate to our measurement?
        H = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred)

# UPDATE:
    # Calculate innovation terms:
        # y = z - h(x)
        innovation = z - sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]])) # Difference between the measurement and the predicted measurement
        # then use big H
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R

    # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))

    # Correct prediction
        est = est_pred + np.dot(K, innovation)
        P = P_pred - np.dot(K, np.dot(H, P_pred))

    # CALCUATE NEES AND NIS
        # For nees, we need to get the error compared to the truth data:
        # Get the true position
        true = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]])
        # Get the error
        error = est - true
        # Get the covariance of the error
        nees = np.dot(error.T, np.dot(np.linalg.inv(P), error))
        nis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))

# SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = innovation
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis
                               
    def calculate_Q(self, dt, intensity=np.array([0.001, 5, 5])):
        # Use Van Loan's method to tune Q using the matrix exponential
        
        # Define the state transition matrix, A.
        A = np.array([[0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        # Assume there could be noise impacting the cartesian acceleration
        Gamma = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
        
        # Assing a maximum intensity of the noise --> 0.001 km/min^2 = 1 m/min^2 over the time step
        
        rangeNoise, elevationNoise, azimuthNoise = intensity
        x = rangeNoise * np.cos(elevationNoise) * np.cos(azimuthNoise)
        y = rangeNoise * np.cos(elevationNoise) * np.sin(azimuthNoise)
        z = rangeNoise * np.sin(elevationNoise)
        
        W = np.array([[x, 0, 0],
                      [0, y, 0],
                      [0, 0, z]])
    
        # Form Block Matrix Z
        Z = dt * np.block([ [-A, Gamma @ W @ Gamma.T], [np.zeros([6,6]), A.T]])
        
        # Compute Matrix Exponential
        vanLoan = expm(Z)

        # Extract Q = F.T * VanLoan[0:6, 6:12]
        F = vanLoan[6:12, 0:6].T

        Q = F @ vanLoan[0:6, 6:12]
        
        return Q
    
    def det_of_fused_covariance(self, omega, cov1, cov2):
        # Calculate the determinant of the fused covariance matrix
        # omega is the weight of the first covariance matrix
        # cov1 and cov2 are the covariance matrices of the two estimates
        omega = omega[0]
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)