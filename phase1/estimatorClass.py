from import_libraries import *

class BaseEstimator:
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

        self.gottenEstimate = False

    # LOCAL EXTENDED KALMAN FILTER
    # Inputs: Satellite with a new bearings measurement, targetID, dt since last measurement, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def local_EKF(self, sat, measurement, target, envTime):
         
        self.gottenEstimate = True

        # Desired estimate: Xdot = [x, vx, y, vy, z, vz]
        # Measurment: Z = [in_track, cross_track] bearings measurement

        targetID = target.targetID

        # GET THE PRIOR DATA
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use the measurement
            # TODO: 
            # We want our initial guess to be based on the initial guess of the target
            # Although the target may not actually have this state, it is sampled from a distrbition
            # est_prior = target.propagate(envTime, envTime, target.initialGuess)
            # P_prior = np.array([[50, 0, 0, 0, 0, 0],
            #                     [0, 100, 0, 0, 0, 0],
            #                     [0, 0, 50, 0, 0, 0],
            #                     [0, 0, 0, 100, 0, 0],
            #                     [0, 0, 0, 0, 50, 0],
            #                     [0, 0, 0, 0, 0, 100]])
            # est_prior = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]]) +  np.random.normal(0, 1, 6)
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0])
            P_prior = np.array([[50, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 50, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 50, 0],
                                [0, 0, 0, 0, 0, 100]])

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
        
        # Define the process noise matrix, Q.
        # Estimate the randomness of the acceleration
        # Use Van Loan's method to tune Q
        # Q = self.calculate_Q(dt)

        # make Q be all zeros:
        Q = np.zeros((6,6))


        # Define the sensor noise matrix, R.
        # This is the covariance estimate of the sensor error
        # just use the bearing error for now?
        R = np.eye(2)*sat.sensor.bearingsError**2

        # EXTRACT THE MEASUREMENTS
        z = measurement # WILL BE BEARINGS ONLY MEASUREMENT

        # PREDICTION:
        # Predict the state and covariance
        # est_pred = np.dot(F, est_prior) # ECI coordinates
        est_pred = self.state_transition(est_prior, dt) # Spherical coordinates
        F = self.state_transition_jacobian(est_prior, dt) # Jacobian of the state transition
        
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
    
    # VAN LOANS METHOD FOR Q
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
    
    def state_transition(self, estPrior, dt):
    
        # Takes in previous ECI State and returns the next state after dt
        x = estPrior[0]
        vx = estPrior[1]
        y = estPrior[2]
        vy = estPrior[3]
        z = estPrior[4]
        vz = estPrior[5]
        
        # Turn into Spherical Coordinates
        range = jnp.sqrt(x**2 + y**2 + z**2)
        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x)
        
        rangeRate = (x * vx + y * vy + z * vz) / (range)

        # Calculate elevation rate
        elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))

        # Calculate azimuth rate
        azimuthRate = (x * vy - y * vx) / (x**2 + y**2)

        # Print intermediate values (comment out if not needed in production)
        # jax.debug.print(
        #     "Predic: Range: {range}, Range Rate: {rangeRate}, Elevation: {elevation}, Elevation Rate: {elevationRate}, Azimuth: {azimuth}, Azimuth Rate: {azimuthRate}",
        #     range=range, rangeRate=rangeRate, elevation=elevation, elevationRate=elevationRate, azimuth=azimuth, azimuthRate=azimuthRate)
        
        # Propagate the State
        range = range + rangeRate * dt
        elevation = elevation + elevationRate * dt
        azimuth = azimuth + azimuthRate * dt
        
        # Convert back to Cartesian
        x = range * jnp.cos(elevation) * jnp.cos(azimuth)
        y = range * jnp.cos(elevation) * jnp.sin(azimuth)
        z = range * jnp.sin(elevation)
        
        # Approximate velocities conversion (simplified version)
        vx = rangeRate * jnp.cos(elevation) * jnp.cos(azimuth) - \
            range * elevationRate * jnp.sin(elevation) * jnp.cos(azimuth) - \
            range * azimuthRate * jnp.cos(elevation) * jnp.sin(azimuth)

        vy = rangeRate * jnp.cos(elevation) * jnp.sin(azimuth) - \
            range * elevationRate * jnp.sin(elevation) * jnp.sin(azimuth) + \
            range * azimuthRate * jnp.cos(elevation) * jnp.cos(azimuth)

        vz = rangeRate * jnp.sin(elevation) + \
            range * elevationRate * jnp.cos(elevation)

        return jnp.array([x, vx, y, vy, z, vz])

    def state_transition_jacobian(self, estPrior, dt):
        jacobian = jax.jacfwd(lambda x: self.state_transition(x, dt))(jnp.array(estPrior))
        
        return jacobian

class indeptEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)
        # Add any additional initialization specific to indeptEstimator here

    def EKF(self, sat, measurement, target, envTime):
        return self.local_EKF(sat, measurement, target, envTime)

class ddfEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)
        # Add any additional initialization specific to ddfEstimator here

    def EKF(self, sat, measurement, target, envTime):
        return self.local_EKF(sat, measurement, target, envTime)

    # Perform covariance intersection with all estimates and covariances sent from satellites
    # Inputs:
    #   - sat: satellite object
    #   - commNode: communication node object, containing potentially queued information
    #   - targetID: target ID
    #   - envTime: environment time
    def CI(self, sat, commNode, targetID):

        # Check if there is any information in the queue:
        if len(commNode['queued_data']) == 0 or targetID not in commNode['queued_data'] or len(commNode['queued_data'][targetID]) == 0:
            # print("No information in the queue for " + str(sat.name) + " for target " + str(targetID))
            return

        # print("Information in the queue for " + str(sat.name) + " for target " + str(targetID))
        # # print out the time for hte information:
        # for sentTime in commNode['queued_data'][targetID].keys():
            # print("Sent time: " + str(sentTime))
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

            # Propegate the estPrior to the new time?
            dt = sentTime - priorTime

            estPrior_prop = self.state_transition(estPrior, dt)
            F = self.state_transition_jacobian(estPrior, dt)
        
        # Check, should we THROW OUT the prior? or do CI with it?
            # If the time b/w prior and new estimate is greater than 5 mins, we should throw out the prior
            if dt > 5 or self.gottenEstimate == False: 
                # print(str(sat.name) + " is throwing out prior estimate and covariance from " + str(priorTime) + " for " + str(commNode['queued_data'][targetID][sentTime]['sender']) + "s new update at " + str(sentTime))
                estPrior_prop = commNode['queued_data'][targetID][sentTime]['est'][0]
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
                estPrior_prop = covPrior @ (omegaOpt * np.linalg.inv(cov1) @ estPrior_prop + (1 - omegaOpt) * np.linalg.inv(cov2) @ estSent)

        # Finally, save the fused estimate and covariance
            self.estHist[targetID][sentTime] = estPrior_prop
            self.covarianceHist[targetID][sentTime] = covPrior

        # Clear the queued data
        commNode['queued_data'][targetID] = {}

    def det_of_fused_covariance(self, omega, cov1, cov2):
        # Calculate the determinant of the fused covariance matrix
        # omega is the weight of the first covariance matrix
        # cov1 and cov2 are the covariance matrices of the two estimates
        omega = omega[0]
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)


class centralEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)
        # Add any additional initialization specific to centralEstimator here

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
        
            # est_prior = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]]) +  np.random.normal(0, 1, 6) 
            est_prior = np.array([target.pos[0], 0, target.pos[1], 0, target.pos[2], 0])
            # start with some covariance, about +- 5 km and +- 15 km/min, then plus some noise 
            P_prior = np.array([[50, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 50, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 50, 0],
                                [0, 0, 0, 0, 0, 100]])# + np.random.normal(0, 1, (6,6))*np.eye(6)
            
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
        
        # F = np.array([[1, dt, 0, 0, 0, 0], # Assume no acceleration, just constant velocity over the time step
        #               [0, 1, 0, 0, 0, 0],
        #               [0, 0, 1, dt, 0, 0],
        #               [0, 0, 0, 1, 0, 0],
        #               [0, 0, 0, 0, 1, dt],
        #               [0, 0, 0, 0, 0, 1]])

        # Define the process noise matrix, Q.
        # Is a 6x6 matrix representing the covariance of the process noise
        # Estimate the randomness of the acceleration
        # Q = self.calculate_Q(dt)

        # all zeros for Q
        Q = np.zeros((6,6))

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
        # est_pred = np.dot(F, est_prior)
        est_pred = self.state_transition(est_prior, dt) # Spherical coordinates
        F = self.state_transition_jacobian(est_prior, dt) # Jacobian of the state transition
        
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
        
        # formatted_corrected_state = [f"{value:.2f}" for value in est]
        # print(f"Corrected state: {formatted_corrected_state}")
        # print(f"Actual state: {formatted_actual_state}")

        #print(*["-"*10]*6, sep="\n")

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
