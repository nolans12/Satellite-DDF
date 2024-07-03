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

    # Extended Kalman Filter for both local and central estimation
    def EKF(self, sats, measurements, target, envTime):
        # This Estimator can handle both the central and local estimation
        # Sats is a list of all satellites but it can also be a single sat for local estimation
        # Measurements is a list of all measurements but it can also be a single measurement for local estimation
        
        self.gottenEstimate = True
        numMeasurements = 2*len(measurements)
        
        # First get the measurements from the satellites at given time and targetID
        targetID = target.targetID
        
        ## For this target: Get the prior Data
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0: # If no prior estimate exists, just use true position plus noise
            # start with true position and velocity plus some noise
            #prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 1, 3)
            #prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 1, 3)
            
            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + 15
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + 1.5
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
                                 
            # start with some covariance, about +- 50 km and +- 100 km/min to make sure the covariance converges
            P_prior = np.array([[50, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 50, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 50, 0],
                                [0, 0, 0, 0, 0, 100]])
                
            # Store these and return for first iteration to intialize the filter consistently
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.innovationHist[targetID][envTime] = np.zeros(2)
            self.innovationCovHist[targetID][envTime] = np.eye(2)
            return est_prior
       
        else:
        # Else, get most recent estimate and covariance
            time_prior = max(self.estHist[targetID].keys())
            est_prior = self.estHist[targetID][time_prior]
            P_prior = self.covarianceHist[targetID][time_prior]

        ### Note that est_prior is a 1x6 array, and P_prior is a 6x6 array
        
        # Now to get dt, use time since last estimate for prediction step
        dt = envTime - time_prior

        # Predict the next state using state transition function
        est_pred = self.state_transition(est_prior, dt)
        
        # Evaluate the Jacobian of the state transition function
        F = self.state_transition_jacobian(est_prior, dt)
        
        # Predict the prcoess noise assosiated with the state transition
        Q = np.zeros((6,6)) 
        
        # Predict the covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q
        
        # Now to get the measurements and update the estimate --> numMeasurements = 2*len(measurements) [alpha, beta]
        z = np.zeros((numMeasurements, 1)) # 2Nx1 stacked vector of measurements
        H = np.zeros((numMeasurements, 6)) # 2Nx6 Jacobian of the sensor model
        R = np.zeros((numMeasurements, numMeasurements)) # NxN Sensor noise matrix
        innovation = np.zeros((numMeasurements,1))
        
        i = 0
        for sat in sats: # for each satellite, get the measurement and update the H, R, and innovation
            # Stack the measurements into a 2Nx1 vector
            z[2*i:2*i+2] = np.reshape(measurements[i][:],(2,1)) # Measurement stack
            
            # Compute the Jacobian Evaluated at the predicted state
            H[2*i:2*i+2,0:6] = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred) # Jacobian of the sensor model
            
            # Compute the block diagonal sensor noise matrix
            R[2*i:2*i+2,2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2 # Sensor noise matrix
            
            # Determine the innovation
            innovation[2*i:2*i+2] = z[2*i:2*i+2] - np.reshape(np.array(sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]]))),(2,1)) # 2N x 1 vector of innovations
            
            i += 1
                        
        ### Extract Measurements and Calcualate Jacobian, Sensor Noise, and Innovation        
        # z = np.zeros((numMeasurements, 2))
        # H = np.zeros((2*numMeasurements, 6))
        # R = np.zeros((2*numMeasurements, 2*numMeasurements))
        # innovation = np.zeros((2*numMeasurements))
        
        # i = 0
        # for sat in sats: # for each satellite, get the measurement and update the H, R, and innovation
        #     z[i] = measurements[i] # get 1x2 measurement vector
        #     H[2*i:2*i+2,0:6] = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred) # Jacobian of the sensor model
        #     R[2*i:2*i+2,2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2 # Sensor noise matrix
        #     innovation[2*i:2*i+2] = (z[i] - sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]]))).flatten() # 1 x 2N vector of innovations
            
        #     i += 1
                        
        # Calculate the innovation covariance
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R
        
        # Solve for the Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))
        
        # Correct the prediction
        est = est_pred + np.reshape(np.dot(K, innovation),(6)) # Note that estimates storedis a 1x6 array, so we need to transpose result
        
        # Correct the Covariance
        P = P_pred - np.dot(K, np.dot(H, P_pred))
        
        # CALCUATE NEES AND NIS
        # Get the true position
        true = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]])
        
        # Get the error
        error = est - true
        
        # Get the covariance of the error and innovation
        nees = np.dot(error.T, np.dot(np.linalg.inv(P), error))
        nis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))

        # SAVE THE DATA
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = np.reshape(innovation,(numMeasurements))
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis[0][0]
    
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

### Central Estimator Class
class centralEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        return super().EKF(sats, measurements, target, envTime)

### Independent Estimator Class
class indeptEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        return super().EKF(sats, measurements, target, envTime)

### DDF Estimator Class
class ddfEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        return super().EKF(sats, measurements, target, envTime)


    def CI(self, sat, commNode):

        # Check if there is any information in the queue:
        if len(commNode['queued_data']) == 0: 
            return
        
        # There is information in the queue, get the newest info
        timeSent = max(commNode['queued_data'].keys())

        # Check all the targets in the queue
        for targetID in commNode['queued_data'][timeSent].keys():

            # For each target, loop through all the estimates and covariances
            for i in range(len(commNode['queued_data'][timeSent][targetID]['est'])):
                
                estSent = commNode['queued_data'][timeSent][targetID]['est'][i]
                covSent = commNode['queued_data'][timeSent][targetID]['cov'][i]

                # Check, does satellite have an estimate and covariance for this target already?
                if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
                    # If not, use the sent estimate and covariance to initialize
                    self.estHist[targetID][timeSent] = estSent
                    self.covarianceHist[targetID][timeSent] = covSent
                    continue

                # If the satellite does have an estimate and covariance for this target already, check if we should CI
                timePrior = max(self.estHist[targetID].keys())

                # If the send time is older than the prior estimate, throw out the sent estimate
                if timeSent < timePrior:
                    continue

                # If the time between the sent estimate and the prior estimate is greater than 5 minutes, throw out the prior
                if timeSent - timePrior > 5:
                    self.estHist[targetID][timeSent] = estSent
                    self.covarianceHist[targetID][timeSent] = covSent
                    continue

                # Else, lets do CI
                estPrior = self.estHist[targetID][timePrior]
                covPrior = self.covarianceHist[targetID][timePrior]

                # Now propegate the prior estimate and cov to the new time
                dt = timeSent - timePrior
                estPrior = self.state_transition(estPrior, dt)
                F = self.state_transition_jacobian(estPrior, dt)
                covPrior = np.dot(F, np.dot(covPrior, F.T))

                # Minimize the covariance determinant
                omegaOpt = minimize(self.det_of_fused_covariance, [0.5], args=(covPrior, covSent), bounds=[(0, 1)]).x

                # Now compute the fused covariance
                cov1 = covPrior
                cov2 = covSent

                covPrior = np.linalg.inv(omegaOpt * np.linalg.inv(cov1) + (1 - omegaOpt) * np.linalg.inv(cov2))
                estPrior = covPrior @ (omegaOpt * np.linalg.inv(cov1) @ estPrior + (1 - omegaOpt) * np.linalg.inv(cov2) @ estSent)

                # Save the fused estimate and covariance
                self.estHist[targetID][timeSent] = estPrior
                self.covarianceHist[targetID][timeSent] = covPrior


    def det_of_fused_covariance(self, omega, cov1, cov2):
        # Calculate the determinant of the fused covariance matrix
        # omega is the weight of the first covariance matrix
        # cov1 and cov2 are the covariance matrices of the two estimates
        omega = omega[0]
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)

    def CI_old(self, sat, commNode, targetID):

        # Check if there is any information in the queue:
        if len(commNode['queued_data']) == 0 or targetID not in commNode['queued_data'] or len(commNode['queued_data'][targetID]) == 0:
            # print("No information in the queue for " + str(sat.name) + " for target " + str(targetID))
            return

        #print("Information in the queue for " + str(sat.name) + " for target " + str(targetID))
        # # print out the time for hte information:
        #for sentTime in commNode['queued_data'][targetID].keys():
            #print("Sent time: " + str(sentTime))
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
            covPrior_prop = np.dot(F, np.dot(covPrior, F.T))

            # print("Propegated prior estimate to new time: " + str(sentTime) + " for " + str(sat.name) + " from " + str(priorTime) + " to " + str(sentTime) + " : " + str(estPrior) + " to " + str(estPrior_prop))

        # Check, should we THROW OUT the prior? or do CI with it?
            # If the time b/w prior and new estimate is greater than 5 mins, we should throw out the prior
            if dt > 5 or self.gottenEstimate == False: 
                # print(str(sat.name) + " is throwing out prior estimate and covariance from " + str(priorTime) + " for " + str(commNode['queued_data'][targetID][sentTime]['sender']) + "s new update at " + str(sentTime))
                estPrior_prop = commNode['queued_data'][targetID][sentTime]['est'][0]
                covPrior_prop = commNode['queued_data'][targetID][sentTime]['cov'][0]
            # If the time b/w prior and new estimate is less than 5 mins, keep the prior the same, then do CI

        # Now do CI on all new estimates and covariances taken at that time step
            for i in range(len(commNode['queued_data'][targetID][sentTime]['est'])):

                estSent = commNode['queued_data'][targetID][sentTime]['est'][i]
                covSent = commNode['queued_data'][targetID][sentTime]['cov'][i]

                # Minimize the covariance determinant
                omegaOpt = minimize(self.det_of_fused_covariance, [0.5], args=(covPrior_prop, covSent), bounds=[(0, 1)]).x

                # Now compute the fused covariance
                cov1 = covPrior_prop
                cov2 = covSent
                
                covPrior_prop = np.linalg.inv(omegaOpt * np.linalg.inv(cov1) + (1 - omegaOpt) * np.linalg.inv(cov2))
                estPrior_prop = covPrior_prop @ (omegaOpt * np.linalg.inv(cov1) @ estPrior_prop + (1 - omegaOpt) * np.linalg.inv(cov2) @ estSent)

        # Finally, save the fused estimate and covariance
            # print("Fused estimate and covariance for " + str(sat.name) + " for target " + str(targetID) + " at time " + str(sentTime))# + " : " + str(estPrior_prop) + " and " + str(covPrior_prop))
            self.estHist[targetID][sentTime] = estPrior_prop
            self.covarianceHist[targetID][sentTime] = covPrior_prop

        # Clear the queued data
        commNode['queued_data'][targetID] = {}
