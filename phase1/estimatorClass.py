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
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) * 1.5
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
                                 
            # start with some covariance, about +- 25 km and +- 10 km/min to make sure the covariance converges
            P_prior = np.array([[625, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 625, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 625, 0],
                                [0, 0, 0, 0, 0, 100]])
                
            # Store these and return for first iteration to intialize the filter consistently
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.innovationHist[targetID][envTime] = np.zeros(2)
            self.innovationCovHist[targetID][envTime] = np.eye(2)
            self.nisHist[targetID][envTime] = 0
            self.neesHist[targetID][envTime] = 0
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
        #est_pred2 = self.state_transition(est_prior, dt)
        est_pred = self.state_transition(est_prior, dt)
        
        # Evaluate the Jacobian of the state transition function
        F = self.state_transition_jacobian(est_prior, dt)
        
        # Predict the prcoess noise assosiated with the state transition
        Q = np.zeros((6,6))
        Q = np.diag([50, 1, 50, 1, 50, 1]) # Process noise matrix
        
        # Q_itr 1
        # Q = np.array([[50, 1, 0, 0, 0, 0],
        #               [0, 1, 0, 0, 0, 0], 
        #               [0, 0, 50, 1, 0, 0],
        #               [0, 0, 0, 1, 0, 0],
        #               [0, 0, 0, 0, 50, 1],
        #               [0, 0, 0, 0, 0, 1]])
        
        # Q_itr 2
        # Q = np.array([[50, 0, 50, 0, 0, 0],
        #               [0, 1, 0, 0, 0, 0], 
        #               [50, 0, 50, 0, 0, 0],
        #               [0, 0, 0, 1, 0, 0],
        #               [0, 0, 0, 0, 50, 1],
        #               [0, 0, 0, 0, 0, 1]])
        
        # Q_itr 3
        # Q = np.array([[100, 0.5, 100, 0.5, 100, 0.5],
        #               [0, 1, 0, 0, 0, 0], 
        #               [100, 0.5, 100, 0.5, 100, 0.5],
        #               [0, 0, 0, 1, 0, 0],
        #               [100, 0.5, 100, 0.5, 100, 0.5],
        #               [0, 0, 0, 0, 0, 1]])
        
        # Q_itr 4
        # Q = np.array([[1000, 0.5, 1000, 0.5, 100, 0.5],
        #               [0, 1, 0, 0, 0, 0], 
        #               [1000, 0.5, 1000, 0.5, 100, 0.5],
        #               [0, 0, 0, 1, 0, 0],
        #               [100, 0.5, 100, 0.5, 100, 0.5],
        #               [0, 0, 0, 0, 0, 1]])
        
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
            #print("Real Measurement: ", z[2*i:2*i+2].flatten())
            
            # Compute the Jacobian Evaluated at the predicted state
            H[2*i:2*i+2,0:6] = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred) # Jacobian of the sensor model
            #print(f"Jacobian: {H[2*i:2*i+2,0:6]}")
            
            # Compute the block diagonal sensor noise matrix
            R[2*i:2*i+2,2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2 * 1000 # Sensor noise matrix with larger magnitude to improve stability
            
            # Predict the measurement
            z_pred = np.array(sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]]))) # 2x1 vector of predicted measurements
            
            #print("Predicted Measurement: ", z_pred)
            # Determine the innovation
            innovation[2*i:2*i+2] = z[2*i:2*i+2] - np.reshape(z_pred,(2,1)) # 2N x 1 vector of innovations
            
            i += 1
                        
        # Calculate the innovation covariance
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R
        
        #print("Innovation Covariance: ", innovationCov)
        
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
        
        # Print Self, Sats, Target, Prediction1, Prediction 2, True, Error in both
        
        # Set the desired precision
        #precision = 2

        # Print statements
        # print("Sat: ", {sat.name for sat in sats}, "Self: ", {self}, "Time: ", {envTime})
        
        # # Prior Error, Prediction Error, and True Error
        # priorError = np.linalg.norm(est_prior - true)
        # predError = np.linalg.norm(est_pred - true)
        # postError = np.linalg.norm(est - true)
        
        # print(f"{'Prior Error:':<20} {'Predicted Error:':<20} {'Actual Error:'}")
        # print(f"{priorError:<20.{precision}f} {predError:<20.{precision}f} {postError:<20.{precision}f}")
        
        # print(f"{'Predicted Error:':<20} {'Actual Error:'}")
    # Print the left norm of the error
        # print(f"{'Predicted Error:':<20} {'Actual Error:'}")
        # print(f"{np.linalg.norm(est_pred - true):<20.{precision}f} {np.linalg.norm(est - true):<20.{precision}f}")
    
        # # Using format specifier to control the precision
        # print(f"{'Vx =':<5} {est_pred[1]-true[1]:<15.{precision}f} {'Vx =':<5} {est[1]-true[1]:.{precision}f}")
        # print(f"{'Vy =':<5} {est_pred[3]-true[3]:<15.{precision}f} {'Vy =':<5} {est[3]-true[3]:.{precision}f}")
        # print(f"{'Vz =':<5} {est_pred[5]-true[5]:<15.{precision}f} {'Vz =':<5} {est[5]-true[5]:.{precision}f}")
        # print(f"{'x  =':<5} {est_pred[0]-true[0]:<15.{precision}f} {'x  =':<5} {est[0]-true[0]:.{precision}f}")
        # print(f"{'y  =':<5} {est_pred[2]-true[2]:<15.{precision}f} {'y  =':<5} {est[2]-true[2]:.{precision}f}")
        # print(f"{'z  =':<5} {est_pred[4]-true[4]:<15.{precision}f} {'z  =':<5} {est[4]-true[4]:.{precision}f}")
        
        # print(f"{'True:':<20}")
        # print(f"{'Vx =':<5} {true[1]:<15.{precision}f}") 
        # print(f"{'Vy =':<5} {true[3]:<15.{precision}f}")
        # print(f"{'Vz =':<5} {true[5]:<15.{precision}f}") 
        # print(f"{'x  =':<5} {true[0]:<15.{precision}f}") 
        # print(f"{'y  =':<5} {true[2]:<15.{precision}f}") 
        # print(f"{'z  =':<5} {true[4]:<15.{precision}f}") 
        # print('-'*50)
       
    def state_transition(self, estPrior, dt):
        # State Transition Function inputs current state and time step, returns next state
        # Input Current ECI Position and Velocity
        x, vx, y, vy, z, vz = estPrior
            
        # Convert to Spherical Coordinates
        range = jnp.sqrt(x**2 + y**2 + z**2)
        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x)
        
        rangeRate = (x * vx + y * vy + z * vz) / range
    
        # Calculate Elevation Rate
        elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))
    
        # Calculate Azimuth Rate
        azimuthRate = (x * vy - y * vx) / (x**2 + y**2)
        
        # Previous Spherical State
        prev_spherical_state = jnp.array([range, rangeRate, elevation, elevationRate, azimuth, azimuthRate])
        
        def derivatives(spherical_state):
            # Let the State be r, rdot, elevation, elevation rate, azimuth, azimuth rate
            r, rdot, elevation, elevationRate, azimuth, azimuthRate = spherical_state
            
            rangeRate = rdot
            rangeAccel = 0
            elevationRate = elevationRate
            elevationAccel = 0
            azimuthRate = azimuthRate
            azimuthAccel = 0
            
            return jnp.array([rangeRate, rangeAccel, elevationRate, elevationAccel, azimuthRate, azimuthAccel])
        
        k1 = derivatives(prev_spherical_state)
        k2 = derivatives(prev_spherical_state + 0.5 * dt * k1)
        k3 = derivatives(prev_spherical_state + 0.5 * dt * k2)
        k4 = derivatives(prev_spherical_state + dt * k3)
        
        next_spherical_state = prev_spherical_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        range = next_spherical_state[0]
        rangeRate = next_spherical_state[1]
        elevation = next_spherical_state[2]
        elevationRate = next_spherical_state[3]
        azimuth = next_spherical_state[4]
        azimuthRate = next_spherical_state[5]
        
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

    def state_transition_RK_O(self, estPrior, dt):
        def derivatives(state):
            x, vx, y, vy, z, vz = state
            
            # Turn into Spherical Coordinates
            range = jnp.sqrt(x**2 + y**2 + z**2)
            elevation = jnp.arcsin(z / range)
            azimuth = jnp.arctan2(y, x)
            
            rangeRate = (x * vx + y * vy + z * vz) / range
            elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))
            azimuthRate = (x * vy - y * vx) / (x**2 + y**2)

            # Convert rates back to Cartesian coordinates
            cos_elevation = jnp.cos(elevation)
            sin_elevation = jnp.sin(elevation)
            cos_azimuth = jnp.cos(azimuth)
            sin_azimuth = jnp.sin(azimuth)

            dx = rangeRate * cos_elevation * cos_azimuth \
                - range * elevationRate * sin_elevation * cos_azimuth \
                - range * azimuthRate * cos_elevation * sin_azimuth

            dy = rangeRate * cos_elevation * sin_azimuth \
                - range * elevationRate * sin_elevation * sin_azimuth \
                + range * azimuthRate * cos_elevation * cos_azimuth

            dz = rangeRate * sin_elevation + range * elevationRate * cos_elevation
            
            dvx = 0
            dvy = 0
            dvz = 0

            return jnp.array([dx, dvx, dy, dvy, dz, dvz])

        # RK4 Integration
        k1 = dt * derivatives(estPrior)
        k2 = dt * derivatives(estPrior + 0.5 * dt * k1)
        k3 = dt * derivatives(estPrior + 0.5 * dt * k2)
        k4 = dt * derivatives(estPrior + dt * k3)

        newState = estPrior + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return newState    
    def state_transition_Orig(self, estPrior, dt):
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
        # print('*'*50)
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