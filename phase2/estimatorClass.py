from import_libraries import *

class BaseEstimator:
    def __init__(self, targetIDs):
        """
        Initialize the BaseEstimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        # Define the targets to track
        self.targs = targetIDs

        # Define history vectors for each extended Kalman filter
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of Kalman estimates in ECI coordinates
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of covariance matrices

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of innovations
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of innovation covariances

        self.neesHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of NEES (Normalized Estimation Error Squared)
        self.nisHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of NIS (Normalized Innovation Squared)
        
        self.trackQualityHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of track quality metric

        self.gottenEstimate = False  # Flag indicating if an estimate has been obtained

    def EKF(self, sats, measurements, target, envTime):
        """
        Extended Kalman Filter for both local and central estimation.

        Args:
        - sats: List of satellites or a single satellite object.
        - measurements: List of measurements or a single measurement.
        - target: Target object containing target information.
        - envTime: Current environment time.

        Returns updated dictionaries containing the following information:
        - est: Updated estimate for the target ECI state = [x vx y vy z vz].
        - cov: Updated covariance matrix for the target ECI state.
        - innovation: Innovation vector.
        - innovationCov: Innovation covariance matrix.
        - nees: Normalized Estimation Error Squared.
        - nis: Normalized Innovation Squared.
        """
        # Assume that the measurements are in the form of [alpha, beta] for each satellite
        numMeasurements = 2 * len(measurements)
        
        # Get the target ID
        targetID = target.targetID
        
        # Get prior data for the target
        if len(self.estHist[targetID]) == 0 and len(self.covarianceHist[targetID]) == 0:
            # If no prior estimate exists, initialize with true position plus noise
            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + 15
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) * 1.5
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
            
            # Initial covariance matrix
            P_prior = np.array([[625, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 625, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 625, 0],
                                [0, 0, 0, 0, 0, 100]])
                
            # Store initial values and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.innovationHist[targetID][envTime] = np.zeros(2)
            self.innovationCovHist[targetID][envTime] = np.eye(2)
            self.nisHist[targetID][envTime] = 0
            self.neesHist[targetID][envTime] = 0
            self.trackQualityHist[targetID][envTime] = 0
            return est_prior
       
        else:
            # Get most recent estimate and covariance
            time_prior = max(self.estHist[targetID].keys())
            est_prior = self.estHist[targetID][time_prior]
            P_prior = self.covarianceHist[targetID][time_prior]

        # Calculate time difference since last estimate
        dt = envTime - time_prior

        # Predict next state using state transition function
        est_pred = self.state_transition(est_prior, dt)
        
        # Evaluate Jacobian of state transition function
        F = self.state_transition_jacobian(est_prior, dt)
        
        # Predict process noise associated with state transition
        Q = np.diag([50, 1, 50, 1, 50, 1])  # Process noise matrix
        
        # Predict covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q
        
        # Prepare for measurements and update the estimate
        z = np.zeros((numMeasurements, 1))  # Stacked vector of measurements
        H = np.zeros((numMeasurements, 6))  # Jacobian of the sensor model
        R = np.zeros((numMeasurements, numMeasurements))  # Sensor noise matrix
        innovation = np.zeros((numMeasurements, 1))
        
        # Iterate over satellites to get measurements and update matrices
        for i, sat in enumerate(sats):
            z[2*i:2*i+2] = np.reshape(measurements[i][:], (2, 1))  # Measurement stack
            H[2*i:2*i+2, 0:6] = sat.sensor.jacobian_ECI_to_bearings(sat, est_pred)  # Jacobian of the sensor model
            R[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2 * 1000  # Sensor noise matrix scaled by 1000x
            
            z_pred = np.array(sat.sensor.convert_to_bearings(sat, np.array([est_pred[0], est_pred[2], est_pred[4]])))  # Predicted measurements
            innovation[2*i:2*i+2] = z[2*i:2*i+2] - np.reshape(z_pred, (2, 1))  # Innovations

        # Calculate innovation covariance
        innovationCov = np.dot(H, np.dot(P_pred, H.T)) + R
                
        # Solve for Kalman gain
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(innovationCov)))
        
        # Correct prediction
        est = est_pred + np.reshape(np.dot(K, innovation), (6))
        
        # Correct covariance
        P = P_pred - np.dot(K, np.dot(H, P_pred))
        
        # Calculate NEES and NIS
        true = np.array([target.pos[0], target.vel[0], target.pos[1], target.vel[1], target.pos[2], target.vel[2]])  # True position
        error = est - true  # Error
        nees = np.dot(error.T, np.dot(np.linalg.inv(P), error))  # NEES
        nis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))[0][0]  # NIS
        
        # Calculate Track Quaility Metric
        trackQuality = self.calcTrackQuailty(est, P)

        # Save data
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = np.reshape(innovation, (numMeasurements))
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis
        self.trackQualityHist[targetID][envTime] = trackQuality
        self.gottenEstimate = True
    
    def state_transition(self, estPrior, dt):
        """
        State Transition Function: Converts ECI Cartesian state to spherical coordinates,
        propagates state over time using Runge-Kutta 4th order integration, and converts
        back to Cartesian. Inputs current state and time step, returns next state.

        Parameters:
        - estPrior (array-like): Current state in Cartesian coordinates [x, vx, y, vy, z, vz].
        - dt (float): Time step for integration.

        Returns:
        - array-like: Next state in Cartesian coordinates [x, vx, y, vy, z, vz].
        """
        x, vx, y, vy, z, vz = estPrior
            
        # Convert to Spherical Coordinates
        range = jnp.sqrt(x**2 + y**2 + z**2)
        elevation = jnp.arcsin(z / range)
        azimuth = jnp.arctan2(y, x)
        
        # Calculate the Range Rate
        rangeRate = (x * vx + y * vy + z * vz) / range
    
        # Calculate Elevation Rate
        elevationRate = -(z * (vx * x + vy * y) - (x**2 + y**2) * vz) / ((x**2 + y**2 + z**2) * jnp.sqrt(x**2 + y**2))
    
        # Calculate Azimuth Rate
        azimuthRate = (x * vy - y * vx) / (x**2 + y**2)
        
        # Previous Spherical State
        prev_spherical_state = jnp.array([range, rangeRate, elevation, elevationRate, azimuth, azimuthRate])
        
        # Define function to compute derivatives for Runge-Kutta method
        def derivatives(spherical_state):
            """
            Computes derivatives of spherical state variables.

            Parameters:
            - spherical_state (array-like): Spherical state variables [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate].

            Returns:
            - array-like: Derivatives [rangeRate, rangeAccel, elevationRate, elevationAccel, azimuthRate, azimuthAccel].
            """
            r, rdot, elevation, elevationRate, azimuth, azimuthRate = spherical_state
            
            rangeRate = rdot
            rangeAccel = 0  # No acceleration assumed in this simplified model
            elevationRate = elevationRate
            elevationAccel = 0  # No acceleration assumed in this simplified model
            azimuthRate = azimuthRate
            azimuthAccel = 0  # No acceleration assumed in this simplified model
            
            return jnp.array([rangeRate, rangeAccel, elevationRate, elevationAccel, azimuthRate, azimuthAccel])
        
        # Runge-Kutta 4th order integration
        k1 = derivatives(prev_spherical_state)
        k2 = derivatives(prev_spherical_state + 0.5 * dt * k1)
        k3 = derivatives(prev_spherical_state + 0.5 * dt * k2)
        k4 = derivatives(prev_spherical_state + dt * k3)
        
        next_spherical_state = prev_spherical_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Extract components from spherical state
        range = next_spherical_state[0]
        rangeRate = next_spherical_state[1]
        elevation = next_spherical_state[2]
        elevationRate = next_spherical_state[3]
        azimuth = next_spherical_state[4]
        azimuthRate = next_spherical_state[5]
        
        # Convert back to Cartesian coordinates
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
        
        # Return the next state in Cartesian coordinates
        return jnp.array([x, vx, y, vy, z, vz])

    def state_transition_jacobian(self, estPrior, dt):
        """
        Calculates the Jacobian matrix of the state transition function.

        Parameters:
        - estPrior (array-like): Current state in Cartesian coordinates [x, vx, y, vy, z, vz].
        - dt (float): Time step for integration.

        Returns:
        - array-like: Jacobian matrix of the state transition function.
        """
        jacobian = jax.jacfwd(lambda x: self.state_transition(x, dt))(jnp.array(estPrior))
        
        return jacobian

    
    def calcTrackQuailty(self, est, cov):
        """
        Calculate the track quality metric for the current estimate.

        Args:
        - est (array-like): Current estimate for the target ECI state = [x, vx, y, vy, z, vz].
        - cov (array-like): Current covariance matrix for the target ECI state.

        Returns:
        - float: Track quality metric.
        """
        # Calculate the track quality metric
        trackQuality = 1 / np.linalg.det(cov)
        
        return trackQuality

### Central Estimator Class
class centralEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        """
        Initialize Central Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        """
        Extended Kalman Filter for central estimation.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.

        Returns:
        - np.array: Estimated state after filtering.
        """
        return super().EKF(sats, measurements, target, envTime)


### Independent Estimator Class
class indeptEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        """
        Initialize Independent Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        """
        Extended Kalman Filter for independent estimation.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.

        Returns:
        - np.array: Estimated state after filtering.
        """
        return super().EKF(sats, measurements, target, envTime)


### DDF Estimator Class
class ddfEstimator(BaseEstimator):
    def __init__(self, targetIDs):
        """
        Initialize DDF Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)

    def EKF(self, sats, measurements, target, envTime):
        """
        Extended Kalman Filter for decentralized data fusion estimation.

        Args:
        - sats (list): List of satellites.
        - measurements (list): List of measurements.
        - target (object): Target object.
        - envTime (float): Current environment time.

        Returns:
        - np.array: Estimated state after filtering.
        """
        return super().EKF(sats, measurements, target, envTime)


    def CI(self, sat, commNode):
        """
        Covariance Intersection function to conservatively combine received estimates and covariances
        into updated target state and covariance.

        Args:
            sat: Satellite object that is receiving information.
            commNode (dict): Communication node containing queued data from satellites.

        Returns:
            estPrior (array-like): Updated current state in Cartesian coordinates [x, vx, y, vy, z, vz].
            covPrior (array-like): Updated current covariance matrix.
        """
        # Check if there is any information in the queue:
        if len(commNode['queued_data']) == 0:
            return

        # There is information in the queue, get the newest info
        time_sent = max(commNode['queued_data'].keys())

        # Check all the targets in the queue
        for targetID in commNode['queued_data'][time_sent].keys():

            # For each target, loop through all the estimates and covariances
            for i in range(len(commNode['queued_data'][time_sent][targetID]['est'])):
                est_sent = commNode['queued_data'][time_sent][targetID]['est'][i]
                cov_sent = commNode['queued_data'][time_sent][targetID]['cov'][i]

                # Check if satellite has an estimate and covariance for this target already
                if not self.estHist[targetID] and not self.covarianceHist[targetID]:
                    # If not, use the sent estimate and covariance to initialize
                    self.estHist[targetID][time_sent] = est_sent
                    self.covarianceHist[targetID][time_sent] = cov_sent
                    continue

                # If satellite has an estimate and covariance for this target already, check if we should CI
                time_prior = max(self.estHist[targetID].keys())

                # If the send time is older than the prior estimate, discard the sent estimate
                if time_sent < time_prior:
                    continue

                # If the time between the sent estimate and the prior estimate is greater than 5 minutes, discard the prior
                if time_sent - time_prior > 5:
                    self.estHist[targetID][time_sent] = est_sent
                    self.covarianceHist[targetID][time_sent] = cov_sent
                    continue

                # Else, let's do CI
                est_prior = self.estHist[targetID][time_prior]
                cov_prior = self.covarianceHist[targetID][time_prior]

                # Propagate the prior estimate and covariance to the new time
                dt = time_sent - time_prior
                est_prior = self.state_transition(est_prior, dt)
                F = self.state_transition_jacobian(est_prior, dt)
                cov_prior = np.dot(F, np.dot(cov_prior, F.T))

                # Minimize the covariance determinant
                omega_opt = minimize(self.det_of_fused_covariance, [0.5], args=(cov_prior, cov_sent), bounds=[(0, 1)]).x

                # Compute the fused covariance
                cov1 = cov_prior
                cov2 = cov_sent
                cov_prior = np.linalg.inv(omega_opt * np.linalg.inv(cov1) + (1 - omega_opt) * np.linalg.inv(cov2))
                est_prior = cov_prior @ (omega_opt * np.linalg.inv(cov1) @ est_prior + (1 - omega_opt) * np.linalg.inv(cov2) @ est_sent)

                # Save the fused estimate and covariance
                self.estHist[targetID][time_sent] = est_prior
                self.covarianceHist[targetID][time_sent] = cov_prior
    
    
    def det_of_fused_covariance(self, omega, cov1, cov2):
        """
        Calculate the determinant of the fused covariance matrix.

        Args:
            omega (float): Weight of the first covariance matrix.
            cov1 (np.ndarray): Covariance matrix of the first estimate.
            cov2 (np.ndarray): Covariance matrix of the second estimate.

        Returns:
            float: Determinant of the fused covariance matrix.
        """
        omega = omega[0]  # Ensure omega is a scalar
        P = np.linalg.inv(omega * np.linalg.inv(cov1) + (1 - omega) * np.linalg.inv(cov2))
        return np.linalg.det(P)

### Event Triggered Estimator Class
class etEstimator(BaseEstimator):
    def __init__(self, targetIDs, targets=None, sat=None, neighbors=None):
        """
        Initialize Event Triggered Estimator object. This object holds a local filter
        and a common information filter between two agents.

        Args:
        - targets (object): List of target objects to track.
        - sat (object): Satellite object.
        - neighbors (list of objects, optional): Neighbor satellite objects.
        
        Returns:
         - Dictionaries containing local and common information filters
        """
        # Create local and shared information filters
        self.targetIDs = targetIDs
        self.targs = targets
        self.sat = sat
        self.neighbors = [sat] + neighbors if neighbors else [sat]
        
        # ET Parameters
        self.delta_alpha = 0.5
        self.delta_beta = 0.5

        # Define history vectors for each extended Kalman filter
        self.estHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.covarianceHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.measHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        
    def update_neighbors(self, neighbors):
        '''
        Update the neighbors of the satellite object. Required since the neighbors are not known at initialization.
        '''
        self.neighbors = [self.sat] + neighbors
        # Update history vectors with new neighbors
        for targetID in self.targetIDs:
            self.estHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.covarianceHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.measHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            

    def event_triggered_fusion(self, sat, envTime, commNode):
        '''
        Should be called by the satellite object to run the event triggered fusion algorithm.
        Predicts the next state in local and common Extended Kalman Filter and then sequentially updates
        the state in local and common filters using implicit and explicit measurements from neighbors
        
        Args:
        - sat (object): Satellite object.
        - envTime (float): Current environment time.
        - commNode (dict): Communication node containing queued data from satellites.
        
        Returns:
        - None
        '''
        # For each target
        for target in self.targs:
            # Get the targetID
            targetID = target.targetID
            # Did I take a measurement on this target
            if len(sat.measurementHist[targetID][envTime]) > 0: ## explicitly update local filter
                # Get the measurements for this target
                alpha, beta = sat.measurementHist[targetID][envTime]
                
                # Check if I have initialized local filter for this target
                if len(self.estHist[targetID][sat][sat]) == 0 and len(self.covarianceHist[targetID][sat][sat]) == 0:
                    # If no prior estimate exists, initialize with true position plus noise
                    prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + 15
                    prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) * 1.5
                    est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
                    
                    # Initial covariance matrix
                    P_prior = np.array([[625, 0, 0, 0, 0, 0],
                                        [0, 100, 0, 0, 0, 0],
                                        [0, 0, 625, 0, 0, 0],
                                        [0, 0, 0, 100, 0, 0],
                                        [0, 0, 0, 0, 625, 0],
                                        [0, 0, 0, 0, 0, 100]])
                    
                    # Store initial values and return for first iteration
                    self.estHist[targetID][sat][sat][envTime] = est_prior
                    self.covarianceHist[targetID][sat][sat][envTime] = P_prior
                    
                    return # No need to continue if this is the first measurement
                
                # Otherwise I have an initialized local filter for this target                
                # Run Prediction Step on this target for local fitler
                self.pred_EKF(sat, sat, targetID, envTime)
                est_pred = self.estHist[targetID][sat][sat][envTime]
                cov_pred = self.covarianceHist[targetID][sat][sat][envTime]
                
                # Proccess my measurement in the local filter
                self.explicit_measurement_update(sat, sat, alpha, 'IT', targetID, envTime)
                self.explicit_measurement_update(sat, sat, beta, 'CT', targetID, envTime)
            
                # Check if there is any information in the queue:
                if len(commNode['measurement_data']) > 0: 
                    time_sent = max(commNode['measurement_data'].keys()) # Get the newest info
                    
                    # If I got a new measurement on the same target from a neighbor'
                    if targetID in commNode['measurement_data'][time_sent].keys():
                        
                        # Process the new measurement on the target
                        for i in range(len(commNode['measurement_data'][time_sent])): # number of messages on this target
                            
                            # Get the satellite sender to check if I have a common filter with this neighbor
                            sender = commNode['measurement_data'][time_sent][targetID]['sender'][i]
                            
                            # Check if I have initialized common filter for this target
                            if len(self.estHist[targetID][sat][sender]) == 0 and len(self.covarianceHist[targetID][sat][sender]) == 0:
                                # If no prior estimate exists, initialize with true position plus noise
                                prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + 15
                                prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) * 1.5
                                est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])

                                # Initial covariance matrix
                                P_prior = np.array([[625, 0, 0, 0, 0, 0],
                                                    [0, 100, 0, 0, 0, 0],
                                                    [0, 0, 625, 0, 0, 0],
                                                    [0, 0, 0, 100, 0, 0],
                                                    [0, 0, 0, 0, 625, 0],
                                                    [0, 0, 0, 0, 0, 100]])

                                # Store initial values and return for first iteration
                                self.estHist[targetID][sat][sender][time_sent] = est_prior
                                self.covarianceHist[targetID][sat][sender][time_sent] = P_prior

                                return # No need to continue if this is the shared first measurement between both
                            
                            # Otherwise I have an initialized common filter with this neighbor
                                                        
                            # Run Prediction Step on this target for common fitler
                            self.pred_EKF(sat, sender, targetID, envTime)
    
                            # Proccess the new measurement with the local and common filter
                            alpha, beta = commNode['measurement_data'][time_sent][targetID]['meas'][i]
                            
                            # In-Track Measurement
                            if not np.isnan(alpha):
                                # update local filter
                                self.explicit_measurement_update(sat, sat, alpha, 'IT', targetID, envTime)
                                # update common filter
                                self.explicit_measurement_update(sat, sender, alpha, 'IT', targetID, envTime)
                            
                            else:
                                # update both filters
                                self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'IT', 'both', targetID, envTime) # update my local and common filter
                
                            # Cross-Track Measurement
                            if not np.isnan(beta):
                                # update local filter
                                self.explicit_measurement_update(sat, sat, beta, 'CT', targetID, envTime)
                                # update common filter
                                self.explicit_measurement_update(sat, sender, beta, 'CT', targetID, envTime)
                            
                            else:
                                # update both filters
                                self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'CT', 'both', targetID, envTime)
                            
                    # If I took a measurement, I probably sent it somewhere, so I should update my common filter with the people i sent it to
                    for neighbor in self.neighbors:
                        if neighbor != sat:
                            if targetID in self.measHist.keys():
                                if neighbor in self.measHist[targetID][sat].keys():
                                    if envTime in self.measHist[targetID][sat][neighbor].keys():
                                        
                                        # Get the mesurement i sent this target
                                        alpha, beta = self.measHist[targetID][sat][neighbor][envTime]
                                        
                                        # Check if I have initialized common filter with this neighbor
                                        if len(self.estHist[targetID][sat][neighbor]) == 0 and len(self.covarianceHist[targetID][sat][neighbor]) == 0:
                                            # If no prior estimate exists, initialize with true position plus noise
                                            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + 15
                                            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) * 1.5
                                            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])

                                            # Initial covariance matrix
                                            P_prior = np.array([[625, 0, 0, 0, 0, 0],
                                                                [0, 100, 0, 0, 0, 0],
                                                                [0, 0, 625, 0, 0, 0],
                                                                [0, 0, 0, 100, 0, 0],
                                                                [0, 0, 0, 0, 625, 0],
                                                                [0, 0, 0, 0, 0, 100]])

                                            # Store initial values and return for first iteration
                                            self.estHist[targetID][sat][neighbor][envTime] = est_prior
                                            self.covarianceHist[targetID][sat][neighbor][envTime] = P_prior

                                            return # no need to continue if this is the first sent measurement to this neighbor
                                        
                                        # Otherwise I have an initialized common filter with this neighbor that I sent something to
                                        
                                        # If either measurement is missing, update the common filter with the implicit measurement
                                        if np.isnan(alpha) or np.isnan(beta):
                                            
                                            # Run Prediction Step on this target for local fitler
                                            self.pred_EKF(sat, sat, targetID, envTime)
                                            est_pred = self.estHist[targetID][sat][neighbor][envTime]
                                            cov_pred = self.covarianceHist[targetID][sat][neighbor][envTime]
                                            
                                            self.pred_EKF(sat, neighbor, targetID, envTime) # TODO: if this got updated previously this should propagate nothing
                                            
                                            # Update the common filter with the sent measurement
                                            if not np.isnan(alpha):
                                                # In-Track Measurement
                                                self.explicit_measurement_update(sat, neighbor, alpha, 'IT', targetID, envTime)
                                            else:
                                                self.implicit_measurement_update(sat, neighbor, est_pred, cov_pred, 'IT', 'common', targetID, envTime)
                                            
                                            if not np.isnan(beta):
                                                # Cross-Track Measurement
                                                self.explicit_measurement_update(sat, neighbor, beta, 'CT', targetID, envTime)
                                            else:
                                                self.implicit_measurement_update(sat, neighbor, est_pred, cov_pred, 'CT', 'common', targetID, envTime)
                                        
                                        
                        
    def pred_EKF(self, sat, neighbor, targetID, envTime):
        '''
        Predict the next state using the Extended Kalman Filter.
        
        Args:
        - sat (object): Satellite object.
        - neighbor (object): Neighbor satellite object.
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        
        Returns:
        - Updated estimate and covariance for the target ECI state = [x, vx, y, vy, z, vz].
        '''
  
        # Get the most recent estimate and covariance        
        time_prior = max(self.estHist[targetID][sat][neighbor].keys())
        est_prior = self.estHist[targetID][sat][neighbor][time_prior]
        P_prior = self.covarianceHist[targetID][sat][neighbor][time_prior]

        # Calculate time difference since last estimate
        dt = envTime - time_prior

        # Predict next state using state transition function
        est_pred = super().state_transition(est_prior, dt)
        
        # Evaluate Jacobian of state transition function
        F = super().state_transition_jacobian(est_prior, dt)
        
        # Predict process noise associated with state transition
        Q = np.diag([50, 1, 50, 1, 50, 1]) # Large # Process noise matrix

        # Predict covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q ## TODO check dot products?
        
        # Store the prediction
        self.estHist[targetID][sat][neighbor][envTime] = est_pred
        self.covarianceHist[targetID][sat][neighbor][envTime] = P_pred
        
    
        
    def explicit_measurement_update(self, sat, sender, measurement, type, targetID, envTime):
        '''
        Explicit measurement updates  the estimate with a measurement from a sender satellite.
        Note satellites can send themselves measurements for local filter.
        
        Args:
        - sat (object): Satellite object that is receiving information.
        - sender (object): Sender satellite object that sent information
        - measurements (np.array): Explicit measurements from the sender satellite: [alpha, 0] or [0, beta]
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        
        Returns:
        - Updated estimate and covariance for the target ECI state = [x, vx, y, vy, z, vz].
        '''
        if type == 'IT':
            scalarIdx = 0
        elif type == 'CT':
            scalarIdx = 1
        
        # The recieving filter most recent Estimate and Covariance
        est_prev = self.estHist[targetID][sat][sender][envTime]
        P_prev = self.covarianceHist[targetID][sat][sender][envTime]
        
        # In Track or Cross Track Measurement actually sent by the sender
        z = measurement
        
        # The measurement I/we think you should have got based on my estimate 
        z_pred = np.array(sender.sensor.convert_to_bearings(sender, np.array([est_prev[0], est_prev[2], est_prev[4]])))  # Predicted measurements
        z_pred = z_pred[scalarIdx]
        
        # The uncertaininty in the measurement you should have got based on my/our estimate
        H = sender.sensor.jacobian_ECI_to_bearings(sender, est_prev)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))
        
        R = sender.sensor.bearingsError[scalarIdx]**2 * 1000 # Sensor noise matrix scaled by 1000x
                    
        # Compute innovation             
        innovation = z - z_pred

        # Calculate innovation covariance
        innovationCov = (H @ P_prev @ H.T + R)**-1
        #np.dot(H, np.dot(P_pred, H.T)) + R
        
        # Solve for Kalman gain
        K = np.reshape(P_prev @ H.T * innovationCov, (6,1))
                        
        # Correct prediction
        est = est_prev + np.reshape(K * innovation, (6))
        
        # Correct covariance # TODO: this fucks cov
        P = P_prev - K @ H @ P_prev
        #np.reshape(H @ P_pred, (1,6)) 
        
        # Save Data into both filters
        self.estHist[targetID][sat][sender][envTime] = est
        self.covarianceHist[targetID][sat][sender][envTime] = P
                
        
    def implicit_measurement_update(self, sat, sender, local_est_pred, local_P_pred, type, update, targetID, envTime):
        """
        Implicit measurement update function to update the estimate without a measurement.
        Fuses the local estimate with implicit information shared from a paired satellite.

        Args:
        - sat (object): Satellite object that is receiving information.
        - target (object): Target object.
        - envTime (float): Current environment time.

        Returns:
        - np.array: Updated estimate for the target ECI state = [x, vx, y, vy, z, vz].
        """
        if type == 'IT':
            scalarIdx = 0
            delta = self.delta_alpha # Threshold Factor
            
        elif type == 'CT':
            scalarIdx = 1
            delta = self.delta_beta
                    
        # Grab the best current local Estimate and Covariance
        local_time = max(self.estHist[targetID][sat][sat].keys())
        local_est_curr = self.estHist[targetID][sat][sat][local_time]
        local_cov_curr = self.covarianceHist[targetID][sat][sat][local_time]

        # Grab the shared best local Estimate and Covariance
        common_time = max(self.estHist[targetID][sat][sender].keys())
        common_est_curr = self.estHist[targetID][sat][sender][common_time]
        
        # Compute Expected Value of Implicit Measurement
        mu = np.array(sat.sensor.convert_to_bearings(sat, np.array([local_est_curr[0], local_est_curr[2], local_est_curr[4]]))) - np.array(sat.sensor.convert_to_bearings(sat, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        mu = mu[scalarIdx]
        
        # Compute Alpha
        alpha = np.array(sat.sensor.convert_to_bearings(sat, np.array([common_est_curr[0], common_est_curr[2], common_est_curr[4]]))) - np.array(sat.sensor.convert_to_bearings(sat, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        alpha = alpha[scalarIdx]
        
        # Compute the Sensor Jacobian
        H = sender.sensor.jacobian_ECI_to_bearings(sender, local_est_curr)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))
                
        # Compute Sensor Noise Matrix
        R = sender.sensor.bearingsError[scalarIdx]**2 * 1000 # Sensor noise matrix scaled by 1000x
                
        # Define innovation covariance
        Qe = H @ local_P_pred @ H.T + R         
        
        # Compute Expected Value of Implicit Measurement
        vminus = (-delta + alpha - mu)/np.sqrt(Qe)
        vplus = (delta + alpha - mu)/np.sqrt(Qe)
        
        # Probability Density Function
        phi1 = norm.pdf(vminus)
        phi2 = norm.pdf(vplus)
        
        # Cumulative Density Function
        Q1 = norm.cdf(vminus)
        Q2 = norm.cdf(vplus)
        
        # Compute Expected Value of Measurement
        zbar = (phi1 - phi2)/(Q1 - Q2) * np.sqrt(Qe)
                                        
        # Compute Expected Variance of Implicit Measurement
        nu = ((phi1 - phi2)/(Q1 - Q2))**2 - (vminus * phi1 - vplus * phi2) / (Q1 - Q2)
                
        # Compute Kalman Gain
        K = np.reshape(local_cov_curr @ H.T * (H @ local_cov_curr @ H.T + R)**-1, (6,1))
        
        # Update Estimate
        est = local_est_curr + np.reshape(K * zbar, (6))
        
        # Update Covariance
        cov = local_cov_curr - nu * K @ H @ local_cov_curr
    
        # Save Data into both filters
        if update == 'both':
            self.estHist[targetID][sat][sat][envTime] = est
            self.estHist[targetID][sat][sender][envTime] = est
        
            self.covarianceHist[targetID][sat][sat][envTime] = cov
            self.covarianceHist[targetID][sat][sender][envTime] = cov
        
        elif update == 'common':
            self.estHist[targetID][sat][sender][envTime] = est
            self.covarianceHist[targetID][sat][sender][envTime] = cov

        
    
    
    def event_trigger(self, sat, neighbor, targetID, time):
        """
        Event Trigger function to determine if an explict or implicit
        measurement update is needed.

        Args:
        - sat (object): Satellite object that is receiving information.
        - target (object): Target object.

        Returns:
        - send_alpha (float): Alpha value to send to neighbor - NaN if not needed.
        - send_beta (float): Beta value to send to neighbor - NaN if not needed.
        """
        # Get the most recent measurement on the target
        alpha, beta = sat.measurementHist[targetID][time]
        
        # For each measurement, does my neighbor think this is important?
        
        # Check if a valid estimate exists
        if not self.estHist[targetID][sat][neighbor]:
            return alpha, beta
        
        # Get most recent common estimate
        satTime = max(self.estHist[targetID][sat][neighbor].keys())
        est = sat.etEstimator.estHist[targetID][sat][neighbor][satTime] # reference state
        
        # Predict the Measurement that the neighbor would make
        meas = sat.sensor.convert_to_bearings(sat, np.array([est[0], est[2], est[4]]))
        
        # Is my measurment surprising based on the prediction?
        send_alpha = np.nan
        send_beta = np.nan
        
        # If the difference between the measurement and the prediction is greater than delta, send the measurement
        if np.abs(alpha - meas[0]) > self.delta_alpha:
            send_alpha = alpha
            
        if np.abs(beta - meas[1]) > self.delta_beta:
            send_beta = beta
        
        return send_alpha, send_beta