from import_libraries import *

class BaseEstimator:
    def __init__(self, targetPriorities):
        """
        Initialize the BaseEstimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """

        # Define the targetPriorities dictioanry
        self.targetPriorities = targetPriorities

        # Define the targets to track
        self.targs = targetPriorities.keys()

        # Define history vectors for each extended Kalman filter
        self.estHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of Kalman estimates in ECI coordinates
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of covariance matrices

        self.innovationHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of innovations
        self.innovationCovHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of innovation covariances

        self.neesHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of NEES (Normalized Estimation Error Squared)
        self.nisHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of NIS (Normalized Innovation Squared)
        
        self.trackErrorHist = {targetID: defaultdict(dict) for targetID in targetPriorities.keys()}  # History of track quality metric

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
            # If not prior estimate exists:
            
            # TODO: Eventually should initialize the kalman filter with the real of what target was sampled from
            # # If the environment time == 0, initialize the filter with what the target was sampled from
            # if envTime == 0:
            #     code would go here

            # # If the environment time is not 0, just initalize the filter with the true position plus some noise
            # else:
            # Initialize with true position plus noise
            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 5, 3)
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
            
            # Initial covariance matrix
            P_prior = np.array([[2500, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 2500, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 2500, 0],
                                [0, 0, 0, 0, 0, 100]])

            # Store initial values and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            
            # Add the trackUncertainty
            trackError = self.calcTrackError(P_prior)
            self.trackErrorHist[targetID][envTime] = trackError

            return est_prior
       
        else:
            # Get most recent estimate and covariance
            time_prior = max(self.estHist[targetID].keys())
            est_prior = self.estHist[targetID][time_prior]
            P_prior = self.covarianceHist[targetID][time_prior]

        # Calculate time difference since last estimate
        dt = envTime - time_prior

        ### Also reset the filter if its been a certain amount of time since the last estimate
        if dt > 30: 
            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 5, 3)
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
            # Initial covariance matrix
            P_prior = np.array([[2500, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 2500, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 2500, 0],
                                [0, 0, 0, 0, 0, 100]])

            dt = 0 # Reset the time difference since were just reinitializing at this time

        # Predict next state using state transition function
        est_pred = self.state_transition(est_prior, dt)
        
        # Evaluate Jacobian of state transition function
        F = self.state_transition_jacobian(est_prior, dt)
        
        # Predict process noise associated with state transition
        # Q = np.diag([50, 1, 50, 1, 50, 1]) # Larger Q 
        Q = np.diag([0.1, 0.01, 0.1, 0.01, 0.1, 0.01])  # Smaller Q 

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
            R[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2  # Sensor noise matrix
            
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
        trackError = self.calcTrackError(P)
        self.trackErrorHist[targetID][envTime] = trackError

        # Save data
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = np.reshape(innovation, (numMeasurements))
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis
    
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

    def calcTrackError(self, cov):
        """
        Calculate the track quality metric for the current estimate.

        Args:
        - est (array-like): Current estimate for the target ECI state = [x, vx, y, vy, z, vz].
        - cov (array-like): Current covariance matrix for the target ECI state.

        Returns:
        - float: Track quality metric.
        """
        # Since we are guaranteeing that the state will always live within the covariance estimate
        # Why not just use the diag of the covariance matrix, then calculate the 2 sigma bounds of xyz
        # Then take norm of xyz max as the error

        # Calculate the 2 sigma bounds of the position
        diag = np.diag(cov)

        # Calculate the std
        posStd = np.array([np.sqrt(diag[0]), np.sqrt(diag[2]), np.sqrt(diag[4])])

        # Now calculate the 2 sigma bounds of the position
        pos2Sigma = 2 * posStd

        # Now get the norm of the 2 sigma bounds xyz
        posError_1side = np.linalg.norm(pos2Sigma)

        # Multiple this value by 2 to get the total error the state could be in
        posError = 2 * posError_1side

        # print(f"Position Error: {posError}")
        
        return posError

### Central Estimator Class
class centralEstimator(BaseEstimator):
    def __init__(self, targPriorities):
        """
        Initialize Central Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)

        self.R_factor = 1  # Factor to scale the sensor noise matrix
        # self.R_factor = 100 # can be used to really ensure filter stays working, pessimiestic

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
    def __init__(self, targPriorities):
        """
        Initialize Independent Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)

        self.R_factor = 1
        # self.R_factor = 100 # can be used to really ensure filter stays working, pessimiestic 

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

### CI Estimator Class
# Always does covariance intersection
class ciEstimator(BaseEstimator):
    def __init__(self, targPriorities):
        """
        Initialize DDF Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targPriorities)
            
        self.R_factor = 1  # Factor to scale the sensor noise matrix
        # self.R_factor = 100 # can be used to really ensure filter stays working, pessimiestic

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

        # Perform EKF for each satellite
        super().EKF(sats, measurements, target, envTime)


    def CI(self, sat, comms):
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

        commNode = comms.G.nodes[sat]

        # Check if there is any information in the queue:
        if len(commNode['estimate_data']) == 0:
            # If theres no information in the queue, return
            return

        # There is information in the queue, get the newest info
        time_sent = max(commNode['estimate_data'].keys())

        # Check all the targets that are being talked about in the queue
        for targetID in commNode['estimate_data'][time_sent].keys():

            # Is that target something this satellite should be tracking? Or just ferrying data?
            if targetID not in sat.targetIDs:
                continue

            # For each target we should track, loop through all the estimates and covariances
            for i in range(len(commNode['estimate_data'][time_sent][targetID]['est'])):
                senderName = commNode['estimate_data'][time_sent][targetID]['sender'][i]
                est_sent = commNode['estimate_data'][time_sent][targetID]['est'][i]
                cov_sent = commNode['estimate_data'][time_sent][targetID]['cov'][i]

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

                # Now check, does the satellite need help on this target?
                if not not sat.ciEstimator.trackErrorHist[targetID][time_prior]: # An estimate exists for this target
                    if (sat.ciEstimator.trackErrorHist[targetID][time_prior] < sat.targPriority[targetID]): # Is the estimate good enough already?
                        # If the track quality is good, don't do CI
                        continue

                # We will now use the estimate and covariance that were sent, so we should store this
                comms.used_comm_data[targetID][sat.name][senderName][time_sent] = est_sent.size*2 + cov_sent.size/2

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

                # Save the trackUncertainty
                trackError = self.calcTrackError(cov_prior)
                self.trackErrorHist[targetID][time_sent] = trackError
    
    
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
        self.delta_alpha = 0.05
        self.delta_beta = 0.05
        
        # R Factor
        self.R_factor = 100

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
        # TODO: AT THE MOMENT THIS INITALIZES A LOCAL FILTER EVEN IF THE SAT CANT SEE THE TARGET
        # For each target
        for target in self.targs:
            # Get the targetID
            targetID = target.targetID
            
            ### First check if I have initialized a local filter on this target ###
            ### Then check if I have a measurement on this target and explicitly update my local filter ###
            ### Store predicted state before measurment for common filter update ###
            
            if len(self.estHist[targetID][sat][sat]) == 0 and len(self.covarianceHist[targetID][sat][sat]) == 0:
                # If no prior estimate exists, initialize with true position plus noise
                prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
                prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 5, 3)
                est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
                    
                # Initial covariance matrix
                P_prior = np.array([[2500, 0, 0, 0, 0, 0],
                                    [0, 100, 0, 0, 0, 0],
                                    [0, 0, 2500, 0, 0, 0],
                                    [0, 0, 0, 100, 0, 0],
                                    [0, 0, 0, 0, 2500, 0],
                                    [0, 0, 0, 0, 0, 100]])
                
                # Store initial values and return for first iteration
                self.estHist[targetID][sat][sat][envTime] = est_prior
                self.covarianceHist[targetID][sat][sat][envTime] = P_prior
                # TODO: get track error metric?

                return # No need to continue if this is the first measurement
            
            # Otherwise I have an initialized local filter for this target                
            # Run Prediction Step on this target for local fitler
            self.pred_EKF(sat, sat, targetID, envTime)
            est_pred = self.estHist[targetID][sat][sat][envTime]
            cov_pred = self.covarianceHist[targetID][sat][sat][envTime]
            
            # Did I take a measurement on this target
            if len(sat.measurementHist[targetID][envTime]) > 0:
                # Get the measurements for this target
                alpha, beta = sat.measurementHist[targetID][envTime]
                                
                # Proccess my measurement in the local filter
                # TODO: share with myself
                # TODO: will this just keep processing the same measurement over and over again?
                self.explicit_measurement_update(sat, sat, alpha, 'IT', 'both', targetID, envTime, sharewith=sat) 
                self.explicit_measurement_update(sat, sat, beta, 'CT', 'both', targetID, envTime, sharewith=sat)
            
            ### Check if I have any information from neighbors and update both filter ###
            # Check if there is any information in the queue:
            if len(commNode['measurement_data']) > 0: 
                time_sent = max(commNode['measurement_data'].keys()) # Get the newest info on this target
                if targetID in commNode['measurement_data'][time_sent].keys():
                    for i in range(len(commNode['measurement_data'][time_sent])): # number of messages on this target?
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

                            return # No need to continue if this is the first shared measurement between both
                        
                        # Otherwise I have an initialized common filter with this neighbor
                        
                        # Run Prediction Step on this target for common fitler
                        # TODO: Rename function to highlight that this is for the common filter
                        self.pred_EKF(sat, sender, targetID, envTime)
                        
                        # Proccess the new measurement from sender with the local and common filter
                        alpha, beta = commNode['measurement_data'][time_sent][targetID]['meas'][i]
                        # In-Track Measurement
                        if not np.isnan(alpha):
                            self.explicit_measurement_update(sat, sender, alpha, 'IT', 'both', targetID, envTime, sharewith=sender) # update our common filter
                        else:
                            self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'IT', 'both', targetID, envTime, sharewith=sender) # update my local and common filter
                        # Cross-Track Measurement
                        if not np.isnan(beta):
                            self.explicit_measurement_update(sat, sender, beta, 'CT', 'both', targetID, envTime, sharewith=sender) # update our common filter
                        else:
                            self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'CT', 'both', targetID, envTime, sharewith=sender) # update my local and common filter

            ### Update the common filter with all measurements I sent to neighbors ###
            for neighbor in self.neighbors: ## TODO: probably easier to do this
                if neighbor != sat:
                    if targetID in self.measHist.keys(): 
                        if neighbor in self.measHist[targetID][sat].keys(): # if I have sent measurements on this target to this neighbor
                            if envTime in self.measHist[targetID][sat][neighbor].keys(): # at this current time 
                                # I need to update the common filters so they don't drift
                                    
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
                            
                                # Run Prediction Step on this target for common fitler
                                self.pred_EKF(sat, neighbor, targetID, envTime)
                                 
                                # Get the mesurement that I just sent this neighbor
                                alpha, beta = self.measHist[targetID][sat][neighbor][envTime]
                                
                                if not np.isnan(alpha): ## TODO: sat neighbor wont work bc i am sender not neighbor
                                    self.explicit_measurement_update(sat, sat, alpha, 'IT', 'common', targetID, envTime, sharewith=neighbor) # update our common filter
                                else:
                                    self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'IT', 'common', targetID, envTime, sharewith=neighbor)
                                if not np.isnan(beta):
                                    self.explicit_measurement_update(sat, sat, beta, 'CT', 'common', targetID, envTime, sharewith=neighbor) # update our common filter
                                else:
                                    self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'CT', 'common', targetID, envTime, sharewith=neighbor)
                                        
                                        
                        
    def pred_EKF(self, sat, sender, targetID, envTime):
        '''
        Predict the next state using the Extended Kalman Filter.
        
        Args:
        - sat (object): Satellite object.
        - sender (object): Sender satellite object.
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        
        Returns:
        - Updated estimate and covariance for the target ECI state = [x, vx, y, vy, z, vz].
        '''
  
        # Get the most recent estimate and covariance        
        time_prior = max(self.estHist[targetID][sat][sender].keys())
        est_prior = self.estHist[targetID][sat][sender][time_prior]
        P_prior = self.covarianceHist[targetID][sat][sender][time_prior]

        # Calculate time difference since last estimate
        dt = envTime - time_prior
        
        if dt == 0:
            self.estHist[targetID][sat][sender][envTime] = est_prior
            self.covarianceHist[targetID][sat][sender][envTime] = P_prior
            return

        # Predict next state using state transition function
        est_pred = super().state_transition(est_prior, dt)
        
        # Evaluate Jacobian of state transition function
        F = super().state_transition_jacobian(est_prior, dt)
        
        # Predict process noise associated with state transition
        Q = np.diag([50, 1, 50, 1, 50, 1]) # Large # Process noise matrix

        # Predict covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q ## TODO check dot products?
        
        # Store the prediction
        self.estHist[targetID][sat][sender][envTime] = est_pred
        self.covarianceHist[targetID][sat][sender][envTime] = P_pred
        
         
    def explicit_measurement_update(self, sat, sender, measurement, type, update, targetID, envTime, sharewith):
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
        est_prev = self.estHist[targetID][sat][sharewith][envTime]
        P_prev = self.covarianceHist[targetID][sat][sharewith][envTime]
        
        # In Track or Cross Track Measurement actually sent by the sender
        z = measurement
        
        # The measurement I/we think you should have got based on my estimate 
        z_pred = np.array(sender.sensor.convert_to_bearings(sender, np.array([est_prev[0], est_prev[2], est_prev[4]])))  # Predicted measurements
        z_pred = z_pred[scalarIdx]
        
        # The uncertaininty in the measurement you should have got based on my/our estimate
        H = sender.sensor.jacobian_ECI_to_bearings(sender, est_prev)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))
        
        R = sender.sensor.bearingsError[scalarIdx]**2 * self.R_factor # Sensor noise matrix scaled by 1000x
                    
        # Compute innovation             
        innovation = z - z_pred

        # Calculate innovation covariance
        innovationCov = (H @ P_prev @ H.T + R)**-1
        #np.dot(H, np.dot(P_pred, H.T)) + R
        
        # Solve for Kalman gain
        K = np.reshape(P_prev @ H.T * innovationCov, (6,1))
                        
        # Correct prediction
        est = est_prev + np.reshape(K * innovation, (6))
        
        # Correct covariance 
        P = P_prev - K @ H @ P_prev
         
        # Save Data into both filters
        if update == 'both':
            self.estHist[targetID][sat][sat][envTime] = est
            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sat][envTime] = P
            self.covarianceHist[targetID][sat][sharewith][envTime] = P
            
        elif update == 'common':
            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sharewith][envTime] = P
            
                
        
    def implicit_measurement_update(self, sat, sender, local_est_pred, local_P_pred, type, update, targetID, envTime, sharewith):
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
        common_time = max(self.estHist[targetID][sat][sharewith].keys())
        common_est_curr = self.estHist[targetID][sat][sharewith][common_time]
        
        # Compute Expected Value of Implicit Measurement
        mu = np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_curr[0], local_est_curr[2], local_est_curr[4]]))) - np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        mu = mu[scalarIdx]
        
        # Compute Alpha
        alpha = np.array(sender.sensor.convert_to_bearings(sender, np.array([common_est_curr[0], common_est_curr[2], common_est_curr[4]]))) - np.array(sender.sensor.convert_to_bearings(sender, np.array([local_est_pred[0], local_est_pred[2], local_est_pred[4]])))
        alpha = alpha[scalarIdx]
        
        # Compute the Sensor Jacobian
        H = sender.sensor.jacobian_ECI_to_bearings(sender, local_est_curr)
        H = H[scalarIdx, :] # use relevant row
        H = np.reshape(H, (1,6))
                
        # Compute Sensor Noise Matrix
        R = sender.sensor.bearingsError[scalarIdx]**2 * self.R_factor # Sensor noise matrix scaled by 1000x
                
        # Define innovation covariance
        Qe = H @ local_P_pred @ H.T + R         
        
        # Compute Expected Value of Implicit Measurement
        vminus = (-delta + alpha - mu)/np.sqrt(Qe)
        vplus = (delta + alpha - mu)/np.sqrt(Qe)
        
        # Probability Density Function
        phi1 = norm.pdf(vminus)
        phi2 = norm.pdf(vplus)
        
        # Cumulative Density Function
        Q1 = 1 - norm.cdf(vminus)
        Q2 = 1 - norm.cdf(vplus)
        
        # Compute Expected Value of Measurement
        zbar = (phi1 - phi2)/(Q1 - Q2) * np.sqrt(Qe)
                                        
        # Compute Expected Variance of Implicit Measurement
        nu = ((phi1 - phi2)/(Q1 - Q2))**2 - (vminus * phi1 - vplus * phi2) / (Q1 - Q2)
        print("Satellite: ", sat.name, "Nu: ", nu)  
                
        # Compute Kalman Gain
        K = np.reshape(local_cov_curr @ H.T * (H @ local_cov_curr @ H.T + R)**-1, (6,1))
        
        # Update Estimate
        est = local_est_curr + np.reshape(K * zbar, (6))
        
        # Update Covariance
        cov = local_cov_curr - nu * K @ H @ local_cov_curr
    
        # Save Data into both filters
        if update == 'both':
            self.estHist[targetID][sat][sat][envTime] = est
            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sat][envTime] = cov
            self.covarianceHist[targetID][sat][sharewith][envTime] = cov
        
        elif update == 'common':
            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sharewith][envTime] = cov

        
    
    
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
        
        # Predict the common estimate to the current time a measurement was taken
        self.pred_EKF(sat, neighbor, targetID, time)
        
        # Get the most recent estimate and covariance
        pred_est = self.estHist[targetID][sat][neighbor][time]
        
        # Predict the Measurement that the neighbor would think I make
        pred_alpha, pred_beta = sat.sensor.convert_to_bearings(sat, np.array([pred_est[0], pred_est[2], pred_est[4]]))
        
        # Is my measurment surprising based on the prediction?
        send_alpha = np.nan
        send_beta = np.nan
        
        # If the difference between the measurement and the prediction is greater than delta, send the measurement
        if np.abs(alpha - pred_alpha) > self.delta_alpha:
            send_alpha = alpha
            
        if np.abs(beta - pred_beta) > self.delta_beta:
            send_beta = beta
        
        return send_alpha, send_beta