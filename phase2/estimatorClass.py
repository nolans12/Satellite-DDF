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
        
        self.trackErrorHist = {targetID: defaultdict(dict) for targetID in targetIDs}  # History of track quality metric

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
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 1.5, 3)
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
            self.innovationHist[targetID][envTime] = np.zeros(2)
            self.innovationCovHist[targetID][envTime] = np.eye(2)
            self.nisHist[targetID][envTime] = 0
            self.neesHist[targetID][envTime] = 0
            self.trackErrorHist[targetID][envTime] = self.calcTrackError(est_prior, P_prior)
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
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 1.5, 3)
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
        trackError = self.calcTrackError(est, P)
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

    def calcTrackError(self, est, cov):
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
    def __init__(self, targetIDs):
        """
        Initialize Central Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)

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
    def __init__(self, targetIDs):
        """
        Initialize Independent Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)

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
    def __init__(self, targetIDs):
        """
        Initialize DDF Estimator object.

        Args:
        - targetIDs (list): List of target IDs to track.
        """
        super().__init__(targetIDs)
            
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
        return super().EKF(sats, measurements, target, envTime)

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
        if len(commNode['queued_data']) == 0:
            return

        # There is information in the queue, get the newest info
        time_sent = max(commNode['queued_data'].keys())

        # Check all the targets in the queue
        for targetID in commNode['queued_data'][time_sent].keys():

            # For each target, loop through all the estimates and covariances
            for i in range(len(commNode['queued_data'][time_sent][targetID]['est'])):
                senderName = commNode['queued_data'][time_sent][targetID]['sender'][i]
                est_sent = commNode['queued_data'][time_sent][targetID]['est'][i]
                cov_sent = commNode['queued_data'][time_sent][targetID]['cov'][i]

                # Check if satellite has an estimate and covariance for this target already
                if not self.estHist[targetID] and not self.covarianceHist[targetID]:
                    # If not, use the sent estimate and covariance to initialize
                    self.estHist[targetID][time_sent] = est_sent
                    self.covarianceHist[targetID][time_sent] = cov_sent
                    self.trackErrorHist[targetID][time_sent] = self.calcTrackError(est_sent, cov_sent)
                    continue

                # If satellite has an estimate and covariance for this target already, check if we should CI
                time_prior = max(self.estHist[targetID].keys())

                # If the send time is older than the prior estimate, discard the sent estimate
                if time_sent < time_prior:
                    continue

                # TODO: THIS IS ONLY TEMP TO VISUALIZE RESULTS
                #### CHECK, IS MY MOST RECENT TRACK ERROR OUTSIDE THE SPECIFIED BOUNDS?
                # If the track error is outside the bounds, discard the sent estimate
                for i in range(1,6):
                    if i == targetID:

                        doCI = True

                        # First, check does trackErrorHist have a value at that time?
                        if not self.trackErrorHist[i][time_prior]:
                            doCI = True # If we dont have a trackErrorHist yet, we dont have an initalized filter, initalize the filter with the one from CI, so do CI!
                        else:
                            # If the trackErrorHist is less than the threshold, we dont need to do any CI
                            if self.trackErrorHist[i][time_prior] < i*50 + 50: 
                                doCI = False
                        
                        if doCI:

                            # If the trackErrorHist is greater than the threshold, we need to do CI
                            
                            # We will now use the estimate and covariance that were sent, so we should store this
                            comms.used_comm_data[targetID][sat.name][senderName][time_sent] = est_sent.size + cov_sent.size

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
                            
                            # Calculate Track Quaility Metric
                            trackError = self.calcTrackError(est_prior, cov_prior)
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
        self.delta_alpha = 1
        self.delta_beta = 1
    
        # R Factor
        self.R_factor = 1

        # Define history vectors for each extended Kalman filter
        self.estHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.estPredHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.covarianceHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.covPredHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.trackErrorHist = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.synchronizeFlag = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        self.doET = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
        
    def update_neighbors(self, neighbors):
        '''
        Update the neighbors of the satellite object. Required since the neighbors are not known at initialization.
        '''
        self.neighbors = [self.sat] + neighbors
        # Update history vectors with new neighbors
        for targetID in self.targetIDs:
            self.estHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.estPredHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.covarianceHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.covPredHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.trackErrorHist[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.synchronizeFlag[targetID][self.sat] = {neighbor: {} for neighbor in self.neighbors}
            self.doET = {targetID: {self.sat: {neighbor: {} for neighbor in self.neighbors}} for targetID in self.targetIDs}
            
    
    # def event_triggered_fusion(self, sat, envTime, comms):
    #     commNode = comms.G.nodes[sat]
        
    #     if envTime in commNode['received_measurements']:
    #         self.process_new_measurements(sat, envTime, comms)
        
    #     if envTime in commNode['sent_measurements']: ## This should be true when I have sent measurements to neighbors
    #         self.update_common_filters(sat, envTime, comms)
    
    def event_trigger_processing(self, sat, envTime, comms):
        commNode = comms.G.nodes[sat]
        
        if envTime in commNode['received_measurements']:
            self.process_new_measurements(sat, envTime, comms)
            
    def event_trigger_updating(self, sat, envTime, comms):
        commNode = comms.G.nodes[sat]
        
        if envTime in commNode['sent_measurements']:
            self.update_common_filters(sat, envTime, comms)
                
    def initialize_filter(self, sat, target, envTime, sharewith=None):
    # If no prior estimate exists, initialize with true position plus noise
        targetID = target.targetID
        prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
        prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 1.5, 3)
        est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])
                             
        # Initial covariance matrix
        P_prior = np.array([[2500, 0, 0, 0, 0, 0],
                            [0, 100, 0, 0, 0, 0],
                            [0, 0, 2500, 0, 0, 0],
                            [0, 0, 0, 100, 0, 0],
                            [0, 0, 0, 0, 2500, 0],
                            [0, 0, 0, 0, 0, 100]])
        
        # Store initial values and return for first iteration
        self.estHist[targetID][sat][sharewith][envTime] = est_prior
        self.estPredHist[targetID][sat][sharewith][envTime] = est_prior
        self.covarianceHist[targetID][sat][sharewith][envTime] = P_prior
        self.covPredHist[targetID][sat][sharewith][envTime] = P_prior
        self.trackErrorHist[targetID][sat][sharewith][envTime] = self.calcTrackError(est_prior, P_prior)
        self.synchronizeFlag[targetID][sat][sharewith][envTime] = True                        
        
        
    def update_common_filters(self, sat, envTime, comms):
        commNode = comms.G.nodes[sat]
        time_sent = max(commNode['sent_measurements'].keys()) # Get the newest info on this target
        for targetID in commNode['sent_measurements'][time_sent].keys():
            for i in range(len(commNode['sent_measurements'][time_sent][targetID]['receiver'])): # number of messages on this target?
                receiver = commNode['sent_measurements'][time_sent][targetID]['receiver'][i]
                
                if len(receiver.etEstimator.estHist[targetID][receiver][receiver]) == 1:
                    continue
                
                if not receiver.etEstimator.doET[targetID][receiver][sat]: 
                    continue
                
                if receiver.etEstimator.doET[targetID][receiver][sat][envTime] == True:
                    est_pred = self.estPredHist[targetID][sat][sat][envTime]
                    cov_pred = self.covPredHist[targetID][sat][sat][envTime]

                    # Run Prediction Step on this target for common fitler
                    self.pred_EKF(sat, receiver, targetID, envTime)
                    
                    # Proccess the new measurement from sender with the local and common filter
                    alpha, beta = commNode['sent_measurements'][time_sent][targetID]['meas'][i]
                    if not np.isnan(alpha):
                        self.explicit_measurement_update(sat, sat, alpha, 'IT', targetID, envTime, sharewith=receiver) # update our common filter
                    else:
                        self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'IT', 'common', targetID, envTime, sharewith=receiver)

                    if not np.isnan(beta):
                        self.explicit_measurement_update(sat, sat, beta, 'CT', targetID, envTime, sharewith=receiver) # update our common filter
                    else:
                        self.implicit_measurement_update(sat, sat, est_pred, cov_pred, 'CT', 'common', targetID, envTime, sharewith=receiver)

                    # Calculate Common Track Quaility Metric
                    est = self.estHist[targetID][sat][receiver][envTime]
                    cov = self.covarianceHist[targetID][sat][receiver][envTime]
                    self.trackErrorHist[targetID][sat][receiver][envTime] = self.calcTrackError(est, cov)
                        
    
    def process_new_measurements(self, sat, envTime, comms):
        '''
        This should process new measurements for this satellites commNode and update the local and common filters
        '''
        commNode = comms.G.nodes[sat]
        time_sent = max(commNode['received_measurements'].keys()) # Get the newest info on this target
        for targetID in commNode['received_measurements'][time_sent].keys():
            for i in range(len(commNode['received_measurements'][time_sent][targetID]['sender'])): # number of messages on this target?
                sender = commNode['received_measurements'][time_sent][targetID]['sender'][i]
                alpha, beta = commNode['received_measurements'][time_sent][targetID]['meas'][i]

                if len(sat.etEstimator.estHist[targetID][sat][sat]) == 1: # if your filter was just intialized dont process anything
                    continue
                
                tqReq = int(targetID)*50 + 50
                sat.etEstimator.doET[targetID][sat][sender][envTime] = False
                
                if (self.trackErrorHist[targetID][sat][sat][envTime] > tqReq):
                    sat.etEstimator.doET[targetID][sat][sender][envTime] = True
                
                elif np.isnan(alpha) and np.isnan(beta):
                    sat.etEstimator.doET[targetID][sat][sender][envTime] = True # if there is just implicit information use it
                
                if sat.etEstimator.doET[targetID][sat][sender][envTime]: 
                
                    if self.synchronizeFlag[targetID][sat][sender][envTime] == True: ## TODO: If any of the last 5 
                        self.synchronize_filters(sat, sender, targetID, envTime, comms)
                        continue
                    
                    
                    # Grab the most recent local prediction for the target
                    est_pred = self.estPredHist[targetID][sat][sat][envTime]
                    cov_pred = self.covPredHist[targetID][sat][sat][envTime]
                    
                    # Run Prediction Step on this target for common fitler
                    self.pred_EKF(sat, sender, targetID, envTime)
                    
                    # Proccess the new measurement from sender with the local and common filter
                    measVec_size = 2

                    # In-Track Measurement
                    if not np.isnan(alpha): # TODO: This is wrong, explicit takes wrong local estimate
                        self.explicit_measurement_update(sat, sender, alpha, 'IT', targetID, envTime, sharewith=sat) # update local filter
                        self.explicit_measurement_update(sat, sender, alpha, 'IT', targetID, envTime, sharewith=sender) # update our common filter
                    else:
                        self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'IT', 'both', targetID, envTime, sharewith=sender) # update my local and common filter
                        measVec_size -= 1

                    # Cross-Track Measurement
                    if not np.isnan(beta):
                        self.explicit_measurement_update(sat, sender, beta, 'CT', targetID, envTime, sharewith=sat) # update local filter
                        self.explicit_measurement_update(sat, sender, beta, 'CT', targetID, envTime, sharewith=sender) # update our common filter
                    else:
                        self.implicit_measurement_update(sat, sender, est_pred, cov_pred, 'CT', 'both', targetID, envTime, sharewith=sender) # update my local and common filter
                        measVec_size -= 1

                    # Calculate Local Track Quaility Metric
                    est = self.estHist[targetID][sat][sat][envTime]
                    cov = self.covarianceHist[targetID][sat][sat][envTime]
                    self.trackErrorHist[targetID][sat][sat][envTime] = self.calcTrackError(est, cov)
                    
                        
                    # Calculate Common Track Quaility Metric
                    est = self.estHist[targetID][sat][sender][envTime]
                    cov = self.covarianceHist[targetID][sat][sender][envTime]
                    self.trackErrorHist[targetID][sat][sender][envTime] = self.calcTrackError(est, cov)
                    
                    comms.used_comm_et_data_values[targetID][sat.name][sender.name][time_sent] = np.array([alpha, beta]) 
                    comms.used_comm_et_data[targetID][sat.name][sender.name][time_sent] = measVec_size

       
    def local_et_filter_prediction(self, sat, target, envTime):
        targetID = target.targetID                
        # Run Prediction Step on this target for local fitler
        self.pred_EKF(sat, sat, targetID, envTime) # updates estHist and covarianceHist
        est = self.estHist[targetID][sat][sat][envTime]
        cov = self.covarianceHist[targetID][sat][sat][envTime]
        
        # Store the prediction
        self.estPredHist[targetID][self.sat][sat][envTime] = est
        self.covPredHist[targetID][self.sat][sat][envTime] = cov
        self.trackErrorHist[targetID][sat][sat][envTime] = self.calcTrackError(est, cov)
        
        
    def local_et_filter_meas_update(self, sat, target, envTime):
        '''
        Update the local filter for a target with the most recent measurements.
        
        Args:
        - sat (object): Satellite object.
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        '''
        targetID = target.targetID
        
        if len(sat.measurementHist[targetID][envTime]) == 0:
            return
        
        alpha, beta = sat.measurementHist[targetID][envTime]
                            
        # Proccess my measurement in the local filter
        self.explicit_measurement_update(sat, sat, alpha, 'IT', targetID, envTime, sharewith=sat) 
        self.explicit_measurement_update(sat, sat, beta, 'CT', targetID, envTime, sharewith=sat)
        
         # Calculate Local Track Quaility Metric
        est = self.estHist[targetID][sat][sat][envTime]
        cov = self.covarianceHist[targetID][sat][sat][envTime]
        self.trackErrorHist[targetID][sat][sat][envTime] = self.calcTrackError(est, cov)
                  
                              
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
            self.trackErrorHist[targetID][sat][sender][envTime] = self.calcTrackError(est_prior, P_prior)
            return

        # Predict next state using state transition function
        est_pred = super().state_transition(est_prior, dt)
        
        # Evaluate Jacobian of state transition function
        F = super().state_transition_jacobian(est_prior, dt)
        
        # Predict process noise associated with state transition
        Q = np.diag([0.1, 0.01, 0.1, 0.01, 0.1, 0.01])  # Smaller Q 

        # Predict covariance
        P_pred = np.dot(F, np.dot(P_prior, F.T)) + Q ## TODO check dot products?
        
        # Store the prediction
        self.estHist[targetID][sat][sender][envTime] = est_pred
        self.covarianceHist[targetID][sat][sender][envTime] = P_pred
        self.trackErrorHist[targetID][sat][sender][envTime] = self.calcTrackError(est_pred, P_pred)
        
         
    def explicit_measurement_update(self, sat, sender, measurement, type, targetID, envTime, sharewith):
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
         
         # Save Data into filter
        self.estHist[targetID][sat][sharewith][envTime] = est
        self.covarianceHist[targetID][sat][sharewith][envTime] = P
        self.trackErrorHist[targetID][sat][sharewith][envTime] = self.calcTrackError(est, P)    
                     
        
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
                
        # Compute Kalman Gain
        K = np.reshape(local_cov_curr @ H.T * (H @ local_cov_curr @ H.T + R)**-1, (6,1))
        
        # Update Estimate
        est = local_est_curr + np.reshape(K * zbar, (6))
        
        # Update Covariance
        cov = local_cov_curr - nu * K @ H @ local_cov_curr
    
        # Save Data into both filters
        if update == 'both':
            self.estHist[targetID][sat][sat][envTime] = est
            self.covarianceHist[targetID][sat][sat][envTime] = cov
            self.trackErrorHist[targetID][sat][sat][envTime] = self.calcTrackError(est, cov)

            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sharewith][envTime] = cov
            self.trackErrorHist[targetID][sat][sharewith][envTime] = self.calcTrackError(est, cov)
        
        elif update == 'common':
            self.estHist[targetID][sat][sharewith][envTime] = est
            self.covarianceHist[targetID][sat][sharewith][envTime] = cov
            self.trackErrorHist[targetID][sat][sharewith][envTime] = self.calcTrackError(est, cov)

        
    
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
        
        # # Predict the Jacobian in the measurement
        # H = sat.sensor.jacobian_ECI_to_bearings(sat, pred_est)
        
        # # Compute the Sensor Noise Matrix
        # R = sat.sensor.bearingsError**2 * self.R_factor # Sensor noise matrix scaled by 1000x
        
        # # Compute the Innovation
        # innovation = np.array([alpha, beta]) - np.array([pred_alpha, pred_beta])
        
        # # Compute the Innovation Covariance
        # innovationCov = np.dot(H, np.dot(self.covarianceHist[targetID][sat][neighbor][time], H.T)) + R
        
        # # Compute the Mahalanobis Distance in each component
        # mahalanobis_alpha = np.dot(innovation[0], np.dot((innovationCov[0,0]**-1), innovation[0]))
        # mahalanobis_beta = np.dot(innovation[1], np.dot((innovationCov[1,1]**-1), innovation[1]))
        # mahalanobis = np.dot(innovation.T, np.dot(np.linalg.inv(innovationCov), innovation))
        
        # # Idea 2: Take cholesky decomposition of the innovation covariance matrix and compute the mahalanobis distance
        # eigvalues, eigvectors = np.linalg.eig(innovationCov)
        
        # U = np.array(eigvectors).T
        # V = np.diag(eigvalues)
        
        # F = U @ np.linalg.inv(np.sqrt(V))
        
        # epsilon = np.dot(F.T, innovation)
        
        
        # This computes how many std the measurement was away from the prediction i think
        # So somehow we can back out another way to modify implicit update
        
        # Is my measurment surprising based on the prediction?
        send_alpha = np.nan
        send_beta = np.nan
        
        if neighbor.etEstimator.trackErrorHist[targetID][neighbor][neighbor][time] > int(targetID)*50 + 50:
            send_alpha = alpha
            send_beta = beta
        
        # else:
            
        #     # If the difference between the measurement and the prediction is greater than delta, send the measurement
        #     if np.abs(alpha - pred_alpha) > self.delta_alpha:
        #         send_alpha = alpha
                
        #     if np.abs(beta - pred_beta) > self.delta_beta:
        #         send_beta = beta
            
        # # print("Predicted Alpha: ", pred_alpha, "Predicted Beta: ", pred_beta)
        # # print("Measured Alpha: ", alpha, "Measured Beta: ", beta)
        # # print("Send Alpha: ", send_alpha, "Send Beta: ", send_beta)
        return send_alpha, send_beta
    
    
    def synchronize_filters(self, sat, neighbor, targetID, envTime, comms):
        """
        Synchronize the local and common filters between two agents.

        Args:
        - sat (object): Satellite object that is receiving information.
        - neighbor (object): Neighbor satellite object.
        - targetID (int): Target ID.
        - envTime (float): Current environment time.
        """
        ### Try just synchronizing the common information filters
        
        # Sat common information filter with neighbor
        time12 = max(self.estHist[targetID][sat][sat].keys())
        est12 = self.estHist[targetID][sat][sat][time12]
        cov12 = self.covarianceHist[targetID][sat][sat][time12]
        
        # Neighbor common information filter with sat
        time21 = max(neighbor.etEstimator.estHist[targetID][neighbor][neighbor].keys())
        est21 = neighbor.etEstimator.estHist[targetID][neighbor][neighbor][time21]
        cov21 = neighbor.etEstimator.covarianceHist[targetID][neighbor][neighbor][time21]
        
        omega_opt = minimize(self.det_of_fused_covariance, [0.5], args=(cov12, cov21), bounds=[(0, 1)]).x
        cov_fused = np.linalg.inv(omega_opt * np.linalg.inv(cov12) + (1 - omega_opt) * np.linalg.inv(cov21))
        est_fused = cov_fused @ (omega_opt * np.linalg.inv(cov12) @ est12 + (1 - omega_opt) * np.linalg.inv(cov21) @ est21)
        
        # Save the local filter
        self.estHist[targetID][sat][sat][envTime] = est_fused
        self.covarianceHist[targetID][sat][sat][envTime] = cov_fused
        self.trackErrorHist[targetID][sat][sat][envTime] = self.calcTrackError(est_fused, cov_fused)
        
        # Save the synchronized filter
        self.estHist[targetID][sat][neighbor][envTime] = est_fused
        self.covarianceHist[targetID][sat][neighbor][envTime] = cov_fused
        self.trackErrorHist[targetID][sat][neighbor][envTime] = self.calcTrackError(est_fused, cov_fused)
        
        # Save the neighbor filter
        neighbor.etEstimator.estHist[targetID][neighbor][neighbor][envTime] = est_fused
        neighbor.etEstimator.covarianceHist[targetID][neighbor][neighbor][envTime] = cov_fused
        neighbor.etEstimator.trackErrorHist[targetID][neighbor][neighbor][envTime] = self.calcTrackError(est_fused, cov_fused)
        
        # Neighbor common information filter with sat
        neighbor.etEstimator.estHist[targetID][neighbor][sat][envTime] = est_fused
        neighbor.etEstimator.covarianceHist[targetID][neighbor][sat][envTime] = cov_fused
        neighbor.etEstimator.trackErrorHist[targetID][neighbor][sat][envTime] = self.calcTrackError(est_fused, cov_fused)

        
        comms.used_comm_et_data[targetID][sat.name][neighbor.name][envTime] = 6 + 21
        
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
        
        