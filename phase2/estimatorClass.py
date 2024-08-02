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
            # If not prior estimate exists:
            
            # TODO: Eventually should initialize the kalman filter with the real of what target was sampled from
            # # If the environment time == 0, initialize the filter with what the target was sampled from
            # if envTime == 0:
            #     test = 1

            # # If the environment time is not 0, just initalize the filter with the true position plus some noise
            # else:
            # Initialize with true position plus noise
            prior_pos = np.array([target.pos[0], target.pos[1], target.pos[2]]) + np.random.normal(0, 15, 3)
            prior_vel = np.array([target.vel[0], target.vel[1], target.vel[2]]) + np.random.normal(0, 5, 3)
            est_prior = np.array([prior_pos[0], prior_vel[0], prior_pos[1], prior_vel[1], prior_pos[2], prior_vel[2]])

            # Initial covariance matrix
            P_prior = np.array([[1000, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 1000, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 1000, 0],
                                [0, 0, 0, 0, 0, 100]])

            # Store initial values and return for first iteration
            self.estHist[targetID][envTime] = est_prior
            self.covarianceHist[targetID][envTime] = P_prior
            self.innovationHist[targetID][envTime] = np.zeros(2)
            self.innovationCovHist[targetID][envTime] = np.eye(2)
            self.nisHist[targetID][envTime] = 0
            self.neesHist[targetID][envTime] = 0
            self.trackErrorHist[targetID][envTime] = self.calcTrackQuailty(est_prior, P_prior)
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
            P_prior = np.array([[1000, 0, 0, 0, 0, 0],
                                [0, 100, 0, 0, 0, 0],
                                [0, 0, 1000, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 1000, 0],
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
            R[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * sat.sensor.bearingsError**2 * self.R_factor  # Sensor noise matrix scaled by X amount
            
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
        trackError = self.calcTrackQuailty(est, P)

        # Save data
        self.estHist[targetID][envTime] = est
        self.covarianceHist[targetID][envTime] = P
        self.innovationHist[targetID][envTime] = np.reshape(innovation, (numMeasurements))
        self.innovationCovHist[targetID][envTime] = innovationCov
        self.neesHist[targetID][envTime] = nees
        self.nisHist[targetID][envTime] = nis
        self.trackErrorHist[targetID][envTime] = trackError
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
        # Since we are guaranteeing that the state will always live within the covariance estimate
        # Why not just use the diag of the covariance matrix, then calculate the 2 sigma bounds of xyz
        # Then take norm of xyz max as the error

        # Calculate the 2 sigma bounds of the position
        diag = np.sqrt(np.diag(cov))

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

### DDF Estimator Class
class ddfEstimator(BaseEstimator):
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
                
                # Calculate Track Quaility Metric
                trackError = self.calcTrackQuailty(est_prior, cov_prior)
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
