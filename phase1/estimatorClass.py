from import_libraries import *
## Creates the estimator class

class centralEstimator:
    def __init__(self, sats, targs): # Takes in both the satellite objects and the targets

    # Define the satellites and targets
        self.sats = sats
        self.targs = targs

        
class localEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targets

    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.stateMeasHist = {targetID: [] for targetID in targetIDs}
        self.estHist = {targetID: [] for targetID in targetIDs} 
        self.covarianceHist = {targetID: [] for targetID in targetIDs}


    def EKF(self, measurementHist, targetID, dt, sat):
        
        # Check if measurements are empty, if so skip
        if len(measurementHist[targetID]) == 0:
            return 0

        # Else, convert the most recent measurement to a spherical coordinate estimate:
        # [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]

    # Get measurement history for that target
        measurements = np.array(measurementHist[targetID])

    # Get estimate history for that target
        estimates = np.array(self.estHist[targetID])

    # Get covert history for that target
        covariance = np.array(self.covarianceHist[targetID])

    # Get last estimate, using spherical coordinates
        # [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
        # Assume prior state estimate exists
        if len(estimates) == 0:
            # If no prior estimate, use the first measurement
            # TODO: Convert to ECI from bearings
            # TODO: ALSO NEED VELOCITY IN STATE, somehow
            state = measurements[0, 3:]
        else:
            # Otherwise use the last estimate
            state = estimates[-1]

    # Get last covariance matrix
        # Assume prior covariance exists
        if len(covariance) == 0:
            # If no prior covariance, use the identity matrix
            P = np.eye(6)
        else:
            # Otherwise use the last covariance
            P = covariance[-1]

# Preciction:
    # USING DWNAM
    # Define the state transition matrix
        F = np.array([[1, dt, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])
        
    # Define the process noise matrix
        # Estimate the randomness of the acceleration, for now just use exact
        q_range = 0.000001
        q_elevation = 0.001
        q_azimuth = 0.001
        q_mat = np.array([0, q_range, 0, q_elevation, 0, q_azimuth])
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]]) * q_mat**2
        
        # Ignore control input for now
        B = np.array([0, 1, 0, 1, 0, 1])
        u = 0

    # Predict the state
        state_pred = np.dot(F, state) + np.dot(B, u)
        P_pred = np.dot(F, np.dot(P, F.T)) + Q

# Update:
    # Solve for the Kalman gain

        H = np.eye(6)
        # H = sensor.H

        R = np.eye(6) * 0.01
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_pred, H.T)) + R)))

    # Get the measurement
        z = measurements[-1, 3:]

    # Update the state
        state = state_pred + np.dot(K, z - np.dot(H, state_pred))
        P = P_pred - np.dot(K, np.dot(H, P_pred))

    # Save the estimate and covariance
        self.estHist[targetID].append(state) # Will be in spherical coordinates
        self.covarianceHist[targetID].append(P)

    # Return the estimate
        # Translate from spherical to ECI
        pos = np.array([state[0]*np.cos(state[4])*np.sin(state[2]), state[0]*np.sin(state[4])*np.sin(state[2]), state[0]*np.cos(state[2])])

        return pos
