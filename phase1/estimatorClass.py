from import_libraries import *
## Creates the estimator class

class centralEstimator:
    def __init__(self, sats, targs): # Takes in both the satellite objects and the targets

    # Define the satellites and targets
        self.sats = sats
        self.targs = targs

    # TODO: USE NESTED DICT STRUCTURE AS SEEN IN LOCAL ESTIMATOR:
        
class localEstimator:
    def __init__(self, targetIDs): # Takes in both the satellite objects and the targets

    # Define the targets to track
        self.targs = targetIDs

    # Define history vectors for each extended kalman filter
        self.measHist = {targetID: defaultdict(dict) for targetID in targetIDs}
        self.estHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates
        self.covarianceHist = {targetID: defaultdict(dict) for targetID in targetIDs} # Will be in ECI coordinates


    # Input: New measurement: estimate in ECI, target ID, time step, and environment time (to time stamp the measurement)
    # Output: New estimate in ECI
    def EKF(self,  newMeas, targetID, dt, envTime):
    # Extended Kalman Filter:
    # Desired Estimate Xdot = [x xdot y ydot z zdot]
    # Measurment Z = [x y z] ECI coordinates
    # Assume CWNA predictive model for dynamics and covariance
    
    # Get measurement time history for that target
        measurments = self.measHist[targetID]

    # Get estimate time history for that target
        estimates = self.estHist[targetID]

    # Get covert time history for that target
        covariance = self.covarianceHist[targetID]

        # Assume prior state estimate exists
        if len(estimates) == 0:
            # If no prior estimate, use the first measurement and assume no velocity
            state = np.array([newMeas[0], 0, newMeas[1], 0, newMeas[2], 0])
        else:
            # To get the last estimate, need to get the last time, which will be the max
            lastTime = max(estimates.keys())
            state = estimates[lastTime]

    # Get last covariance matrix
        # Assume prior covariance exists
        if len(covariance) == 0:
            # If no prior covariance, use the identity matrix
            P = np.eye(6)
        else:
            # To get the last covariance, need to get the last time, which will be the max
            lastTime = max(covariance.keys())
            P = covariance[lastTime]

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
        # Estimate the randomness of the acceleration
        q_x = 0.001
        q_y = 0.001
        q_z = 0.001
        q_mat = np.array([0, q_x, 0, q_y, 0, q_z])
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

        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0]])
        # H = sensor.H

        R = np.eye(3) * 0.01
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_pred, H.T)) + R)))

    # Get the measurement
        z = newMeas

    # Update the state
        state = state_pred + np.dot(K, z - np.dot(H, state_pred))
        P = P_pred - np.dot(K, np.dot(H, P_pred))

    # Save the estimate and covariance
        self.estHist[targetID][envTime] = state # x, xdot, y, ydot, z, zdot in ECI coordinates
        self.covarianceHist[targetID][envTime] = P
        self.measHist[targetID][envTime] = newMeas

    # # Return the estimate
    #     # Translate from spherical to ECI
    #     pos = np.array([state[0]*np.cos(state[4])*np.sin(state[2]), state[0]*np.sin(state[4])*np.sin(state[2]), state[0]*np.cos(state[2])])

        return state