from import_libraries import *


# Target class that moves linearly around the earth with constant velocity
# Inputs: Name, TargetID, Initial Position, Heading, Speed, Color
class target:
    def __init__(self, name, targetID, coords, heading, speed, color, uncertainty = np.array([0, 0, 0, 0, 0, 0]), climbrate = 0, changeAoA = False):

        # Initialize the targets parameters
        self.targetID = targetID
        self.name = name
        self.color = color
        self.time = 0

        # Now convert the spherical coordinates and heading into a state vector
        # Want state = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]
            # range = distance from center of earth
            # rangeRate = speed in direction of heading
            # elevation = angle from equator, towards north pole is positive
            # elevationRate = speed in direction of elevation
            # azimuth = angle from prime meridian
            # azimuthRate = speed in direction of azimuth
        
        range = coords[2] + 6378 # range from center of earth km
        rangeRate = climbrate # constant altitude
        self.changaAoA = changeAoA # True if target should change Angle of Attack 
        
        elevation = np.deg2rad(coords[0]) # [rad] latitude where 0 is equator
        azimuth = np.deg2rad(coords[1]) # [rad] longitude where 0 prime meridian
        
        # Angular Rates are speed in direction of heading
        elevationRate = speed/range*np.cos(np.deg2rad(heading)) # [rad/min]
        azimuthRate = speed/range*np.sin(np.deg2rad(heading)) # [rad/min]

        # Initalize the mean (guess of target position and velocity for the kalman filter)
        self.initialGuess = np.array([range, rangeRate, elevation, elevationRate, azimuth, azimuthRate])

        # Set the covariance (uncertainty in the initial guess)
        # Defined by the uncertainty input
        self.covariance = np.array([[uncertainty[0]**2, 0, 0, 0, 0, 0],
                                    [0, uncertainty[1]**2, 0, 0, 0, 0],
                                    [0, 0, np.deg2rad(uncertainty[2])**2, 0, 0, 0],
                                    [0, 0, 0, np.deg2rad(uncertainty[3])**2, 0, 0],
                                    [0, 0, 0, 0, np.deg2rad(uncertainty[4])**2, 0],
                                    [0, 0, 0, 0, 0, np.deg2rad(uncertainty[5])**2]])

        # Now define the initial state vector X by sampling from the initalGuess and covariance
        self.X = np.random.multivariate_normal(self.initialGuess, self.covariance)

        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI [x xdot y ydot z zdot]
        
    def propagate(self, time_step, time, initialGuess = None):
        # Linearly Propagate Target State in Spherical Cords then transform back to ECI
        # x = [r, r', e, e', a, a'] = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate
        # xDot = [r', r'', e', e'', a', a''] = Ax 
        # xNew = x + xDot*dt

        # if dt is a float64, dont have to convert
        if isinstance(time_step, (float, np.float64)):
            dt = time_step
        else:
            dt = time_step.value
        
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        #B = np.array([0, 1, 0, 0, 0, 0]).T # Assume control input can only change angle of attack
        # TODO: Add control input to change angle of attack
        if (self.changaAoA):
            # Change angle of attack every 10 minutes
            if (time % 10 == 0):
                # Change angle of attack
                self.X[1] = -self.X[1]
            
        if initialGuess is not None:
            X = initialGuess
        else:
            X = self.X
        
        # Assume no control input, then new state = prev state + xdot*dt
        x_dot = np.dot(A, X)
        X = X + x_dot*dt
        self.X = X
        
        # Extract for reading simplicity
        range = X[0]
        rangeRate = X[1]
        elevation = X[2]
        elevationRate = X[3]
        azimuth = X[4]
        azimuthRate = X[5]
 
        # Convert Spherical to Cartesian
        x = range * np.cos(elevation) * np.cos(azimuth)
        y = range * np.cos(elevation) * np.sin(azimuth)
        z = range * np.sin(elevation)

        # Approximate velocities conversion (simplified version)
        vx = rangeRate * np.cos(elevation) * np.cos(azimuth) - \
            range * elevationRate * np.sin(elevation) * np.cos(azimuth) - \
            range * azimuthRate * np.cos(elevation) * np.sin(azimuth)

        vy = rangeRate * np.cos(elevation) * np.sin(azimuth) - \
            range * elevationRate * np.sin(elevation) * np.sin(azimuth) + \
            range * azimuthRate * np.cos(elevation) * np.cos(azimuth)

        vz = rangeRate * np.sin(elevation) + \
            range * elevationRate * np.cos(elevation)

        self.pos = np.array([x, y, z])# + np.random.normal(0, 15.5, 3)
        self.vel = np.array([vx, vy, vz])# + np.random.normal(0, 5.5, 3)

        # print("Target Position: ", self.pos)
        # print("Target Velocity: ", self.vel)

        if initialGuess is not None:
            # If initial guess is passed into propegate, then return the updated state guess
            return np.array([x, y, z, vx, vy, vz])
        