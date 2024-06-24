from import_libraries import *


# Target class that moves linearly around the earth with constant velocity
# Inputs: Name, TargetID, Initial Position, Heading, Speed, Color
class target:
    def __init__(self, name, targetID, cords, heading, speed, climbrate, color, changeAoA = False):
        # Intial target ID, name, color
        self.targetID = targetID
        self.name = name
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # Desired State X = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]' -> [km, km/s, rad, rad/s, rad, rad/s]
        # Input: Cords = [lat lon alt] -> [deg, deg, km]
        #        Heading = Deg CW from North
        #        Speed = Speed km/min           
        #        ClimbRate = Rate of climb km/min
                
        range = cords[2] + 6378 # range from center of earth km
        rangeRate = climbrate # constant altitude
        self.changaAoA = changeAoA # True if target should change Angle of Attack 
        
        elevation = np.deg2rad(cords[0]) # [rad] latitude where 0 is equator
        azimuth = np.deg2rad(cords[1]) # [rad] longitude where 0 prime meridian
        
        # Angular Rates are speed in direction of heading
        elevationRate = speed/range*np.cos(np.deg2rad(heading)) # [rad/min]
        azimuthRate = speed/range*np.sin(np.deg2rad(heading)) # [rad/min]
        
        self.X = np.array([range, rangeRate, elevation, elevationRate, azimuth, azimuthRate])
        
        # Initialize ECI Position History
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI [x xdot y ydot z zdot]
        
    def propagate(self, time_step, time):
        # Linearly Propagate Target State in Spherical Cords then transform back to ECI
        # x = [r, r', e, e', a, a'] = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate
        # xDot = [r', r'', e', e'', a', a''] = Ax 
        # xNew = x + xDot*dt
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
            
        
        # Assume no control input, then new state = prev state + xdot*dt
        x_dot = np.dot(A, self.X)
        self.X = self.X + x_dot*dt
        
        # Extract for reading simplicity
        range = self.X[0]
        rangeRate = self.X[1]
        elevation = self.X[2]
        elevationRate = self.X[3]
        azimuth = self.X[4]
        azimuthRate = self.X[5]
        
        # print("Target: Range: ", range, "Range Rate: ", rangeRate, "Elevation: ", elevation, "Elevation Rate: ", elevationRate, "Azimuth: ", azimuth, "Azimuth Rate: ", azimuthRate)
        
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
        
        self.pos = np.array([x, y, z])  
        self.vel = np.array([vx, vy, vz])

        # print("Norm of velocity: ", np.linalg.norm(self.vel))
        
        