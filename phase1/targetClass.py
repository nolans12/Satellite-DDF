from import_libraries import *


# Target class that moves linearly around the earth with constant velocity
# Inputs: Name, TargetID, Initial Position, Heading, Speed, Color
class target:
    def __init__(self, name, targetID, cords, heading, speed, color):
        # Intial target ID, name, color
        self.targetID = targetID
        self.name = name
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # Desired State X = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]' -> [km, km/s, rad, rad/s, rad, rad/s]
        # Input: Cords = [lat lon alt] -> [deg, deg, km]
        #        Heading = Deg CW from North
        #        Speed = Speed km/min           add climb rate later
                
        range = cords[2] + 6378 # range from center of earth km
        rangeRate = 0 # constant altitude
        
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
        
        # Convert back to x,y,z ECI
        x = range*np.sin(azimuth)*np.cos(elevation)
        y = range*np.sin(azimuth)*np.sin(elevation)
        z = range*np.cos(elevation)
        
        self.pos = np.array([x, y, z])
        
        # Convert back to xdot, ydot, zdot ECI
        vx = (rangeRate * np.sin(azimuth) * np.cos(elevation) 
               + range * np.cos(azimuth) * np.cos(elevation) * azimuthRate 
               - range * np.sin(azimuth) * np.sin(elevation) * elevationRate)
        
        vy = (rangeRate * np.sin(azimuth) * np.sin(elevation) 
               + range * np.cos(azimuth) * np.sin(elevation) * azimuthRate 
               + range * np.sin(azimuth) * np.cos(elevation) * elevationRate)
        
        vz = (rangeRate * np.cos(azimuth) 
               - range * np.sin(azimuth) * azimuthRate)
        
        self.vel = np.array([vx, vy, vz])