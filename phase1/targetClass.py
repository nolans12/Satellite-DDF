from import_libraries import *
## Creates the target class
class target:
    def __init__(self, name, targetID, r, color):
        # Intial target ID
        self.targetID = targetID
        self.name = name
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # ECI Position
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI [x xdot y ydot z zdot]
        
        #  Target State r = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]'
        self.r = np.array(r)
        
    def propagate(self, time_step, time):
        # TODO: Make targets follow CWNA model but then constrain z to surface make this in terms for x xdot y ydot z zdot
        
        # Input: Target Current State,
        # TimeStep
        dt = time_step.value

        # print("Integrating target with timestep: ", dt)

        t = time.value
        
        # White Noise Intensity Vector -> should be order of maximim magnitude acceleration over dt    
        rangeAccelNoise = 0.000001
        elevationAccelNoise = 0.001
        azimuthAccelNoise = 0.001
        q = np.array([0, rangeAccelNoise, 0, elevationAccelNoise, 0, azimuthAccelNoise]).T
        
        # Define Continuous White Noise State Equation
        
        # State Dynamics Matrix
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        
        # Control Input Matrix --> Acceleration
        B = np.array([0, 1, 0, 1, 0, 1]).T
        
        # Covariance Matrix
        Q = np.array([[0, 0, 0, 0, 0, 0],
                      [0, dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, dt]])*q**2        
        
        # Sample MultiVariate White Noise as input to State Equation
        mean = np.array([0, 0, 0, 0, 0, 0])
        
        # 10% chance of acceleration
        if(np.random.uniform(0,1) < 0):
            acceleration = np.random.multivariate_normal(mean, Q, 1).T
        else:
            acceleration = np.array([0, 0, 0, 0, 0, 0]).T

        # TODO: try to lower the process noise
        # TODO: simulate the truth with no process noise
        # TODO: plot monte carlo sensing estimates
        # TODO: for now only have static target or just one static target
            

        # Propagate the state
        rDot = np.dot(A, self.r) + np.dot(B, acceleration)
        self.r = self.r + rDot*dt
        # self.r = self.r
        
        # Extract for reading simplicity
        range = self.r[0]
        rangeRate = self.r[1]
        elevation = self.r[2]
        elevationRate = self.r[3]
        azimuth = self.r[4]
        azimuthRate = self.r[5]
        
        # Extract for reading simplicity
        range = self.r[0]
        rangeRate = self.r[1]
        elevation = self.r[2]
        elevationRate = self.r[3]
        azimuth = self.r[4]
        azimuthRate = self.r[5]
        
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