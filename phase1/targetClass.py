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
        self.hist = defaultdict(dict) # contains time and xyz and velocity history in ECI
        
        #  Target State r = [range, rangeRate, elevation, elevationRate, azimuth, azimuthRate]'
        self.r = np.array(r)
        
    def propagate(self, time_step, time):
        # Input: Target Current State,
        # TimeStep
        dt = time_step.value
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
        
        # 50% chance of acceleration
        if(np.random.uniform(0,1) < 0.5):
            acceleration = np.random.multivariate_normal(mean, Q, 1).T
        else:
            acceleration = np.array([0, 0, 0, 0, 0, 0]).T
            
        # Propagate the state
        rDot = np.dot(A, self.r) + np.dot(B, acceleration)
        self.r = self.r + rDot*dt
        
        # Convert back to x,y,z ECI
        self.pos = np.array([self.r[0]*np.cos(self.r[4])*np.sin(self.r[2]), self.r[0]*np.sin(self.r[4])*np.sin(self.r[2]), self.r[0]*np.cos(self.r[2])])
        x_dot = ( self.r[1] * np.cos( self.r[4]) * np.cos( self.r[2])
             -  self.r[0] * np.sin( self.r[4]) *  self.r[5] * np.cos( self.r[2])
             -  self.r[0] * np.cos( self.r[4]) * np.sin( self.r[2]) *  self.r[3])
    
        y_dot = ( self.r[1] * np.cos( self.r[4]) * np.sin( self.r[2])
             -  self.r[0] * np.sin( self.r[4]) *  self.r[5] * np.sin( self.r[2])
             +  self.r[0] * np.cos( self.r[4]) * np.cos( self.r[2]) *  self.r[3])
    
        z_dot = ( self.r[1] * np.sin( self.r[4])
             +  self.r[0] * np.cos( self.r[4]) *  self.r[5])
        
        self.vel = np.array([x_dot, y_dot, z_dot])