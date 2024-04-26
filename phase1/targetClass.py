from import_libraries import *
## Creates the target class

class target:
    def __init__(self, name, targetID, pos, vel, color):
        # Intial target ID
        self.targetID = targetID
        self.name = name
        self.color = color
        
        # Set time to 0
        self.time = 0
        
        # Change to intial speed and heading
        
        # Initial State X = [x y z vx vy vz]' relative to 0,0,0
        self.pos = np.array(pos)
        self.hist = []
        
        self.vel = np.array(vel)
        self.speed = np.linalg.norm(self.vel)
                
        # Spherical Cordinates for Propagation
        self.theta = np.arctan2(self.pos[1],self.pos[0]) #+ 2*np.pi # positive because omega*t below makes negative time sometimes
        self.inc = np.arcsin(self.pos[2]/6378.0) #+ 2*np.pi
        
       
    def propagate(self, time_step, time):         
        # TimeStep
        dt = time_step.value
        t = time.value
        # Radius of Earth
        r = 6378.0
        
        # Constant Angular Rate
        omega = self.vel/r
        
        rNum = np.random.uniform(0,1)
        thresh = 0.2
        xNoise = 0
        yNoise = 0
        zNoise = 0
        
        if (rNum < thresh):
            xNoise = np.random.uniform(-0.2,0.2)
            yNoise = np.random.uniform(-0.2,0.2)
            zNoise = np.random.uniform(-0.2,0.2)
        
        
        # Assume Polar Path in YZ plane
            # Update to 3D Path, maybe consider an orbit with a = r?
        x = 0 + xNoise                       # r*np.cos(omega[0]*t)#self.theta) # circular movement
        y = r*np.sin(omega[1]*t) + yNoise
        z = r*np.cos(omega[2]*t) + zNoise
        
        self.pos = r * np.array([x,y,z])/np.linalg.norm(np.array([x,y,z])) # constrain to surface
        
        
        # print("ID", self.targetID)
        # print("Position", self.pos)
        # print("theta", self.theta)
        # print("phi", self.phi)
        
        
        
        return self.pos