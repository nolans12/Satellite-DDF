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
        
        # Theta and Phi New assuming same angular rate
        # self.theta = self.theta + omega*dt
        # self.inc = self.inc + omega*dt
        
        # New [x,y,z] position of Target
        x = 0#r*np.cos(omega[0]*t)#self.theta) # circular movement
        y = r*np.sin(omega[1]*t)#self.theta)
        z = r*np.cos(omega[2]*t)#self.inc) # with inclination
        
        self.pos = r * np.array([x,y,z])/np.linalg.norm(np.array([x,y,z])) # constrain to surface
        
        
        # print("ID", self.targetID)
        # print("Position", self.pos)
        # print("theta", self.theta)
        # print("phi", self.phi)
        
        
        
        return self.pos
