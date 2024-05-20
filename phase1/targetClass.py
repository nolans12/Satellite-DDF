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
        self.hist = []
        self.fullHist = []
        
        #  Target State r = [range, evlevation, azimuth, range rate, elevation rate, azimuth rate]'
        self.r = np.array(r)
        
    def propagate(self, time_step, time):
        # TimeStep
        dt = time_step.value
        t = time.value
        
        # Add randomnes to the target position
        rNum = np.random.uniform(0,1)
        thresh = 0.2
        xNoise = 0
        yNoise = 0
        zNoise = 0
        if (rNum < thresh):
            xNoise = np.random.uniform(-0.2,0.2)
            yNoise = np.random.uniform(-0.002,0.002)
            zNoise = np.random.uniform(-0.002,0.002)
            
        # self.r intial range, elevation, azimuth
        currRange = self.r[0]
        currElevation = self.r[1]
        currAzimuth =  self.r[2]
        
        # Linear Equation of Motion around sphere
        rangeRate = 0
        elevationRate = 0.0005 + yNoise
        azimuthRate = 0.0005 + zNoise 
        
        newRange = currRange + rangeRate*dt
        newElevation = currElevation + elevationRate*dt
        newAzimuth = currAzimuth + azimuthRate*dt
        
        self.r = np.array([newRange, newElevation, newAzimuth, rangeRate, elevationRate, azimuthRate])
        self.pos = np.array([newRange*np.cos(newAzimuth)*np.sin(newElevation), newRange*np.sin(newAzimuth)*np.sin(newElevation), newRange*np.cos(newElevation)])
        # print("ID", self.targetID)
        # print("Position", self.pos)
        # print("theta", self.theta)
        # print("phi", self.phi)
        return self.pos