from import_libraries import *
## Creates the target class

class target:
    def __init__(self, speed, x0, name, color):
        self.name = name
        self.color = color
        self.speed = speed
        self.x0 = x0 # xyz position relative to 0, 0, 0
        self.x = x0
        self.hist = []
        self.v = np.array([0, 0, 0])
        self.time = 0

    def propagate(self, time_step):
        
        # Create random variable
            # Check if that random variable is less than a certain amounnt
        
        return self.x # for now assume no change in position