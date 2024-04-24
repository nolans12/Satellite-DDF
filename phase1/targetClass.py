from import_libraries import *
## Creates the environment class, which contains a vector of satellites all other parameters

class target:
    def __init__(self, speed, x0):
        self.speed = speed
        self.x0 = x0 # xyz position relative to 0, 0, 0
        self.x = x0
        self.hist = [x0]
        self.v = np.array([0, 0, 0])


    def propagate(self, time_step):
        
        # Create random variable
            # Check if that random variable is less than a certain amounnt
            

        # self.x = ... 
