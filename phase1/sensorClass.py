from import_libraries import *
# Sensor Class

class sensor:
    def __init__(self, fov, resolution, sensorError, name):
        self.fov = fov
        self.targetMeasurement = []
        self.sensorError = sensorError
        self.name = name
        self.resolution = resolution
        self.time = 0

    
    def sensor_model(self, sat, targ):
        # Takes the target measurement, adds noise, and returns the angle of the target in the sensor frame
        projVec = sat.visible_projection()

        # Distance Covered By Image
        dX = np.abs(projVec[0][0] - projVec[1][0])
        dY = np.abs(projVec[0][1] - projVec[1][1])
        
        # Resolution of Image
        sX = dX/self.resolution # km/pixel
        sY = np.abs(projVec[0][1] - projVec[1][1])/self.resolution # km/pixel
        
        # Use target location in image frame
        xtarg, ytarg, ztarg = targ.pos
        
        # Determine Pixel Location with some noise
        xTargetPix = np.abs(xtarg - projVec[1][0])/sX   + np.random.normal(0, self.sensorError)
        yTargetPix = np.abs(ytarg - projVec[1][1])/sY + np.random.normal(0, self.sensorError)
        
        # Determine angles from center of image
        xtarg_Angle = (xTargetPix/self.resolution)*self.fov - self.fov/2
        ytarg_Angle = (yTargetPix/self.resolution)*self.fov - self.fov/2
        
        #print(xtarg_Angle, ytarg_Angle)
        return [xtarg_Angle, ytarg_Angle]