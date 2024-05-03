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

        
    def in_fov(self, sat, targ):

    # Get all x values of the projection box
        x_vals = sat.projBox[:, 0]
        y_vals = sat.projBox[:, 1]
        z_vals = sat.projBox[:, 2]
        z_tolerance = 0.05 * 6378  # 5% of earth radius, so curvature of earth is accounted for

    # Now, just check if the targets xyz is within the xyz of the projBox
        if (targ.x[0] > min(x_vals) and targ.x[0] < max(x_vals) and targ.x[1] > min(y_vals) and targ.x[1] < max(y_vals) and min(z_vals) - z_tolerance < targ.x[2] and targ.x[2] < max(z_vals) + z_tolerance):
            return True
        else:
            return False
        
    def sensor_model(self, sat, targetMeasurement):
        # Assume sensor is pointed directly at center of earth
        x_sat, y_sat, z_sat = sat.orbit.r.value
        r = np.linalg.norm([x_sat, y_sat, z_sat])
        
        # Use target location in image frame        
        xtarg, ytarg = targetMeasurement
        xtarg_fromCP = (xtarg - self.resolution/2)/self.resolution
        ytarg_fromCP = (ytarg - self.resolution/2)/self.resolution
        
        xtarg_angle = xtarg_fromCP * self.fov
        ytarg_angle = ytarg_fromCP * self.fov
        # x_targ = x_targ + np.random.normal(0, self.sensorError)
        # y_targ = y_targ + np.random.normal(0, self.sensorError)
        # z_targ = z_targ + np.random.normal(0, self.sensorError)
        print(sat.name, "views", targetMeasurement)
        print(xtarg_angle, ytarg_angle)
        
        return [xtarg_angle, ytarg_angle]