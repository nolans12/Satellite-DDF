from import_libraries import *
## Creates the estimator class

class estimator:
    def __init__(self, sats, targs, sensor): # Takes in both the satellite objects and the targets

    # Define the satellites and targets
        self.sats = sats
        self.targs = targs
        self.sensor = sensor

        self.time = 0

    # Define the estimator data packet, will be a list of lists, for each satellite, a list of the targets it can see and their xyz, t position
        self.rawEstimation = np.zeros((len(self.sats), len(self.targs), 4))

# Loop through each satellite and get a raw xyz estimate of the target for the given time step
    def estimate_raw(self): 
        for s, sat in enumerate(self.sats):
            x_sat, y_sat, z_sat = sat.orbit.r.value
            r_sat = np.linalg.norm([x_sat, y_sat, z_sat])

            for t, targ in enumerate(self.targs):
                x_targ, y_targ, z_targ = targ.x

                if self.in_fov(sat, targ):
                    
                    # FIX THIS
                    #targMeasurement = np.array([x_targ-x_sat,y_targ-y_sat]) * self.sensor.resolution/((r_sat-6378)*np.tan(np.deg2rad(self.sensor.fov/2)))
# IMPLEMENT SENSOR ERROR HERE
                    #xtargAngle, ytargAngle = self.sensor.sensor_model(sat, targMeasurement)

                    x_targ = x_targ + np.random.normal(0, sat.sensorError)
                    y_targ = y_targ + np.random.normal(0, sat.sensorError)
                    z_targ = z_targ + np.random.normal(0, sat.sensorError)

                    self.rawEstimation[s][t] = [x_targ, y_targ, z_targ, self.time]
                    # print(sat.name, "views", targ.name)

                else:
                    self.rawEstimation[s][t] = [0, 0, 0, self.time]

            sat.estimateHist.append(self.rawEstimation[s].copy())


# Returns T/F if target is within the fov of the satellite
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
                
        
        
